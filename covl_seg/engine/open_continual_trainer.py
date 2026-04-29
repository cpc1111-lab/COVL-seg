from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_log = logging.getLogger(__name__)

from covl_seg.continual.methods import build_continual_method
from covl_seg.continual.task_partition import TaskDef, TaskPlan, build_task_plan
from covl_seg.engine.balanced_controller import (
    BalancedControllerConfig,
    BalancedControllerState,
    update_controller_state,
)
from covl_seg.engine.detectron2_runner import (
    build_covl_training_model,
    _d2_project_root,
    _resolve_experiment_spec,
    _resolve_class_names,
    _select_base_eval_spec,
    run_detectron2_eval,
    run_detectron2_train,
)
from covl_seg.engine.evaluator import compute_basic_miou
from covl_seg.engine.hooks import append_metrics_jsonl
from covl_seg.engine.mock_continual_runner import infer_num_classes_from_config as _infer_num_classes_from_config_mock
from covl_seg.engine.mock_training_loop import run_mock_task_training
from covl_seg.engine.phase_runner import (
    run_phase1_hciba,
    run_phase2_joint,
    run_phase3_subspace_and_fusion,
    run_phase4_replay_update,
)
from covl_seg.model.covl_seg_model_new import COVLSegModelV2
from covl_seg.data.datasets import ADE20KDataset, COCOStuffDataset, SegmentationAugmentation, SegmentationEvalTransform, COCO_STUFF_164_CLASSES, ClassFilteredDataset
from covl_seg.continual.ewc import EWCRegularizer
from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer
from covl_seg.continual.spectral_ogp import compute_gradient_basis, unflatten_and_project, hard_project_gradient, flatten_gradients


def _infer_num_classes_from_config(config_path: str, *, strict: bool) -> int:
    if strict:
        return int(_resolve_experiment_spec(config_path)["num_classes"])
    try:
        return int(_resolve_experiment_spec(config_path)["num_classes"])
    except ValueError:
        return int(_infer_num_classes_from_config_mock(config_path))


def _clip_overrides(mode: str) -> List[str]:
    lowered = mode.lower()
    if lowered not in {"none", "attention", "full"}:
        raise ValueError(f"Unsupported clip finetune mode: {mode}")
    return ["MODEL.SEM_SEG_HEAD.CLIP_FINETUNE", lowered]


def _write_task_class_indexes(task_dir: Path, task: TaskDef) -> Dict[str, Path]:
    split_dir = task_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_indexes = split_dir / "seen_indexes.json"
    old_indexes = split_dir / "old_indexes.json"
    new_indexes = split_dir / "new_indexes.json"
    test_indexes = split_dir / "unseen_indexes.json"
    new_set = set(task.new_classes)
    old_values = [idx for idx in task.seen_classes if idx not in new_set]
    train_indexes.write_text(json.dumps(task.seen_classes, indent=2), encoding="utf-8")
    old_indexes.write_text(json.dumps(old_values, indent=2), encoding="utf-8")
    new_indexes.write_text(json.dumps(task.new_classes, indent=2), encoding="utf-8")
    test_indexes.write_text(json.dumps(task.background_classes, indent=2), encoding="utf-8")
    return {"train": train_indexes, "old": old_indexes, "new": new_indexes, "test": test_indexes}


def _write_task_class_artifacts(task_dir: Path, task: TaskDef, class_names: List[str]) -> Dict[str, Path]:
    class_index_paths = _write_task_class_indexes(task_dir=task_dir, task=task)
    split_dir = task_dir / "splits"
    train_class_names = split_dir / "train_class_names.json"
    test_class_names = split_dir / "test_class_names.json"
    required_indexes = set(task.seen_classes + task.new_classes + task.background_classes)
    max_index = max(required_indexes, default=-1)
    resolved_names = list(class_names)
    if max_index >= len(resolved_names):
        _log.warning(
            "Resolved class taxonomy is shorter than task split indexes; "
            "padding taxonomy with synthetic class names "
            "for task %s (max_index=%s, taxonomy_size=%s)",
            task.task_id,
            max_index,
            len(resolved_names),
        )
        for idx in range(len(resolved_names), max_index + 1):
            resolved_names.append(f"class_{idx}")
    train_class_names.write_text(
        json.dumps(resolved_names, indent=2),
        encoding="utf-8",
    )
    test_class_names.write_text(
        json.dumps(resolved_names, indent=2),
        encoding="utf-8",
    )
    return {
        "train_indexes": class_index_paths["train"],
        "old_indexes": class_index_paths["old"],
        "new_indexes": class_index_paths["new"],
        "test_indexes": class_index_paths["test"],
        "train_class_json": train_class_names,
        "test_class_json": test_class_names,
    }


def _resolve_task_class_names(config_path: str) -> List[str]:
    class_json = _select_base_eval_spec(config_path)["class_json"]
    try:
        project_root = _d2_project_root()
    except RuntimeError:
        project_root = Path(__file__).resolve().parents[2]
    class_names = _resolve_class_names(
        project_root=project_root,
        class_json=class_json,
    )
    if class_names:
        return class_names
    raise ValueError(
        "Unable to resolve class names for continual runtime artifacts from "
        f"{class_json}. Add the dataset taxonomy JSON instead of using synthetic fallback names."
    )


def _build_phase_batch(task: TaskDef) -> Dict[str, object]:
    features_c = [float((c % 17) / 17.0) for c in task.seen_classes[:32]] or [0.0]
    features_s = [float((c % 19) / 19.0) for c in task.new_classes[:32]] or [0.0]
    targets = [float((c % 13) / 13.0) for c in task.seen_classes[:32]] or [0.0]
    bg_logits = [
        [float(((c + i) % 11) / 11.0) for i in range(5)]
        for c in (task.background_classes[:8] or [0])
    ]
    return {
        "features_c": features_c,
        "features_s": features_s,
        "targets": targets,
        "bg_logits": bg_logits,
    }


def _build_phase_batch_from_d2_metrics(task_dir: Path, fallback_batch: Dict[str, object]) -> Dict[str, object]:
    metrics_path = task_dir / "metrics.json"
    if not metrics_path.exists():
        return fallback_batch
    losses: List[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_loss = payload.get("total_loss")
        if total_loss is None:
            continue
        losses.append(float(total_loss))
    if not losses:
        return fallback_batch

    clipped = [max(1e-6, min(10.0, x)) for x in losses[:32]]
    avg = sum(clipped) / len(clipped)
    centered = [x - avg for x in clipped]
    bg_logits = [[x, x * 0.8, x * 0.6, x * 0.4, x * 0.2] for x in clipped[:8]]
    return {
        "features_c": [float(x) for x in clipped],
        "features_s": [float(x) for x in centered],
        "targets": [float(x / (avg + 1e-6)) for x in clipped],
        "bg_logits": bg_logits if bg_logits else fallback_batch["bg_logits"],
    }


def _compute_task_coverage_metrics(task: TaskDef, eval_payload: Dict[str, object], task_main_iters: int) -> Dict[str, float]:
    visible_count = len(task.seen_classes)
    new_count = len(task.new_classes)
    old_count = max(0, visible_count - new_count)

    old_map = eval_payload.get("class_iou_old")
    new_map = eval_payload.get("class_iou_new")
    old_eval_count = len(old_map) if isinstance(old_map, dict) else 0
    new_eval_count = len(new_map) if isinstance(new_map, dict) else 0
    visible_eval_count = old_eval_count + new_eval_count

    coverage_visible_ratio = float(visible_eval_count / visible_count) if visible_count > 0 else 0.0
    coverage_old_ratio = float(old_eval_count / old_count) if old_count > 0 else 0.0
    coverage_new_ratio = float(new_eval_count / new_count) if new_count > 0 else 0.0
    steps_per_visible = float(task_main_iters / visible_count) if visible_count > 0 else 0.0

    return {
        "visible_class_count": float(visible_count),
        "new_class_count": float(new_count),
        "old_class_count": float(old_count),
        "evaluated_visible_class_count": float(visible_eval_count),
        "coverage_visible_ratio": coverage_visible_ratio,
        "coverage_old_ratio": coverage_old_ratio,
        "coverage_new_ratio": coverage_new_ratio,
        "task_main_iters": float(task_main_iters),
        "steps_per_visible_class": steps_per_visible,
    }


def _compute_omega_tau_t(current_basis: List[float], basis_history: List[List[float]]) -> float:
    if not basis_history or not current_basis:
        return 0.0
    curr = torch.tensor(current_basis, dtype=torch.float32)
    curr_norm = torch.norm(curr) + 1e-8
    overlaps = []
    for prev_basis in basis_history:
        prev = torch.tensor(prev_basis, dtype=torch.float32)
        prev_norm = torch.norm(prev) + 1e-8
        cos = torch.dot(prev, curr) / (prev_norm * curr_norm)
        overlaps.append(float(torch.clamp(cos * cos, 0.0, 1.0).item()))
    return max(overlaps) if overlaps else 0.0


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_float_or(value: object, default: float) -> float:
    parsed = _safe_float(value)
    return default if parsed is None else parsed


def _format_metric(value: object, precision: int = 3) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.{precision}f}"


def _mean_metric_map(values: object) -> Optional[float]:
    if not isinstance(values, dict):
        return None
    finite: List[float] = []
    for value in values.values():
        parsed = _safe_float(value)
        if parsed is not None:
            finite.append(parsed)
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _read_d2_metrics(task_dir: Path) -> List[Dict[str, float]]:
    metrics_path = task_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    rows: List[Dict[str, float]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _resolve_prior_task_weight_path(output_dir: Path, state: dict, task_id: int) -> Optional[Path]:
    if task_id <= 1:
        return None
    expected = output_dir / f"task_{task_id - 1:03d}" / "model_final.pth"
    if expected.exists() and expected.is_file():
        return expected
    persisted = state.get("latest_model_path")
    if persisted:
        persisted_path = Path(str(persisted))
        if persisted_path.exists() and persisted_path.is_file():
            return persisted_path
    return expected


def _mean_from_keys(rows: List[Dict[str, float]], keys: List[str]) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        for key in keys:
            parsed = _safe_float(row.get(key))
            if parsed is not None:
                values.append(parsed)
                break
    if not values:
        return None
    return float(sum(values) / len(values))


def _estimate_beta_star_from_train(rows: List[Dict[str, float]]) -> Optional[float]:
    losses: List[float] = []
    for row in rows:
        parsed = _safe_float(row.get("total_loss"))
        if parsed is not None:
            losses.append(parsed)
    if len(losses) < 2:
        return None
    mean_loss = sum(losses) / len(losses)
    if mean_loss <= 0.0:
        return 0.0
    variance = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
    std = math.sqrt(max(variance, 0.0))
    return float(max(0.0, std / (mean_loss + 1e-8)))


def _summarize_train_loss_fields(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    return {
        "loss_sem_seg": _mean_from_keys(rows, ["loss_sem_seg"]),
        "loss_old_kd": _mean_from_keys(rows, ["loss_old_kd"]),
        "loss_old_clip": _mean_from_keys(rows, ["loss_old_clip"]),
        "loss_unseen_clip": _mean_from_keys(rows, ["loss_unseen_clip"]),
        "loss_ciba": _mean_from_keys(rows, ["loss_ciba"]),
        "loss_ctr": _mean_from_keys(rows, ["loss_ctr", "ctr_loss"]),
    }


def _compute_class_iou_overlap(
    current: Optional[Dict[str, float]],
    history: List[Dict[str, float]],
) -> float:
    if not isinstance(current, dict) or not current or not history:
        return 0.0
    best = 0.0
    for previous in history:
        if not previous:
            continue
        common = sorted(set(current.keys()).intersection(previous.keys()))
        if not common:
            continue
        curr_vec = torch.tensor([float(current[name]) for name in common], dtype=torch.float32)
        prev_vec = torch.tensor([float(previous[name]) for name in common], dtype=torch.float32)
        denom = (torch.norm(curr_vec) * torch.norm(prev_vec)).item()
        if denom <= 0.0:
            continue
        cosine = float(torch.dot(curr_vec, prev_vec).item() / denom)
        best = max(best, max(0.0, min(1.0, cosine)))
    return best


def _build_mock_rgb_palette(num_classes: int) -> torch.Tensor:
    idx = torch.arange(num_classes, dtype=torch.float32)
    return torch.stack([torch.sin(idx), torch.cos(idx), torch.tanh(idx / 10.0)], dim=1)


def _sample_mock_eval_batch(
    task: TaskDef,
    num_classes: int,
    batch_size: int,
    image_size: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    seen = [int(x) for x in task.seen_classes] or [0]
    class_pool = torch.tensor(seen, dtype=torch.long)
    sampled = torch.randint(0, class_pool.shape[0], (batch_size, image_size, image_size), generator=generator)
    targets = class_pool[sampled]
    palette = _build_mock_rgb_palette(num_classes=num_classes)
    images = palette[targets] + 0.05 * torch.randn(batch_size, image_size, image_size, 3, generator=generator)
    images = images.permute(0, 3, 1, 2).contiguous().float()
    return {"images": images, "targets": targets.long()}


def _compute_mock_task_eval_metrics(
    model: nn.Module,
    task: TaskDef,
    num_classes: int,
    seed: int,
    batches: int = 2,
) -> Dict[str, float]:
    core_model = getattr(model, "core_model", model)
    text_embeddings = getattr(model, "text_embeddings", None)
    if text_embeddings is None:
        return {"mIoU_all": 0.0, "mIoU_old": 0.0, "mIoU_new": 0.0, "BG-mIoU": 0.0}

    device = next(core_model.parameters()).device
    old_classes = [int(c) for c in task.seen_classes if int(c) not in {int(x) for x in task.new_classes}]
    new_classes = [int(x) for x in task.new_classes]
    bg_classes = [int(x) for x in task.background_classes]

    vals_all: List[float] = []
    vals_old: List[float] = []
    vals_new: List[float] = []
    vals_bg: List[float] = []

    core_model.eval()
    with torch.no_grad():
        for idx in range(max(1, batches)):
            batch = _sample_mock_eval_batch(
                task=task,
                num_classes=num_classes,
                batch_size=2,
                image_size=32,
                seed=seed + idx,
            )
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            logits = core_model(images=images, text_embeddings=text_embeddings, targets=None)["logits"]
            pred = logits.argmax(dim=1)

            vals_all.append(compute_basic_miou(pred=pred, target=targets, num_classes=num_classes))
            if old_classes:
                vals_old.append(compute_basic_miou(pred=pred, target=targets, num_classes=max(old_classes) + 1))
            if new_classes:
                vals_new.append(compute_basic_miou(pred=pred, target=targets, num_classes=max(new_classes) + 1))
            if bg_classes:
                vals_bg.append(compute_basic_miou(pred=pred, target=targets, num_classes=max(bg_classes) + 1))

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "mIoU_all": _mean(vals_all),
        "mIoU_old": _mean(vals_old),
        "mIoU_new": _mean(vals_new),
        "BG-mIoU": _mean(vals_bg),
    }


def _save_mock_inference_preview(
    task_dir: Path,
    model: nn.Module,
    task: TaskDef,
    num_classes: int,
    seed: int,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    core_model = getattr(model, "core_model", model)
    text_embeddings = getattr(model, "text_embeddings", None)
    if text_embeddings is None:
        return None

    batch = _sample_mock_eval_batch(task=task, num_classes=num_classes, batch_size=1, image_size=64, seed=seed)
    device = next(core_model.parameters()).device
    images = batch["images"].to(device)
    targets = batch["targets"]
    with torch.no_grad():
        logits = core_model(images=images, text_embeddings=text_embeddings, targets=None)["logits"]
    pred = logits.argmax(dim=1).detach().cpu()

    image_np = images.detach().cpu()[0].permute(1, 2, 0)
    image_np = ((image_np + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
    target_np = targets[0].numpy()
    pred_np = pred[0].numpy()

    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / "mock_inference_preview.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(target_np, cmap="tab20")
    axes[1].set_title("Target")
    axes[2].imshow(pred_np, cmap="tab20")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _build_d2_progress_callback(task_id: int, total_tasks: int, task_total_iters: int):
    iter_pattern = re.compile(r"iter:\s*(\d+)")
    last_iter = {"value": -1}
    started_at = time.monotonic()


class _D2TaskProgress:
    def __init__(self, task_id: int, total_tasks: int, task_total_iters: int, show_train: bool = True):
        self.task_id = task_id
        self.total_tasks = total_tasks
        self.task_total_iters = max(task_total_iters, 1)
        self._iter_pattern = re.compile(r"iter:\s*(\d+)")
        self._infer_pattern = re.compile(r"Inference done\s+(\d+)/(\d+)")
        self._train_pbar = None
        if show_train:
            self._train_pbar = tqdm(
                total=self.task_total_iters,
                desc=f"task {task_id}/{total_tasks} train",
                dynamic_ncols=True,
                leave=True,
            )
        self._eval_pbar = None
        self._last_iter = -1
        self._train_finalize_notified = False
        self._train_started_at = time.monotonic()

    def train_callback(self, line: str) -> None:
        if self._train_pbar is None:
            return
        match = self._iter_pattern.search(line)
        if not match:
            return
        current_iter = int(match.group(1))
        if current_iter == self._last_iter:
            return
        self._last_iter = current_iter
        self._train_pbar.n = min(max(current_iter + 1, 0), self.task_total_iters)
        self._train_pbar.refresh()
        if (
            self._train_pbar.n >= self.task_total_iters
            and not self._train_finalize_notified
        ):
            self._train_finalize_notified = True
            tqdm.write(
                "[open-continual] "
                f"task {self.task_id}/{self.total_tasks} train finalize | "
                "waiting for checkpoint write"
            )

    def eval_callback(self, line: str) -> None:
        match = self._infer_pattern.search(line)
        if not match:
            return
        done = int(match.group(1))
        total = max(int(match.group(2)), 1)
        if self._eval_pbar is None or self._eval_pbar.total != total:
            if self._eval_pbar is not None:
                self._eval_pbar.close()
            self._eval_pbar = tqdm(
                total=total,
                desc=f"task {self.task_id}/{self.total_tasks} eval",
                dynamic_ncols=True,
                leave=True,
            )
        self._eval_pbar.n = min(max(done, 0), total)
        self._eval_pbar.refresh()

    def close(self) -> None:
        if self._eval_pbar is not None:
            self._eval_pbar.n = self._eval_pbar.total
            self._eval_pbar.refresh()
            self._eval_pbar.close()
        if self._train_pbar is not None:
            self._train_pbar.n = self._train_pbar.total
            self._train_pbar.refresh()
            self._train_pbar.close()

    def train_elapsed_sec(self) -> int:
        return int(max(0.0, time.monotonic() - self._train_started_at))


class _MockTaskProgress:
    def __init__(self, task_id: int, total_tasks: int, n_pre: int, n_main: int):
        self.task_id = task_id
        self.total_tasks = total_tasks
        self._started_at = time.monotonic()
        self._totals = {
            "phase1": max(1, int(n_pre)),
            "phase2": max(1, int(n_main)),
            "phase3": 1,
            "phase4": 1,
            "infer": 1,
        }
        self._bars: Dict[str, object] = {}
        self._order = ["phase1", "phase2", "phase3", "phase4", "infer"]

    def callback(self, event: Dict[str, object]) -> None:
        phase = str(event.get("phase", "")).strip().lower()
        if phase not in self._totals:
            return
        total = max(1, int(event.get("total", self._totals[phase])))
        current = min(max(int(event.get("current", 0)), 0), total)
        pbar = self._bars.get(phase)
        if pbar is None or pbar.total != total:
            if pbar is not None:
                pbar.close()
            pbar = tqdm(
                total=total,
                desc=f"task {self.task_id}/{self.total_tasks} {phase}",
                dynamic_ncols=True,
                leave=True,
            )
            self._bars[phase] = pbar
        pbar.n = current
        pbar.refresh()
        message = event.get("message")
        if isinstance(message, str) and message:
            tqdm.write(
                "[open-continual] "
                f"task {self.task_id}/{self.total_tasks} {phase} | {message}"
            )

    def close(self) -> None:
        for phase in self._order:
            pbar = self._bars.get(phase)
            if pbar is None:
                continue
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

    def elapsed_sec(self) -> int:
        return int(max(0.0, time.monotonic() - self._started_at))


@dataclass
class OpenContinualTrainer:
    config_path: str
    output_dir: Path
    engine: str
    seed: int
    method_name: str
    clip_finetune: str
    task_spec: Optional[str]
    num_tasks: Optional[int]
    classes_per_task: Optional[int]
    task_seed: int
    n_pre: int
    n_main: int
    eps_f: float
    t_mem: str
    mix_ratio: List[int]
    m_max_total: int
    m_max_per_class: int
    ewc_lambda: float
    ewc_topk: int
    ewc_iters: int
    enable_ciba: bool
    enable_ctr: bool
    enable_spectral_ogp: bool
    enable_sacr: bool
    balanced_profile: str = "off"
    target_delta_new: float = 0.30
    epsilon_old: float = 0.20
    epsilon_all: float = 0.15
    epsilon_ov: float = 0.20
    seg_net: Optional[str] = None
    open_vocab_eval: bool = False
    skip_per_task_eval: bool = False
    eval_sliding_window: bool = True
    eval_max_samples_per_task: Optional[int] = None
    train_iters_mode: str = "auto"
    min_iters_per_visible_class: int = 350
    max_iters_multiplier: float = 2.0
    lambda_old_kd: float = 1.0
    lambda_old_clip: float = 0.1
    lambda_unseen_clip: float = 0.2
    resume_task: int = 0
    use_real_training: bool = False
    clip_model_name: str = "ViT-B-16"
    dino_model_name: str = "dinov2_vitb14"
    learning_rate: float = 1e-4
    text_learning_rate: float = 1e-5
    batch_size: int = 4
    dataset_root: str = "datasets/ADE20K"
    image_size: int = 518
    lr_scheduler: str = "cosine"
    num_workers: int = 4
    use_amp: bool = True
    eval_max_samples: int = 500
    _mock_model: Optional[nn.Module] = field(default=None, init=False, repr=False)

    @classmethod
    def from_args(cls, args) -> "OpenContinualTrainer":
        return cls(
            config_path=args.config,
            output_dir=Path(args.output_dir),
            engine=args.engine,
            seed=args.seed,
            method_name=args.col_method,
            clip_finetune=args.clip_finetune,
            seg_net=args.seg_net,
            task_spec=args.task_spec,
            num_tasks=args.num_tasks,
            classes_per_task=args.classes_per_task,
            task_seed=args.task_seed,
            n_pre=args.n_pre,
            n_main=args.n_main,
            eps_f=args.eps_f,
            t_mem=args.t_mem,
            mix_ratio=list(args.mix_ratio),
            m_max_total=args.m_max_total,
            m_max_per_class=args.m_max_per_class,
            ewc_lambda=args.ewc_lambda,
            ewc_topk=args.ewc_topk,
            ewc_iters=args.ewc_iters,
            balanced_profile=args.balanced_profile,
            target_delta_new=args.target_delta_new,
            epsilon_old=args.epsilon_old,
            epsilon_all=args.epsilon_all,
            epsilon_ov=args.epsilon_ov,
            enable_ciba=args.enable_ciba,
            enable_ctr=args.enable_ctr,
            enable_spectral_ogp=args.enable_spectral_ogp,
            enable_sacr=args.enable_sacr,
            open_vocab_eval=args.open_vocab,
            skip_per_task_eval=args.skip_per_task_eval,
            eval_sliding_window=args.eval_sliding_window,
            eval_max_samples_per_task=args.eval_max_samples_per_task,
            train_iters_mode=args.train_iters_mode,
            min_iters_per_visible_class=args.min_iters_per_visible_class,
            max_iters_multiplier=args.max_iters_multiplier,
            lambda_old_kd=getattr(args, "lambda_old_kd", 1.0),
            lambda_old_clip=getattr(args, "lambda_old_clip", 0.1),
            lambda_unseen_clip=getattr(args, "lambda_unseen_clip", 0.2),
            resume_task=args.resume_task,
            use_real_training=getattr(args, "use_real_training", False),
            clip_model_name=getattr(args, "clip_model_name", "ViT-B-16"),
            dino_model_name=getattr(args, "dino_model_name", "dinov2_vitb14"),
            learning_rate=getattr(args, "learning_rate", 1e-4),
            text_learning_rate=getattr(args, "text_learning_rate", 1e-5),
            batch_size=getattr(args, "batch_size", 4),
            dataset_root=getattr(args, "dataset_root", "datasets/ADE20K"),
            image_size=getattr(args, "image_size", 518),
            lr_scheduler=getattr(args, "lr_scheduler", "cosine"),
            num_workers=getattr(args, "num_workers", 4),
            use_amp=getattr(args, "use_amp", True),
            eval_max_samples=getattr(args, "eval_max_samples", 500),
        )

    def _resolve_task_main_iters(self, task: TaskDef) -> int:
        base_iters = int(max(1, self.n_main))
        mode = str(self.train_iters_mode).strip().lower()
        if mode not in {"off", "on", "auto"}:
            raise ValueError(f"Unsupported train_iters_mode: {self.train_iters_mode}")
        if mode == "off" or base_iters < 50:
            return base_iters

        should_scale = mode == "on" or (mode == "auto" and len(task.seen_classes) > len(task.new_classes))
        if not should_scale:
            return base_iters

        visible_count = max(1, len(task.seen_classes))
        target_iters = max(base_iters, int(self.min_iters_per_visible_class) * visible_count)
        cap_iters = max(base_iters, int(round(base_iters * float(self.max_iters_multiplier))))
        return int(max(base_iters, min(target_iters, cap_iters)))

    def _resolve_total_classes(self) -> int:
        config_lower = self.config_path.lower()
        if "coco" in config_lower:
            return 164
        if "ade" in config_lower:
            return 150
        return max(self.classes_per_task or 1, self.num_tasks or 1) * (self.classes_per_task or 15)

    def _build_model(self, num_classes: int) -> COVLSegModelV2:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[open-continual] building model on {device} (clip={self.clip_model_name}, dino={self.dino_model_name})")
        model = COVLSegModelV2(
            clip_model_name=self.clip_model_name,
            dino_model_name=self.dino_model_name,
            clip_finetune=self.clip_finetune,
            num_classes=num_classes,
        )
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[open-continual] model initialized ({n_params:,} trainable params)")
        return model

    def _get_all_class_names(self) -> List[str]:
        config_lower = self.config_path.lower()
        total = self._resolve_total_classes()
        if "coco" in config_lower:
            total = min(total, len(COCO_STUFF_164_CLASSES))
            return COCO_STUFF_164_CLASSES[:total]
        return [f"class_{i}" for i in range(total)]

    def _build_dataloader(self, task: TaskDef, mode: str = "train", class_names: Optional[List[str]] = None, replay_buffer: Optional[SACRReplayBuffer] = None) -> DataLoader:
        if class_names is None:
            class_names = self._get_all_class_names()
        config_lower = self.config_path.lower()
        augment = SegmentationAugmentation(image_size=self.image_size) if mode == "train" else None
        eval_transform = SegmentationEvalTransform(image_size=self.image_size) if mode == "eval" else None
        num_classes = self._resolve_total_classes()
        visible_class_ids = list(task.seen_classes) if mode == "train" else None

        _log.info("[dataloader] mode=%s dataset=%s seed=%d split=%s n_classes=%d", mode, "coco" if "coco" in config_lower else "ade", task.task_id, "train" if mode == "train" else "val", num_classes)
        print(f"[open-continual] task {task.task_id} loading {mode} dataset from {self.dataset_root} ...", flush=True)

        dataset_kwargs = dict(
            root=self.dataset_root,
            split="training" if mode == "train" else "validation",
            class_names=class_names,
            num_classes=num_classes,
            augmentation=augment or eval_transform,
        )

        if "coco" in config_lower:
            DatasetClass = COCOStuffDataset
        else:
            DatasetClass = ADE20KDataset

        try:
            dataset = DatasetClass(**dataset_kwargs, visible_class_ids=visible_class_ids)
        except TypeError:
            dataset = DatasetClass(**dataset_kwargs)

        if mode == "train" and visible_class_ids is not None and not isinstance(dataset, (ClassFilteredDataset, torch.utils.data.Subset)):
            try:
                dataset = ClassFilteredDataset(
                    dataset,
                    visible_class_ids=visible_class_ids,
                    min_visible_ratio=0.01,
                )
                print(f"[open-continual] task {task.task_id} filtered {mode} dataset: {len(dataset)} samples with {len(visible_class_ids)} classes", flush=True)
            except Exception:
                pass

        if mode == "eval" and self.eval_max_samples > 0 and len(dataset) > self.eval_max_samples:
            indices = torch.randperm(len(dataset))[:self.eval_max_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)

        n_samples = len(dataset)
        print(f"[open-continual] task {task.task_id} {mode} dataset ready: {n_samples} samples", flush=True)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=(mode == "train"), num_workers=self.num_workers, pin_memory=True)

    def _train_task(self, model: COVLSegModelV2, dataloader: DataLoader, class_names: List[str], ewc: Optional[EWCRegularizer] = None, task_id: int = 0, optimizer_state: Optional[dict] = None, seen_class_ids: Optional[List[int]] = None, unseen_class_ids: Optional[List[int]] = None, ciba_weight: float = 0.0, ctr_weight: float = 0.0, ogp_basis: Optional[torch.Tensor] = None) -> tuple:
        device = next(model.parameters()).device
        use_amp = self.use_amp and device.type == "cuda"
        model.train()

        if self.clip_finetune in ("none", "attention"):
            model.clip_visual.eval()
        model.dino.eval()

        optimizer = torch.optim.AdamW([
            {"params": model.hciba_head.parameters()},
            {"params": model.fusion_head.parameters()},
            {"params": model.clip_logit_proj.parameters()},
            {"params": model.clip_text.parameters(), "lr": self.text_learning_rate},
        ], lr=self.learning_rate)

        if self.clip_finetune in ("attention", "full"):
            existing_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            visual_params = [p for p in model.clip_visual.parameters() if p.requires_grad and id(p) not in existing_ids]
            if visual_params:
                optimizer.add_param_group({"params": visual_params, "lr": self.learning_rate * 0.1})

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        n_batches = 0
        max_batches = self.n_main
        warmup_steps = min(500, max_batches // 10)
        if self.lr_scheduler == "cosine" and max_batches > 0:
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_batches - warmup_steps, eta_min=self.learning_rate * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                    base_scheduler,
                ],
                milestones=[warmup_steps],
            )
        else:
            scheduler = None

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"task {task_id} train", total=max_batches, leave=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["sem_seg"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            mi_estimate = None
            if ciba_weight > 0.0:
                with torch.no_grad():
                    mi_estimate = torch.tensor(0.0, device=device)

            forward_kwargs = {"targets": targets}
            if seen_class_ids is not None and hasattr(model, 'compute_ctr_loss'):
                forward_kwargs["seen_class_ids"] = seen_class_ids
                forward_kwargs["unseen_class_ids"] = unseen_class_ids
                forward_kwargs["mi_estimate"] = mi_estimate
                forward_kwargs["ciba_weight"] = ciba_weight
                forward_kwargs["ctr_weight"] = ctr_weight

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    images=images,
                    class_names=class_names,
                    **forward_kwargs,
                )
                loss = outputs["loss"]

            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                _log.warning("[open-continual] NaN/Inf loss at task %d batch %d, skipping update", task_id, n_batches)
                with torch.no_grad():
                    for k, v in outputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  [{k}] shape={tuple(v.shape)} min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f} nan={torch.isnan(v).any().item()} inf={torch.isinf(v).any().item()}", flush=True)
                    print(f"  [images] min={images.min():.3f} max={images.max():.3f} mean={images.mean():.3f}", flush=True)
                    print(f"  [targets] min={targets.min().item()} max={targets.max().item()} unique={targets.unique().tolist()}", flush=True)
                    for name, param in model.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            print(f"  [GRAD NaN/Inf] {name} shape={tuple(param.shape)}", flush=True)
                n_batches += 1
                continue

            if ewc is not None:
                loss = loss + ewc.penalty(model)

            if scaler is not None:
                scaler.scale(loss).backward()
                if ogp_basis is not None:
                    grad_vec = flatten_gradients(model)
                    if grad_vec.numel() == ogp_basis.shape[0]:
                        projected = hard_project_gradient(grad_vec, ogp_basis)
                        offset = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                numel = param.grad.numel()
                                if offset + numel <= projected.shape[0]:
                                    param.grad.copy_(projected[offset:offset + numel].view_as(param.grad))
                                offset += numel
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if ogp_basis is not None:
                    unflatten_and_project(model, flatten_gradients(model), ogp_basis)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            avg_loss = total_loss / n_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if n_batches >= max_batches:
                break
        pbar.close()

        return {"train_loss": total_loss / max(n_batches, 1), "n_batches": float(n_batches)}, optimizer, scheduler

    def _eval_task(self, model: COVLSegModelV2, dataloader: DataLoader, class_names: List[str], task_id: int = 0, seen_class_ids: Optional[List[int]] = None) -> Dict[str, float]:
        device = next(model.parameters()).device
        use_amp = self.use_amp and device.type == "cuda"
        model.eval()
        num_classes = model.num_classes
        # Evaluate only on seen classes (not all classes)
        eval_class_ids = list(range(num_classes)) if seen_class_ids is None else sorted(c for c in seen_class_ids if c < num_classes)
        n_eval_classes = len(eval_class_ids) if eval_class_ids else num_classes
        correct = 0
        total = 0
        conf_matrix = torch.zeros(n_eval_classes, n_eval_classes, dtype=torch.long)
        # Build index mapping: global class id → eval matrix index
        global_to_eval = {gid: eid for eid, gid in enumerate(eval_class_ids)}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"task {task_id} eval", leave=True):
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["sem_seg"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images, class_names, targets=targets)
                if targets is not None:
                    preds = F.interpolate(outputs["logits"], size=targets.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                else:
                    preds = outputs["logits"].argmax(dim=1)
                mask = targets != 255
                if seen_class_ids is not None:
                    seen_set = set(seen_class_ids)
                    class_mask = torch.isin(targets, torch.tensor(list(seen_set), device=targets.device))
                    mask = mask & class_mask
                correct += (preds[mask] == targets[mask]).sum().item()
                total += mask.sum().item()
                valid = mask
                pred_valid = preds[valid].clamp(0, num_classes - 1)
                tgt_valid = targets[valid].clone()
                tgt_valid[(tgt_valid >= num_classes)] = num_classes - 1
                # Remap both to eval matrix indices
                for p, t in zip(pred_valid.tolist(), tgt_valid.tolist()):
                    ep = global_to_eval.get(p)
                    et = global_to_eval.get(t)
                    if ep is not None and et is not None:
                        conf_matrix[ep, et] += 1
        accuracy = correct / max(total, 1)
        iou_per_class = conf_matrix.diag().float() / (conf_matrix.sum(0) + conf_matrix.sum(1) - conf_matrix.diag()).float().clamp(min=1)
        present = (conf_matrix.sum(0) + conf_matrix.sum(1)) > 0
        miou = iou_per_class[present].mean().item() if present.any() else 0.0
        return {"accuracy": accuracy, "mIoU": miou * 100.0, "total_pixels": float(total), "eval_classes": n_eval_classes}

    def _build_task_plan(self) -> TaskPlan:
        total_classes = _infer_num_classes_from_config(
            self.config_path,
            strict=self.engine == "d2",
        )
        resolved_num_tasks = self.num_tasks
        resolved_classes_per_task = self.classes_per_task

        if self.task_spec is None:
            if resolved_num_tasks is None and resolved_classes_per_task is None:
                resolved_num_tasks = 1
                resolved_classes_per_task = total_classes
            elif resolved_num_tasks is None and resolved_classes_per_task is not None:
                resolved_num_tasks = max(1, (total_classes + resolved_classes_per_task - 1) // resolved_classes_per_task)
            elif resolved_num_tasks is not None and resolved_classes_per_task is None:
                resolved_classes_per_task = max(1, (total_classes + resolved_num_tasks - 1) // resolved_num_tasks)

        return build_task_plan(
            task_spec=self.task_spec,
            num_tasks=resolved_num_tasks,
            classes_per_task=resolved_classes_per_task,
            all_classes=list(range(total_classes)),
            seed=self.task_seed,
        )

    def _state_path(self) -> Path:
        return self.output_dir / "continual_state.json"

    def _metrics_path(self) -> Path:
        return self.output_dir / "metrics.jsonl"

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_state(self) -> dict:
        state_path = self._state_path()
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _validate_resume(self, state: dict) -> None:
        if not state:
            if self.engine == "d2" and self.resume_task > 0:
                raise ValueError(
                    "D2 resume requires a prior model artifact, but no continual state was found."
                )
            if self.engine == "mock" and self.resume_task > 0:
                raise ValueError(
                    "Mock resume requires prior continual state, but no continual state was found."
                )
            return
        prior_method = state.get("method")
        if prior_method is not None and prior_method != self.method_name:
            raise ValueError(
                f"Resume method mismatch: existing={prior_method}, requested={self.method_name}"
            )
        if self.engine == "mock" and self.resume_task > 0:
            prior_mock_model_path = state.get("latest_mock_model_path") or state.get("latest_model_path")
            if not prior_mock_model_path:
                raise ValueError(
                    "Mock resume requires a valid mock model artifact path in continual state "
                    f"for resume_task={self.resume_task}."
                )
            prior_path = Path(str(prior_mock_model_path))
            if not prior_path.is_absolute():
                prior_path = (self.output_dir / prior_path).resolve()
            if not prior_path.exists() or not prior_path.is_file():
                raise ValueError(
                    "Mock resume requires a valid mock model artifact at "
                    f"{prior_path} for resume_task={self.resume_task}."
                )
            return

        if self.engine != "d2" or self.resume_task <= 0:
            return
        prior_weight_path = _resolve_prior_task_weight_path(
            output_dir=self.output_dir,
            state=state,
            task_id=self.resume_task + 1,
        )
        if prior_weight_path is None or not prior_weight_path.exists() or not prior_weight_path.is_file():
            raise ValueError(
                "D2 resume requires a valid prior model artifact at "
                f"{prior_weight_path} for resume_task={self.resume_task}."
            )

    def run(self, max_tasks: Optional[int] = None) -> Dict[str, float]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        method = build_continual_method(
            name=self.method_name,
            config={
                "enable_ciba": self.enable_ciba,
                "enable_ctr": self.enable_ctr,
                "enable_spectral_ogp": self.enable_spectral_ogp,
                "enable_sacr": self.enable_sacr,
                "ewc_lambda": self.ewc_lambda,
                "ewc_topk": self.ewc_topk,
                "ewc_iters": self.ewc_iters,
            },
        )
        plan = self._build_task_plan()
        self._write_json(self.output_dir / "task_plan.json", plan.to_dict())

        state = self._load_state()
        self._validate_resume(state)

        alpha_star_history = list(state.get("alpha_star_history", []))
        tau_pred_history = list(state.get("tau_pred_history", []))
        basis_history = list(state.get("basis_history", []))
        class_iou_history = list(state.get("class_iou_history", []))
        balanced_prev_eval = dict(state.get("balanced_prev_eval", {}))

        balanced_enabled = self.balanced_profile == "balanced"
        balanced_cfg = BalancedControllerConfig(
            epsilon_ov=self.epsilon_ov,
            target_delta_new=self.target_delta_new,
        )
        balanced_state_payload = state.get("balanced_controller", {}) if balanced_enabled else {}
        balanced_state = BalancedControllerState(
            alpha_floor=_safe_float_or(balanced_state_payload.get("alpha_floor"), 0.0),
            g_stab=_safe_float_or(balanced_state_payload.get("g_stab"), 0.0),
            rho_old=_safe_float_or(balanced_state_payload.get("rho_old"), 0.0),
            rho_new=_safe_float_or(balanced_state_payload.get("rho_new"), 0.0),
            w_ctr=_safe_float_or(balanced_state_payload.get("w_ctr"), 0.0),
            ov_guard_triggered=bool(balanced_state_payload.get("ov_guard_triggered", False)),
        )

        task_cfg = {
            "n_pre": self.n_pre,
            "n_main": self.n_main,
            "eps_f": self.eps_f,
            "t_mem": self.t_mem,
            "mix_ratio": self.mix_ratio,
            "m_max_total": self.m_max_total,
            "m_max_per_class": self.m_max_per_class,
            "ewc_lambda": self.ewc_lambda,
            "ewc_topk": self.ewc_topk,
            "ewc_iters": self.ewc_iters,
            "enable_ciba": self.enable_ciba,
            "enable_ctr": self.enable_ctr,
            "enable_spectral_ogp": self.enable_spectral_ogp,
            "enable_sacr": self.enable_sacr,
        }
        clip_overrides = _clip_overrides(self.clip_finetune)
        class_names = _resolve_task_class_names(self.config_path) if self.engine == "d2" else None
        if self.engine == "mock" and self._mock_model is None:
            all_task_classes = [
                c
                for task in plan.tasks
                for c in (task.seen_classes + task.new_classes + task.background_classes)
            ]
            num_classes = max(all_task_classes, default=-1) + 1
            self._mock_model = build_covl_training_model(
                num_classes=max(1, num_classes),
                seed=self.seed,
            )
            prior_mock_model_path = state.get("latest_mock_model_path") or state.get("latest_model_path")
            if self.resume_task > 0:
                if not prior_mock_model_path:
                    raise ValueError(
                        "Mock resume requires a valid mock model artifact path in continual state "
                        f"for resume_task={self.resume_task}."
                    )
                prior_path = Path(str(prior_mock_model_path))
                if not prior_path.is_absolute():
                    prior_path = (self.output_dir / prior_path).resolve()
                if not prior_path.exists() or not prior_path.is_file():
                    raise ValueError(
                        "Mock resume requires a valid mock model artifact at "
                        f"{prior_path} for resume_task={self.resume_task}."
                    )
                prior_payload = torch.load(prior_path, map_location="cpu")
                model_state_dict = prior_payload.get("model_state_dict", prior_payload)
                self._mock_model.load_state_dict(model_state_dict)

        completed = 0
        task_pbar = tqdm(
            total=len(plan.tasks) - sum(1 for t in plan.tasks if t.task_id <= self.resume_task),
            desc="continual tasks",
            unit="task",
            leave=True,
        )
        for task in plan.tasks:
            if task.task_id <= self.resume_task:
                continue
            if max_tasks is not None and completed >= max_tasks:
                break

            task_dir = self.output_dir / f"task_{task.task_id:03d}"
            total_tasks = len(plan.tasks)
            remaining_tasks = max(0, total_tasks - task.task_id)
            task_main_iters = self._resolve_task_main_iters(task)
            progress = _D2TaskProgress(
                task_id=task.task_id,
                total_tasks=total_tasks,
                task_total_iters=task_main_iters,
            ) if self.engine == "d2" else None
            method.before_task(task_state={"task_id": task.task_id})
            method_phase_cfg = method.phase_overrides()
            balanced_knobs = {
                "balanced_w_ctr": 1.0,
                "balanced_w_oldfix": 0.0,
                "balanced_g_stab": 0.0,
                "balanced_alpha_floor": 0.0,
                "balanced_rho_new": 0.0,
                "balanced_rho_old": 0.0,
            }
            if balanced_enabled:
                balanced_knobs = {
                    "balanced_w_ctr": float(balanced_state.w_ctr),
                    "balanced_w_oldfix": float(balanced_state.rho_old),
                    "balanced_g_stab": float(balanced_state.g_stab),
                    "balanced_alpha_floor": float(balanced_state.alpha_floor),
                    "balanced_rho_new": float(balanced_state.rho_new),
                    "balanced_rho_old": float(balanced_state.rho_old),
                }
            phase_cfg = {**task_cfg, **method_phase_cfg, **balanced_knobs, "n_main": task_main_iters}
            print(
                f"\n[open-continual] === Task {task.task_id}/{total_tasks} === "
                f"new_classes={len(task.new_classes)} seen_classes={len(task.seen_classes)} "
                f"bg_classes={len(task.background_classes)} iters={task_main_iters}"
            )

            model = None
            ewc = None
            dataloader = None
            replay_buffer = None
            ogp_basis_tensor = None
            if self.use_real_training:
                print(f"[open-continual] task {task.task_id}/{total_tasks} building model...")
                all_class_names = self._get_all_class_names()
                total_classes = len(all_class_names)
                model = self._build_model(num_classes=total_classes)
                print(f"[open-continual] task {task.task_id}/{total_tasks} model ready (device={next(model.parameters()).device}, classes={total_classes})")
                optimizer_state = None
                ewc_state_from_ckpt = None
                prev_ckpt_path = self.output_dir / f"task_{task.task_id - 1:03d}" / "model.pt" if task.task_id > 1 else None
                if prev_ckpt_path is not None and prev_ckpt_path.exists():
                    ckpt = torch.load(prev_ckpt_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                    if "alpha_star" in ckpt:
                        model.inject_alpha_tau(ckpt["alpha_star"], ckpt.get("tau_pred", 1.0))
                    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                        optimizer_state = ckpt["optimizer_state_dict"]
                    if "ewc_state_dict" in ckpt and ckpt["ewc_state_dict"] is not None:
                        ewc_state_from_ckpt = ckpt["ewc_state_dict"]
                    print(f"[open-continual] task {task.task_id}/{total_tasks} resumed from {prev_ckpt_path}")

                if self.enable_sacr and task.task_id > 1:
                    replay_path = self.output_dir / f"task_{task.task_id - 1:03d}" / "replay_buffer.json"
                    if replay_path.exists():
                        replay_buffer = SACRReplayBuffer.load(replay_path)

                if self.enable_spectral_ogp and task.task_id > 1:
                    basis_path = self.output_dir / f"task_{task.task_id - 1:03d}" / "ogp_basis.pt"
                    if basis_path.exists():
                        ogp_basis_tensor = torch.load(basis_path, map_location="cpu", weights_only=False)

                seen_class_ids = [int(c) for c in task.seen_classes]
                unseen_class_ids = [int(c) for c in task.background_classes]
                ciba_weight = float(method_phase_cfg.get("enable_ciba", 0.0)) if isinstance(method_phase_cfg.get("enable_ciba"), (int, float)) else (1.0 if method_phase_cfg.get("enable_ciba", False) else 0.0)
                ctr_weight = float(method_phase_cfg.get("enable_ctr", 0.0)) if isinstance(method_phase_cfg.get("enable_ctr"), (int, float)) else (0.1 if method_phase_cfg.get("enable_ctr", False) else 0.0)

                dataloader = self._build_dataloader(task, class_names=all_class_names, replay_buffer=replay_buffer)
                eval_dataloader = self._build_dataloader(task, mode="eval", class_names=all_class_names)
                try:
                    train_len = len(dataloader.dataset)
                    eval_len = len(eval_dataloader.dataset)
                except Exception:
                    train_len = "?"
                    eval_len = "?"
                print(f"[open-continual] task {task.task_id}/{total_tasks} data ready (train={train_len}, eval={eval_len})")
                ewc_for_train = None
                if ewc_state_from_ckpt is not None and task.task_id > 1:
                    ewc_for_train = EWCRegularizer(model, lambda_ewc=self.ewc_lambda)
                    ewc_for_train.load_state_dict(ewc_state_from_ckpt)
                train_metrics, optimizer, scheduler = self._train_task(
                    model, dataloader, all_class_names,
                    ewc=ewc_for_train, task_id=task.task_id, optimizer_state=optimizer_state,
                    seen_class_ids=seen_class_ids, unseen_class_ids=unseen_class_ids,
                    ciba_weight=ciba_weight, ctr_weight=ctr_weight,
                    ogp_basis=ogp_basis_tensor,
                )
                append_metrics_jsonl(self._metrics_path(), {"phase": "train", "task": float(task.task_id), **train_metrics})
                eval_metrics = self._eval_task(model, eval_dataloader, all_class_names, task_id=task.task_id, seen_class_ids=seen_class_ids)
                append_metrics_jsonl(self._metrics_path(), {"phase": "eval", "task": float(task.task_id), **eval_metrics})
                print(
                    f"[open-continual] task {task.task_id}/{total_tasks} | "
                    f"train_loss={train_metrics.get('train_loss', 'N/A'):.4f} | "
                    f"mIoU={eval_metrics.get('mIoU', 0.0):.2f} | "
                    f"acc={eval_metrics.get('accuracy', 0.0):.4f}"
                )
            mock_progress = _MockTaskProgress(
                task_id=task.task_id,
                total_tasks=total_tasks,
                n_pre=int(phase_cfg.get("n_pre", self.n_pre)),
                n_main=int(task_main_iters),
            ) if self.engine == "mock" else None
            batch = _build_phase_batch(task)
            if self.use_real_training:
                # Real training already done above; skip mock/phase paths
                p1 = p2 = p3 = p4 = {}
            elif self.engine == "mock":
                assert self._mock_model is not None
                try:
                    try:
                        self._mock_model, phase_metrics = run_mock_task_training(
                            model=self._mock_model,
                            task=task,
                            cfg=phase_cfg,
                            basis_history=[torch.tensor(b, dtype=torch.float32) for b in basis_history],
                            progress_callback=mock_progress.callback if mock_progress is not None else None,
                        )
                    except TypeError as exc:
                        if "progress_callback" not in str(exc):
                            raise
                        self._mock_model, phase_metrics = run_mock_task_training(
                            model=self._mock_model,
                            task=task,
                            cfg=phase_cfg,
                            basis_history=[torch.tensor(b, dtype=torch.float32) for b in basis_history],
                        )
                    if mock_progress is not None:
                        print(
                            "[open-continual] "
                            f"task {task.task_id}/{total_tasks} mock train done | "
                            f"elapsed_sec={mock_progress.elapsed_sec()}"
                        )
                finally:
                    if mock_progress is not None:
                        mock_progress.close()
                p1 = dict(phase_metrics["phase1"])
                p2 = dict(phase_metrics["phase2"])
                p3 = dict(phase_metrics["phase3"])
                p4 = dict(phase_metrics["phase4"])
            else:
                p1 = run_phase1_hciba(task.task_id, phase_cfg, batch=batch)
                p2 = run_phase2_joint(task.task_id, phase_cfg, batch=batch)
                p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=batch, prev_phase_metrics=p1)
                p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=batch)

            current_basis = p3.get("subspace_basis", [])
            p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))
            real_continual = {
                "beta_1_star": _safe_float(p1.get("beta_1_star")),
                "ctr_loss": _safe_float(p2.get("ctr_loss")),
                "alpha_star": _safe_float(p3.get("alpha_star")),
                "tau_pred": _safe_float(p3.get("tau_pred")),
                "omega_tau_t": _safe_float(p3.get("omega_tau_t")),
            }

            if self.use_real_training and model is not None:
                alpha_star = _safe_float_or(real_continual.get("alpha_star"), 0.5)
                tau_pred = _safe_float_or(real_continual.get("tau_pred"), 1.0)
                model.inject_alpha_tau(alpha_star, tau_pred)

                if self.ewc_lambda > 0:
                    ewc = EWCRegularizer(model, lambda_ewc=self.ewc_lambda)
                    ewc.compute_fisher(
                        dataloader,
                        loss_fn=lambda inputs, targets: model(
                            inputs,
                            all_class_names,
                            targets=targets,
                        )["loss"],
                        n_samples=self.ewc_iters,
                    )
                    ewc.consolidate()

                ewc_state_dict = None
                if ewc is not None:
                    ewc_state_dict = ewc.state_dict()
                elif ewc_state_from_ckpt is not None:
                    ewc = EWCRegularizer(model, lambda_ewc=self.ewc_lambda)
                    ewc.load_state_dict(ewc_state_from_ckpt)
                    ewc_state_dict = ewc.state_dict()

                if self.enable_spectral_ogp and dataloader is not None:
                    try:
                        new_basis = compute_gradient_basis(
                            model=model,
                            dataloader=dataloader,
                            loss_fn=lambda inputs, targets: model(
                                inputs, all_class_names, targets=targets,
                            )["loss"],
                            n_samples=min(self.ewc_iters, 200),
                            top_k=10,
                            device=next(model.parameters()).device,
                        )
                        task_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(new_basis, task_dir / "ogp_basis.pt")
                        ogp_basis_tensor = new_basis
                    except Exception as exc:
                        _log.warning("OGP basis computation failed: %s", exc)

                if self.enable_sacr and self.m_max_total > 0:
                    if replay_buffer is None:
                        replay_buffer = SACRReplayBuffer(
                            max_total_items=self.m_max_total,
                            max_per_class=self.m_max_per_class,
                        )
                    seen_ids = [int(c) for c in task.seen_classes]
                    for sample_idx, sample in enumerate(dataloader.dataset):
                        if isinstance(sample, dict) and "image_id" in sample:
                            img_path = str(sample.get("image_path", ""))
                            lbl_path = str(sample.get("mask_path", ""))
                            if img_path and lbl_path:
                                for cid in seen_ids:
                                    replay_buffer.add(ReplayItem(
                                        image_path=img_path,
                                        label_path=lbl_path,
                                        class_id=cid,
                                        priority=1.0 / (1 + sample_idx),
                                    ))
                    task_dir.mkdir(parents=True, exist_ok=True)
                    replay_buffer.save(task_dir / "replay_buffer.json")

                task_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                    "alpha_star": alpha_star,
                    "tau_pred": tau_pred,
                    "task_id": task.task_id,
                    "total_classes": total_classes,
                    "clip_model_name": self.clip_model_name,
                    "dino_model_name": self.dino_model_name,
                    "clip_finetune": self.clip_finetune,
                    "ewc_state_dict": ewc_state_dict,
                }
                if scheduler is not None:
                    checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(checkpoint_dict, task_dir / "model.pt")

                (self.output_dir / "latest_model.txt").write_text(str(task_dir / "model.pt"), encoding="utf-8")

            eval_payload = None
            if self.engine == "d2":
                assert class_names is not None
                class_artifacts = _write_task_class_artifacts(
                    task_dir=task_dir,
                    task=task,
                    class_names=class_names,
                )
                task_overrides = clip_overrides + [
                    "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON",
                    str(class_artifacts["train_class_json"].resolve()),
                    "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON",
                    str(class_artifacts["test_class_json"].resolve()),
                    "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES",
                    str(class_artifacts["train_indexes"].resolve()),
                    "MODEL.SEM_SEG_HEAD.TRAIN_OLD_CLASS_INDEXES",
                    str(class_artifacts["old_indexes"].resolve()),
                    "MODEL.SEM_SEG_HEAD.TRAIN_UNSEEN_CLASS_INDEXES",
                    str(class_artifacts["test_indexes"].resolve()),
                    "MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES",
                    str(class_artifacts["test_indexes"].resolve()),
                    "MODEL.SEM_SEG_HEAD.OLD_TEACHER_WEIGHTS",
                    "",
                    "MODEL.COVL.ENABLE_CIBA",
                    str(bool(self.enable_ciba)),
                    "MODEL.COVL.ENABLE_CTR",
                    str(bool(self.enable_ctr)),
                    "MODEL.COVL.ENABLE_OGP",
                    str(bool(self.enable_spectral_ogp)),
                    "MODEL.SEM_SEG_HEAD.LAMBDA_OLD_KD",
                    str(float(self.lambda_old_kd)),
                    "MODEL.SEM_SEG_HEAD.LAMBDA_OLD_CLIP",
                    str(float(self.lambda_old_clip)),
                    "MODEL.SEM_SEG_HEAD.LAMBDA_UNSEEN_CLIP",
                    str(float(self.lambda_unseen_clip)),
                    "SOLVER.MAX_ITER",
                    str(task_main_iters),
                    "TEST.EVAL_PERIOD",
                    "0",
                ]
                prior_weight_path = _resolve_prior_task_weight_path(
                    output_dir=self.output_dir,
                    state=state,
                    task_id=task.task_id,
                )
                if prior_weight_path is not None:
                    task_overrides[task_overrides.index("MODEL.SEM_SEG_HEAD.OLD_TEACHER_WEIGHTS") + 1] = str(
                        prior_weight_path.resolve()
                    )
                    task_overrides.extend(["MODEL.WEIGHTS", str(prior_weight_path.resolve())])
                try:
                    run_detectron2_train(
                        config_path=self.config_path,
                        output_dir=task_dir,
                        seed=self.seed,
                        resume_task=task.task_id - 1,
                        max_tasks=1,
                        seg_network=self.seg_net,
                        extra_overrides=task_overrides,
                        progress_callback=progress.train_callback if progress is not None else None,
                    )
                    if progress is not None:
                        print(
                            "[open-continual] "
                            f"task {task.task_id}/{total_tasks} train done | "
                            f"elapsed_sec={progress.train_elapsed_sec()}"
                        )
                finally:
                    if progress is not None:
                        progress.close()
                d2_batch = _build_phase_batch_from_d2_metrics(task_dir=task_dir, fallback_batch=batch)
                p1 = run_phase1_hciba(task.task_id, phase_cfg, batch=d2_batch)
                p2 = run_phase2_joint(task.task_id, phase_cfg, batch=d2_batch)
                p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=d2_batch, prev_phase_metrics=p1)
                p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=d2_batch)
                current_basis = p3.get("subspace_basis", [])
                p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))
                for record in (p1, p2, p3, p4):
                    d2_record = dict(record)
                    d2_record["proxy_source"] = "derived_from_d2_train_metrics"
                    d2_record["engine"] = "d2"
                    append_metrics_jsonl(self._metrics_path(), d2_record)

                if not self.skip_per_task_eval:
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} eval start | "
                        f"sliding_window={self.eval_sliding_window} | "
                        f"max_samples={self.eval_max_samples_per_task}"
                    )
                    eval_progress = _D2TaskProgress(
                        task_id=task.task_id,
                        total_tasks=total_tasks,
                        task_total_iters=1,
                        show_train=False,
                    )
                    try:
                        eval_payload = run_detectron2_eval(
                            config_path=self.config_path,
                            output_dir=task_dir,
                            resume_task=task.task_id,
                            checkpoint=None,
                            open_vocab=self.open_vocab_eval,
                            seg_network=self.seg_net,
                            eval_sliding_window=self.eval_sliding_window,
                            eval_max_samples=self.eval_max_samples_per_task,
                            progress_callback=eval_progress.eval_callback,
                        )
                    finally:
                        eval_progress.close()
                    coverage_metrics = _compute_task_coverage_metrics(
                        task=task,
                        eval_payload=eval_payload,
                        task_main_iters=task_main_iters,
                    )
                    append_metrics_jsonl(
                        self._metrics_path(),
                        {
                            "task": float(task.task_id),
                            "phase": "eval",
                            "mIoU_all": _safe_float(eval_payload.get("mIoU_all")),
                            "mIoU_old": _safe_float(eval_payload.get("mIoU_old")),
                            "mIoU_new": _safe_float(eval_payload.get("mIoU_new")),
                            "BG-mIoU": _safe_float(eval_payload.get("BG-mIoU")),
                            "class_iou_all": eval_payload.get("class_iou_all") if isinstance(eval_payload.get("class_iou_all"), dict) else {},
                            "class_iou_old": eval_payload.get("class_iou_old") if isinstance(eval_payload.get("class_iou_old"), dict) else {},
                            "class_iou_new": eval_payload.get("class_iou_new") if isinstance(eval_payload.get("class_iou_new"), dict) else {},
                            "class_iou_bg": eval_payload.get("class_iou_bg") if isinstance(eval_payload.get("class_iou_bg"), dict) else {},
                            "engine": "d2",
                            **coverage_metrics,
                        },
                    )
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} eval result | "
                        f"mIoU_all={_format_metric(eval_payload.get('mIoU_all'))} | "
                        f"mIoU_old={_format_metric(eval_payload.get('mIoU_old'))} | "
                        f"mIoU_new={_format_metric(eval_payload.get('mIoU_new'))} | "
                        f"BG-mIoU={_format_metric(eval_payload.get('BG-mIoU'))}"
                    )

                    ov_pairs = []
                    for key in sorted(eval_payload.keys()):
                        if key.startswith("ov_mIoU_"):
                            ov_pairs.append(f"{key}={_format_metric(eval_payload.get(key))}")
                    if ov_pairs:
                        print(
                            "[open-continual] "
                            f"task {task.task_id}/{total_tasks} open-vocab result | "
                            + " | ".join(ov_pairs)
                        )

                    old_map = eval_payload.get("class_iou_old")
                    new_map = eval_payload.get("class_iou_new")
                    bg_map = eval_payload.get("class_iou_bg")
                    class_iou_all = eval_payload.get("class_iou_all")
                    old_count = len(old_map) if isinstance(old_map, dict) else 0
                    new_count = len(new_map) if isinstance(new_map, dict) else 0
                    bg_count = len(bg_map) if isinstance(bg_map, dict) else 0
                    old_avg = _mean_metric_map(old_map)
                    new_avg = _mean_metric_map(new_map)
                    bg_avg = _mean_metric_map(bg_map)

                    train_rows = _read_d2_metrics(task_dir)
                    loss_summary = _summarize_train_loss_fields(train_rows)
                    real_beta = _estimate_beta_star_from_train(train_rows)
                    real_ctr = _mean_from_keys(train_rows, ["loss_ctr", "ctr_loss", "loss_contrastive"])
                    real_alpha = None
                    if old_avg is not None and new_avg is not None and (old_avg + new_avg) > 0.0:
                        real_alpha = float(old_avg / (old_avg + new_avg))
                    elif new_avg is not None:
                        real_alpha = 0.0
                    real_tau = _mean_from_keys(train_rows, ["loss_sem_seg", "total_loss"])
                    real_overlap = _compute_class_iou_overlap(
                        class_iou_all if isinstance(class_iou_all, dict) else None,
                        [x for x in class_iou_history if isinstance(x, dict)],
                    )
                    real_continual = {
                        "beta_1_star": real_beta,
                        "ctr_loss": real_ctr,
                        "alpha_star": real_alpha,
                        "tau_pred": real_tau,
                        "omega_tau_t": real_overlap,
                    }
                    append_metrics_jsonl(
                        self._metrics_path(),
                        {
                            "task": float(task.task_id),
                            "phase": "continual_real",
                            "metric_source": "derived_from_real_artifacts",
                            "beta_1_star": _safe_float(real_continual.get("beta_1_star")),
                            "ctr_loss": _safe_float(real_continual.get("ctr_loss")),
                            "alpha_star": _safe_float(real_continual.get("alpha_star")),
                            "tau_pred": _safe_float(real_continual.get("tau_pred")),
                            "omega_tau_t": _safe_float(real_continual.get("omega_tau_t")),
                            "engine": "d2",
                        },
                    )
                    append_metrics_jsonl(
                        self._metrics_path(),
                        {
                            "task": float(task.task_id),
                            "phase": "task_summary",
                            "engine": "d2",
                            "beta_1_star": _safe_float(real_continual.get("beta_1_star")),
                            "ctr_loss": _safe_float(real_continual.get("ctr_loss")),
                            "alpha_star": _safe_float(real_continual.get("alpha_star")),
                            "tau_pred": _safe_float(real_continual.get("tau_pred")),
                            "omega_tau_t": _safe_float(real_continual.get("omega_tau_t")),
                            "loss_sem_seg": _safe_float(loss_summary.get("loss_sem_seg")),
                            "loss_old_kd": _safe_float(loss_summary.get("loss_old_kd")),
                            "loss_old_clip": _safe_float(loss_summary.get("loss_old_clip")),
                            "loss_unseen_clip": _safe_float(loss_summary.get("loss_unseen_clip")),
                            "loss_ciba": _safe_float(loss_summary.get("loss_ciba")),
                            "loss_ctr": _safe_float(loss_summary.get("loss_ctr")),
                        },
                    )
                    if isinstance(class_iou_all, dict) and class_iou_all:
                        class_iou_history.append(class_iou_all)

                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} continual-real | "
                        f"beta_1_star={_format_metric(real_continual.get('beta_1_star'), precision=4)} | "
                        f"ctr_loss={_format_metric(real_continual.get('ctr_loss'), precision=4)} | "
                        f"alpha_star={_format_metric(real_continual.get('alpha_star'), precision=4)} | "
                        f"tau_pred={_format_metric(real_continual.get('tau_pred'), precision=4)} | "
                        f"omega_tau_t={_format_metric(real_continual.get('omega_tau_t'), precision=4)}"
                    )
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} continual-real summary | "
                        f"old_count={old_count} old_mIoU={_format_metric(old_avg)} | "
                        f"new_count={new_count} new_mIoU={_format_metric(new_avg)} | "
                        f"bg_count={bg_count} bg_mIoU={_format_metric(bg_avg)}"
                    )

                    if isinstance(class_iou_all, dict) and class_iou_all:
                        old_names = set(old_map.keys()) if isinstance(old_map, dict) else set()
                        new_names = set(new_map.keys()) if isinstance(new_map, dict) else set()
                        bg_names = set(bg_map.keys()) if isinstance(bg_map, dict) else set()
                        print(
                            "[open-continual] "
                            f"task {task.task_id}/{total_tasks} class IoU detail | "
                            f"count={len(class_iou_all)}"
                        )
                        for class_name in sorted(class_iou_all):
                            if class_name in new_names:
                                group = "new"
                            elif class_name in old_names:
                                group = "old"
                            elif class_name in bg_names:
                                group = "bg"
                            else:
                                group = "other"
                            print(
                                "[open-continual] "
                                f"task {task.task_id}/{total_tasks} class IoU | "
                                f"group={group} | class={class_name} | IoU={float(class_iou_all[class_name]):.3f}"
                            )
                else:
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} eval skipped | "
                        "per-task eval disabled"
                    )
            else:
                for record in (p1, p2, p3, p4):
                    mock_record = dict(record)
                    mock_record["proxy_source"] = "mock_training_loop"
                    mock_record["engine"] = "mock"
                    append_metrics_jsonl(self._metrics_path(), mock_record)
                if self._mock_model is not None:
                    all_task_classes = [
                        int(c)
                        for t in plan.tasks
                        for c in (t.seen_classes + t.new_classes + t.background_classes)
                    ]
                    num_classes = max(all_task_classes, default=-1) + 1
                    mock_eval_payload = _compute_mock_task_eval_metrics(
                        model=self._mock_model,
                        task=task,
                        num_classes=max(1, num_classes),
                        seed=self.seed + task.task_id * 100,
                    )
                    append_metrics_jsonl(
                        self._metrics_path(),
                        {
                            "task": float(task.task_id),
                            "phase": "eval",
                            "mIoU_all": _safe_float(mock_eval_payload.get("mIoU_all")),
                            "mIoU_old": _safe_float(mock_eval_payload.get("mIoU_old")),
                            "mIoU_new": _safe_float(mock_eval_payload.get("mIoU_new")),
                            "BG-mIoU": _safe_float(mock_eval_payload.get("BG-mIoU")),
                            "engine": "mock",
                        },
                    )
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} eval result | "
                        f"mIoU_all={_format_metric(mock_eval_payload.get('mIoU_all'))} | "
                        f"mIoU_old={_format_metric(mock_eval_payload.get('mIoU_old'))} | "
                        f"mIoU_new={_format_metric(mock_eval_payload.get('mIoU_new'))} | "
                        f"BG-mIoU={_format_metric(mock_eval_payload.get('BG-mIoU'))}"
                    )
                    preview_path = _save_mock_inference_preview(
                        task_dir=task_dir,
                        model=self._mock_model,
                        task=task,
                        num_classes=max(1, num_classes),
                        seed=self.seed + task.task_id * 200,
                    )
                    if preview_path is not None:
                        print(
                            "[open-continual] "
                            f"task {task.task_id}/{total_tasks} inference preview | "
                            f"path={preview_path}"
                        )

            if balanced_enabled:
                current_eval = {
                    "mIoU_all": _safe_float((eval_payload or {}).get("mIoU_all")),
                    "mIoU_old": _safe_float((eval_payload or {}).get("mIoU_old")),
                    "mIoU_new": _safe_float((eval_payload or {}).get("mIoU_new")),
                }
                prev_eval = {
                    "mIoU_all": _safe_float(balanced_prev_eval.get("mIoU_all")),
                    "mIoU_old": _safe_float(balanced_prev_eval.get("mIoU_old")),
                    "mIoU_new": _safe_float(balanced_prev_eval.get("mIoU_new")),
                }

                required_metrics = ("mIoU_all", "mIoU_old", "mIoU_new")
                has_finite_current = all(current_eval[key] is not None for key in required_metrics)
                has_finite_prev = all(prev_eval[key] is not None for key in required_metrics)

                delta_new = None
                delta_old = None
                delta_all = None
                ov_min_delta = None

                if has_finite_current and has_finite_prev:
                    delta_new = float(current_eval["mIoU_new"] - prev_eval["mIoU_new"])
                    delta_old = float(current_eval["mIoU_old"] - prev_eval["mIoU_old"])
                    delta_all = float(current_eval["mIoU_all"] - prev_eval["mIoU_all"])
                    ov_min_delta = float(min(delta_old, delta_all))
                    signals = {
                        "delta_new": delta_new,
                        "delta_old": delta_old,
                        "delta_all": delta_all,
                        "ov_min_delta": ov_min_delta,
                        "old_constraint_violated": delta_old < -self.epsilon_old,
                        "all_constraint_violated": delta_all < -self.epsilon_all,
                    }
                    balanced_state = update_controller_state(
                        state=balanced_state,
                        cfg=balanced_cfg,
                        signals=signals,
                    )

                append_metrics_jsonl(
                    self._metrics_path(),
                    {
                        "task": float(task.task_id),
                        "phase": "balanced_ctrl",
                        "delta_new": delta_new,
                        "delta_old": delta_old,
                        "delta_all": delta_all,
                        "ov_min_delta": ov_min_delta,
                        "alpha_floor": _safe_float(balanced_state.alpha_floor),
                        "ov_guard_triggered": bool(balanced_state.ov_guard_triggered),
                        "ov_guard_state": bool(balanced_state.ov_guard_state),
                    },
                )

                if has_finite_current:
                    balanced_prev_eval = current_eval

            method_state = method.after_task(task_state={"task_id": task.task_id})
            alpha_star_history.append(_safe_float_or(real_continual.get("alpha_star"), 0.5))
            tau_pred_history.append(_safe_float_or(real_continual.get("tau_pred"), 1.0))
            if current_basis:
                basis_history.append(current_basis)
            latest_model_path = state.get("latest_model_path")
            latest_mock_model_path = state.get("latest_mock_model_path")
            if self.use_real_training and model is not None:
                latest_model_path = str((task_dir / "model.pt").resolve())
            elif self.engine == "d2":
                latest_model_path = str(task_dir / "model_final.pth")
            elif self.engine == "mock":
                assert self._mock_model is not None
                task_dir.mkdir(parents=True, exist_ok=True)
                mock_model_path = (task_dir / "mock_model.pt").resolve()
                torch.save({"model_state_dict": self._mock_model.state_dict()}, mock_model_path)
                latest_mock_model_path = str(mock_model_path)
                latest_model_path = latest_mock_model_path
                self._write_json(
                    self.output_dir / f"checkpoint_task_{task.task_id:03d}.json",
                    {
                        "config": self.config_path,
                        "seed": self.seed,
                        "resume_task": task.task_id - 1,
                        "last_task": task.task_id,
                        "engine": "mock",
                        "mock_model_path": latest_mock_model_path,
                    },
                )
            state = {
                "current_task": task.task_id,
                "method": method.name,
                "clip_finetune": self.clip_finetune,
                "alpha_star": _safe_float_or(real_continual.get("alpha_star"), 0.5),
                "tau_pred": _safe_float_or(real_continual.get("tau_pred"), 1.0),
                "alpha_star_history": alpha_star_history,
                "tau_pred_history": tau_pred_history,
                "basis_history": basis_history,
                "class_iou_history": class_iou_history,
                "method_state": method_state.values,
                "last_task_dir": str(task_dir),
                "latest_model_path": latest_model_path,
            }
            if latest_mock_model_path is not None:
                state["latest_mock_model_path"] = latest_mock_model_path
            if balanced_enabled:
                state["balanced_controller"] = {
                    "alpha_floor": _safe_float(balanced_state.alpha_floor),
                    "g_stab": _safe_float(balanced_state.g_stab),
                    "rho_old": _safe_float(balanced_state.rho_old),
                    "rho_new": _safe_float(balanced_state.rho_new),
                    "w_ctr": _safe_float(balanced_state.w_ctr),
                    "ov_guard_triggered": bool(balanced_state.ov_guard_triggered),
                    "ov_guard_state": bool(balanced_state.ov_guard_state),
                }
                state["balanced_prev_eval"] = {
                    "mIoU_all": _safe_float(balanced_prev_eval.get("mIoU_all")),
                    "mIoU_old": _safe_float(balanced_prev_eval.get("mIoU_old")),
                    "mIoU_new": _safe_float(balanced_prev_eval.get("mIoU_new")),
                }
            self._write_json(self._state_path(), state)
            if self.engine != "d2":
                print(
                    "[open-continual] "
                    f"task {task.task_id}/{total_tasks} done | remaining_task_iters=0 | "
                    f"remaining_tasks={remaining_tasks}"
                )
            task_pbar.update(1)
            task_pbar.set_postfix({
                "task": f"{task.task_id}/{total_tasks}",
                "loss": f"{_safe_float_or(real_continual.get('alpha_star'), 0.0):.3f}",
            })
            completed += 1

        task_pbar.close()

        if completed > 0:
            try:
                from covl_seg.engine.report_generator import generate_report
                generated = generate_report(run_dir=self.output_dir)
                _log.info("[report] generated %d figures → %s", len(generated), self.output_dir / "analysis")
            except Exception as exc:
                _log.warning("[report] chart generation skipped: %s", exc)

        return {
            "tasks_executed": float(completed),
            "last_task": float(state.get("current_task", self.resume_task)),
        }
