from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

from covl_seg.continual.methods import build_continual_method
from covl_seg.continual.task_partition import TaskDef, TaskPlan, build_task_plan
from covl_seg.engine.detectron2_runner import run_detectron2_eval, run_detectron2_train
from covl_seg.engine.hooks import append_metrics_jsonl
from covl_seg.engine.phase_runner import (
    run_phase1_hciba,
    run_phase2_joint,
    run_phase3_subspace_and_fusion,
    run_phase4_replay_update,
)


def _infer_num_classes_from_config(config_path: str) -> int:
    text = Path(config_path).read_text(encoding="utf-8").lower()
    if "ade20k_15" in text or "ade20k_100" in text:
        return 150
    if "coco" in text:
        return 164
    return 20


def _clip_overrides(mode: str) -> List[str]:
    lowered = mode.lower()
    if lowered not in {"none", "attention", "full"}:
        raise ValueError(f"Unsupported clip finetune mode: {mode}")
    return ["MODEL.SEM_SEG_HEAD.CLIP_FINETUNE", lowered]


def _write_task_class_indexes(task_dir: Path, task: TaskDef) -> Dict[str, Path]:
    split_dir = task_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_indexes = split_dir / "seen_indexes.json"
    new_indexes = split_dir / "new_indexes.json"
    test_indexes = split_dir / "unseen_indexes.json"
    train_indexes.write_text(json.dumps(task.seen_classes, indent=2), encoding="utf-8")
    new_indexes.write_text(json.dumps(task.new_classes, indent=2), encoding="utf-8")
    test_indexes.write_text(json.dumps(task.background_classes, indent=2), encoding="utf-8")
    return {"train": train_indexes, "new": new_indexes, "test": test_indexes}


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
    seg_net: Optional[str] = None
    open_vocab_eval: bool = False
    skip_per_task_eval: bool = False
    eval_sliding_window: bool = True
    eval_max_samples_per_task: Optional[int] = None
    resume_task: int = 0

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
            enable_ciba=args.enable_ciba,
            enable_ctr=args.enable_ctr,
            enable_spectral_ogp=args.enable_spectral_ogp,
            enable_sacr=args.enable_sacr,
            open_vocab_eval=args.open_vocab,
            skip_per_task_eval=args.skip_per_task_eval,
            eval_sliding_window=args.eval_sliding_window,
            eval_max_samples_per_task=args.eval_max_samples_per_task,
            resume_task=args.resume_task,
        )

    def _build_task_plan(self) -> TaskPlan:
        total_classes = _infer_num_classes_from_config(self.config_path)
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
            return
        prior_method = state.get("method")
        if prior_method is not None and prior_method != self.method_name:
            raise ValueError(
                f"Resume method mismatch: existing={prior_method}, requested={self.method_name}"
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

        completed = 0
        for task in plan.tasks:
            if task.task_id <= self.resume_task:
                continue
            if max_tasks is not None and completed >= max_tasks:
                break

            task_dir = self.output_dir / f"task_{task.task_id:03d}"
            total_tasks = len(plan.tasks)
            remaining_tasks = max(0, total_tasks - task.task_id)
            progress = _D2TaskProgress(
                task_id=task.task_id,
                total_tasks=total_tasks,
                task_total_iters=self.n_main,
            ) if self.engine == "d2" else None
            method.before_task(task_state={"task_id": task.task_id})
            method_phase_cfg = method.phase_overrides()
            phase_cfg = {**task_cfg, **method_phase_cfg}
            batch = _build_phase_batch(task)

            p1 = run_phase1_hciba(task.task_id, phase_cfg, batch=batch)
            p2 = run_phase2_joint(task.task_id, phase_cfg, batch=batch)
            p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=batch)
            p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=batch)

            current_basis = p3.get("subspace_basis", [])
            p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))

            class_index_paths = _write_task_class_indexes(task_dir=task_dir, task=task)
            task_overrides = clip_overrides + [
                "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES",
                str(class_index_paths["train"]),
                "MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES",
                str(class_index_paths["test"]),
                "SOLVER.MAX_ITER",
                str(self.n_main),
                "TEST.EVAL_PERIOD",
                "0",
            ]

            if self.engine == "d2":
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
                p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=d2_batch)
                p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=d2_batch)
                current_basis = p3.get("subspace_basis", [])
                p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))

                for record in (p1, p2, p3, p4):
                    record["proxy_source"] = "d2_metrics"
                    append_metrics_jsonl(self._metrics_path(), record)

                print(
                    "[open-continual] "
                    f"task {task.task_id}/{total_tasks} continual | "
                    f"beta_1_star={float(p1.get('beta_1_star', float('nan'))):.4f} | "
                    f"ctr_loss={float(p2.get('ctr_loss', float('nan'))):.4f} | "
                    f"alpha_star={float(p3.get('alpha_star', float('nan'))):.4f} | "
                    f"tau_pred={float(p3.get('tau_pred', float('nan'))):.4f} | "
                    f"omega_tau_t={float(p3.get('omega_tau_t', float('nan'))):.4f}"
                )

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
                    append_metrics_jsonl(
                        self._metrics_path(),
                        {
                            "task": float(task.task_id),
                            "phase": "eval",
                            "mIoU_all": float(eval_payload.get("mIoU_all", float("nan"))),
                            "mIoU_old": float(eval_payload.get("mIoU_old", float("nan"))),
                            "mIoU_new": float(eval_payload.get("mIoU_new", float("nan"))),
                            "BG-mIoU": float(eval_payload.get("BG-mIoU", float("nan"))),
                            "engine": "d2",
                        },
                    )
                    print(
                        "[open-continual] "
                        f"task {task.task_id}/{total_tasks} eval result | "
                        f"mIoU_all={float(eval_payload.get('mIoU_all', float('nan'))):.3f} | "
                        f"mIoU_old={float(eval_payload.get('mIoU_old', float('nan'))):.3f} | "
                        f"mIoU_new={float(eval_payload.get('mIoU_new', float('nan'))):.3f} | "
                        f"BG-mIoU={float(eval_payload.get('BG-mIoU', float('nan'))):.3f}"
                    )

                    class_iou_all = eval_payload.get("class_iou_all")
                    if isinstance(class_iou_all, dict) and class_iou_all:
                        old_map = eval_payload.get("class_iou_old")
                        new_map = eval_payload.get("class_iou_new")
                        bg_map = eval_payload.get("class_iou_bg")
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
                    record["proxy_source"] = "synthetic"
                    append_metrics_jsonl(self._metrics_path(), record)

            method_state = method.after_task(task_state={"task_id": task.task_id})
            alpha_star_history.append(float(p3["alpha_star"]))
            tau_pred_history.append(float(p3["tau_pred"]))
            if current_basis:
                basis_history.append(current_basis)
            state = {
                "current_task": task.task_id,
                "method": method.name,
                "clip_finetune": self.clip_finetune,
                "alpha_star": p3["alpha_star"],
                "tau_pred": p3["tau_pred"],
                "alpha_star_history": alpha_star_history,
                "tau_pred_history": tau_pred_history,
                "basis_history": basis_history,
                "method_state": method_state.values,
                "last_task_dir": str(task_dir),
            }
            self._write_json(self._state_path(), state)
            if self.engine != "d2":
                print(
                    "[open-continual] "
                    f"task {task.task_id}/{total_tasks} done | remaining_task_iters=0 | "
                    f"remaining_tasks={remaining_tasks}"
                )
            completed += 1

        return {
            "tasks_executed": float(completed),
            "last_task": float(state.get("current_task", self.resume_task)),
        }
