from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

from covl_seg.model import BoundaryDetector, COVLSegModel, ContinualBackbone, FusionHead, HCIBAHead

from .hooks import append_metrics_jsonl


def detectron2_available() -> bool:
    try:
        import detectron2  # noqa: F401
    except Exception:
        return False
    return True


class _COVLTrainingModel(nn.Module):
    def __init__(self, core_model: COVLSegModel, text_embeddings: torch.Tensor):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("text_embeddings", text_embeddings)

    def forward(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        targets = torch.stack([x["sem_seg"] for x in batch], dim=0).long()
        outputs = self.core_model(images=images, text_embeddings=self.text_embeddings, targets=targets)
        return {"loss_total": outputs["loss"]}


def build_covl_training_model(num_classes: int = 4, text_dim: int = 16, seed: int = 0) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    text_embeddings = torch.randn(num_classes, text_dim, generator=g)
    core = COVLSegModel(
        backbone=ContinualBackbone(in_channels=3, hidden_dim=32, out_dim=32),
        hciba_head=HCIBAHead(in_dim=32, out_dim=text_dim),
        boundary_detector=BoundaryDetector(threshold=0.15),
        fusion_head=FusionHead(alpha=0.5, tau=1.0),
        num_classes=num_classes,
        text_dim=text_dim,
    )
    return _COVLTrainingModel(core_model=core, text_embeddings=text_embeddings)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _d2_project_root() -> Path:
    env_root = os.environ.get("COVL_SEG_D2_PROJECT_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise RuntimeError(f"Configured Detectron2 project root does not exist: {root}")
        return root

    default_root = _workspace_root() / "covl_seg" / "vendor" / "covl_seg_d2_runtime"
    if default_root.exists():
        return default_root

    raise RuntimeError(
        "Unable to locate Detectron2 project root. "
        "Expected internal runtime at covl_seg/vendor/covl_seg_d2_runtime or set "
        "COVL_SEG_D2_PROJECT_ROOT to an external project root."
    )


def _d2_entrypoint() -> str:
    entrypoint = os.environ.get("COVL_SEG_D2_ENTRYPOINT", "").strip()
    return entrypoint if entrypoint else "train_net.py"


_SEG_NET_TO_CONFIG = {
    "vitb": "vitb_384.yaml",
    "vitl": "vitl_336.yaml",
    "r50": "r50_384.yaml",
    "r101": "r101_384.yaml",
    "swin_t": "swin_t_384.yaml",
    "swin_b": "swin_b_384.yaml",
}


_EXPERIMENT_SPECS: Dict[str, Dict[str, str]] = {
    "coco": {
        "alias": "coco",
        "class_json": "datasets/coco.json",
        "dataset_name": "coco_2017_test_stuff_all_sem_seg",
        "num_classes": "164",
    },
    "ade150": {
        "alias": "ade150",
        "class_json": "datasets/ade150.json",
        "dataset_name": "ade20k_150_test_sem_seg",
        "num_classes": "150",
    },
}


def _match_dataset_family(value: str) -> Optional[str]:
    lowered = value.lower()
    matches = []

    if re.search(r"(^|[^a-z0-9])coco([^a-z0-9]|$)", lowered):
        matches.append("coco")
    if re.search(r"(^|[^a-z0-9])ade(?:20k|15|100|150)?([^a-z0-9]|$)", lowered):
        matches.append("ade150")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous dataset family in '{value}'")
    if matches:
        return matches[0]
    return None


def _infer_seg_network(config_path: str) -> str:
    lowered = config_path.lower()
    if "vitl" in lowered:
        return "vitl"
    if "r101" in lowered:
        return "r101"
    if "r50" in lowered:
        return "r50"
    if "swin_t" in lowered:
        return "swin_t"
    if "swin_b" in lowered:
        return "swin_b"
    return "vitb"


def _read_covl_config_text(config_path: str) -> str:
    candidate = Path(config_path)
    if candidate.exists():
        try:
            return candidate.read_text(encoding="utf-8")
        except OSError:
            return ""

    workspace_candidate = (_workspace_root() / config_path).resolve()
    if workspace_candidate.exists():
        try:
            return workspace_candidate.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _resolve_experiment_spec(config_path: str) -> Dict[str, str]:
    config_text = _read_covl_config_text(config_path)
    dataset_match = re.search(r"^\s*DATASET\s*:\s*([^\s#]+)", config_text, flags=re.MULTILINE)
    if dataset_match is not None:
        dataset_name = dataset_match.group(1).strip().lower()
        config_family = _match_dataset_family(dataset_name)
        if config_family is None:
            raise ValueError(f"Unsupported dataset family in '{dataset_name}'")
        return dict(_EXPERIMENT_SPECS[config_family])

    path_family = _match_dataset_family(config_path)
    if path_family is None:
        raise ValueError(f"Unsupported dataset family for '{config_path}'")
    return dict(_EXPERIMENT_SPECS[path_family])


def _select_base_eval_spec(config_path: str) -> Dict[str, str]:
    spec = _resolve_experiment_spec(config_path)
    return {
        "alias": spec["alias"],
        "class_json": spec["class_json"],
        "dataset_name": spec["dataset_name"],
    }


def _resolve_d2_config(config_path: str, project_root: Path, seg_network: Optional[str] = None) -> Path:
    workspace = _workspace_root()
    covl_config_root = (workspace / "covl_seg" / "configs").resolve()

    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    if candidate.exists():
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(covl_config_root)):
            return resolved

    selected_network = (
        (seg_network or "").strip().lower()
        or os.environ.get("COVL_SEG_D2_SEG_NET", "").strip().lower()
        or _infer_seg_network(config_path)
    )
    fallback_name = _SEG_NET_TO_CONFIG.get(selected_network)
    if fallback_name is None:
        valid = ", ".join(sorted(_SEG_NET_TO_CONFIG))
        raise RuntimeError(
            f"Unsupported segmentation network preset '{selected_network}'. Valid presets: {valid}"
        )
    fallback = project_root / "configs" / fallback_name
    if not fallback.exists():
        raise RuntimeError(
            f"Unable to resolve Detectron2 config. Missing config file: {fallback}. "
            "You can set COVL_SEG_D2_SEG_NET to another preset or pass an explicit --config path."
        )
    return fallback


def _run_d2_command_stream(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    on_output: Callable[[str], None],
) -> None:
    tail: List[str] = []
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to stream Detectron2 command output")

    for line in process.stdout:
        text = line.rstrip("\n")
        tail.append(text)
        if len(tail) > 40:
            tail.pop(0)
        on_output(text)

    rc = process.wait()
    if rc != 0:
        details = ""
        if tail:
            details = "\noutput tail:\n" + "\n".join(tail)
        raise RuntimeError(
            "Detectron2 command failed with exit code "
            f"{rc}: {' '.join(cmd)}{details}"
        )


def _run_d2_command(
    cmd: List[str],
    cwd: Path,
    on_output: Optional[Callable[[str], None]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    env.setdefault("DETECTRON2_DATASETS", str(_workspace_root() / "datasets"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    workspace = str(_workspace_root())
    pythonpath_entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if workspace not in pythonpath_entries:
        env["PYTHONPATH"] = os.pathsep.join([workspace] + pythonpath_entries)
    if env_overrides:
        env.update(env_overrides)

    if on_output is not None:
        _run_d2_command_stream(cmd=cmd, cwd=cwd, env=env, on_output=on_output)
        return

    try:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = ""
        if getattr(exc, "stderr", None):
            details = f"\nstderr:\n{exc.stderr}"
        elif getattr(exc, "output", None):
            details = f"\noutput:\n{exc.output}"
        raise RuntimeError(
            "Detectron2 command failed with exit code "
            f"{exc.returncode}: {' '.join(cmd)}{details}"
        ) from exc


def _is_cuda_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message


def _is_nan_loss_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "loss became infinite or nan" in message or "loss_sem_seg': nan" in message


def _low_mem_override_profiles() -> List[List[str]]:
    return [
        [
            "SOLVER.IMS_PER_BATCH",
            "1",
            "SOLVER.AMP.ENABLED",
            "False",
            "SOLVER.BASE_LR",
            "0.0001",
            "SOLVER.CLIP_GRADIENTS.CLIP_VALUE",
            "0.005",
            "DATALOADER.NUM_WORKERS",
            "2",
            "INPUT.CROP.SIZE",
            "(320,320)",
            "INPUT.MIN_SIZE_TRAIN",
            "(320,)",
        ],
        [
            "SOLVER.IMS_PER_BATCH",
            "1",
            "SOLVER.AMP.ENABLED",
            "False",
            "SOLVER.BASE_LR",
            "0.00008",
            "SOLVER.CLIP_GRADIENTS.CLIP_VALUE",
            "0.003",
            "DATALOADER.NUM_WORKERS",
            "2",
            "INPUT.CROP.SIZE",
            "(256,256)",
            "INPUT.MIN_SIZE_TRAIN",
            "(256,)",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
        ],
    ]


def _default_stable_train_overrides() -> List[str]:
    return [
        "SOLVER.IMS_PER_BATCH",
        "1",
        "SOLVER.AMP.ENABLED",
        "False",
        "SOLVER.BASE_LR",
        "0.0001",
        "SOLVER.CLIP_GRADIENTS.CLIP_VALUE",
        "0.005",
        "DATALOADER.NUM_WORKERS",
        "2",
        "INPUT.CROP.SIZE",
        "(256,256)",
        "INPUT.MIN_SIZE_TRAIN",
        "(256,)",
        "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
        "[1,1]",
    ]


def _stability_override_profiles() -> List[List[str]]:
    return [
        [
            "SOLVER.IMS_PER_BATCH",
            "1",
            "SOLVER.AMP.ENABLED",
            "False",
            "SOLVER.BASE_LR",
            "0.0001",
            "SOLVER.CLIP_GRADIENTS.CLIP_VALUE",
            "0.005",
            "DATALOADER.NUM_WORKERS",
            "2",
            "INPUT.CROP.SIZE",
            "(256,256)",
            "INPUT.MIN_SIZE_TRAIN",
            "(256,)",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
        ]
    ]


def _load_eval_sem_seg(eval_output: Path) -> Dict[str, float]:
    eval_pth = eval_output / "inference" / "sem_seg_evaluation.pth"
    if not eval_pth.exists():
        return {}

    raw = torch.load(eval_pth, map_location="cpu")
    sem_seg = raw.get("sem_seg") if isinstance(raw, dict) else None
    if not isinstance(sem_seg, dict):
        sem_seg = raw if isinstance(raw, dict) else {}
    return sem_seg


def _read_index_list(path: Path) -> List[int]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    values: List[int] = []
    for item in payload:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return values


def _resolve_class_names(project_root: Path, class_json: str) -> List[str]:
    candidate = Path(class_json)
    if not candidate.is_absolute():
        local_candidate = (project_root / class_json).resolve()
        if local_candidate.exists():
            candidate = local_candidate
        else:
            candidate = (_workspace_root() / class_json).resolve()
    if not candidate.exists():
        return []
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    names: List[str] = []
    for item in payload:
        if isinstance(item, str):
            names.append(item)
    return names


def _avg(values: List[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _derive_split_metrics(
    eval_output: Path,
    project_root: Path,
    class_json: str,
    sem_seg: Dict[str, float],
) -> Dict[str, Any]:
    task_dir = eval_output.parent.parent
    splits_dir = task_dir / "splits"
    seen_indexes = _read_index_list(splits_dir / "seen_indexes.json")
    new_indexes = _read_index_list(splits_dir / "new_indexes.json")
    bg_indexes = _read_index_list(splits_dir / "unseen_indexes.json")
    if not seen_indexes or not new_indexes:
        return {}

    class_names = _resolve_class_names(project_root=project_root, class_json=class_json)
    if not class_names:
        return {}

    new_set = set(new_indexes)
    old_indexes = [idx for idx in seen_indexes if idx not in new_set]

    def _gather(indexes: List[int]) -> List[float]:
        values: List[float] = []
        for idx in indexes:
            if idx < 0 or idx >= len(class_names):
                continue
            key = f"IoU-{class_names[idx]}"
            value = sem_seg.get(key)
            if value is None:
                continue
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            values.append(value_f)
        return values

    derived: Dict[str, Any] = {}
    old_mean = _avg(_gather(old_indexes))
    new_mean = _avg(_gather(new_indexes))
    bg_mean = _avg(_gather(bg_indexes))
    if old_mean is not None:
        derived["mIoU_old"] = old_mean
    if new_mean is not None:
        derived["mIoU_new"] = new_mean
    if bg_mean is not None:
        derived["BG-mIoU"] = bg_mean

    old_name_set = {class_names[idx] for idx in old_indexes if 0 <= idx < len(class_names)}
    new_name_set = {class_names[idx] for idx in new_indexes if 0 <= idx < len(class_names)}
    bg_name_set = {class_names[idx] for idx in bg_indexes if 0 <= idx < len(class_names)}

    class_iou_all: Dict[str, float] = {}
    class_iou_old: Dict[str, float] = {}
    class_iou_new: Dict[str, float] = {}
    class_iou_bg: Dict[str, float] = {}
    for key, value in sem_seg.items():
        if not key.startswith("IoU-"):
            continue
        class_name = key[4:]
        score = float(value)
        if not math.isfinite(score):
            continue
        class_iou_all[class_name] = score
        if class_name in old_name_set:
            class_iou_old[class_name] = score
        elif class_name in new_name_set:
            class_iou_new[class_name] = score
        elif class_name in bg_name_set:
            class_iou_bg[class_name] = score

    if class_iou_all:
        derived["class_iou_all"] = class_iou_all
    if class_iou_old:
        derived["class_iou_old"] = class_iou_old
    if class_iou_new:
        derived["class_iou_new"] = class_iou_new
    if class_iou_bg:
        derived["class_iou_bg"] = class_iou_bg
    return derived


def _extract_eval_metrics(
    eval_output: Path,
    project_root: Optional[Path] = None,
    class_json: Optional[str] = None,
) -> Dict[str, Any]:
    sem_seg = _load_eval_sem_seg(eval_output)
    if not sem_seg:
        return {}

    resolved: Dict[str, Any] = {}

    def _pick(*keys: str) -> Optional[float]:
        for key in keys:
            value = sem_seg.get(key)
            if value is not None:
                value_f = float(value)
                if math.isfinite(value_f):
                    return value_f
        return None

    all_miou = _pick("mIoU")
    old_miou = _pick("mIoU_old", "seen_IoU")
    new_miou = _pick("mIoU_new", "unseen_IoU")
    bg_miou = _pick("BG-mIoU", "IoU-background", "IoU-bg", "IoU-background, ground")

    if all_miou is not None:
        resolved["mIoU_all"] = all_miou
    if old_miou is not None:
        resolved["mIoU_old"] = old_miou
    if new_miou is not None:
        resolved["mIoU_new"] = new_miou
    if bg_miou is not None:
        resolved["BG-mIoU"] = bg_miou

    if project_root is not None and class_json is not None:
        derived = _derive_split_metrics(
            eval_output=eval_output,
            project_root=project_root,
            class_json=class_json,
            sem_seg=sem_seg,
        )
        for key, value in derived.items():
            if key not in resolved:
                resolved[key] = value
    return resolved


_EVAL_DATASET_LAYOUTS: Dict[str, Dict[str, str]] = {
    "ade20k_150_test_sem_seg": {
        "image_rel": "ADEChallengeData2016/images/validation",
        "gt_rel": "ADEChallengeData2016/annotations_detectron2/validation",
        "image_ext": "jpg",
        "gt_ext": "png",
    },
    "coco_2017_test_stuff_all_sem_seg": {
        "image_rel": "coco-stuff/images/val2017",
        "gt_rel": "coco-stuff/annotations_detectron2/val2017",
        "image_ext": "jpg",
        "gt_ext": "png",
    },
    "context_59_test_sem_seg": {
        "image_rel": "VOCdevkit/VOC2010/JPEGImages",
        "gt_rel": "VOCdevkit/VOC2010/annotations_detectron2/pc59_val",
        "image_ext": "jpg",
        "gt_ext": "png",
    },
    "context_459_test_sem_seg": {
        "image_rel": "VOCdevkit/VOC2010/JPEGImages",
        "gt_rel": "VOCdevkit/VOC2010/annotations_detectron2/pc459_val",
        "image_ext": "jpg",
        "gt_ext": "tif",
    },
    "voc_2012_test_sem_seg": {
        "image_rel": "VOCdevkit/VOC2012/JPEGImages",
        "gt_rel": "VOCdevkit/VOC2012/annotations_detectron2/val",
        "image_ext": "jpg",
        "gt_ext": "png",
    },
}


def _build_eval_subset_root(
    dataset_name: str,
    source_root: Path,
    subset_root: Path,
    max_samples: int,
) -> Optional[Path]:
    if max_samples <= 0:
        return None
    layout = _EVAL_DATASET_LAYOUTS.get(dataset_name)
    if layout is None:
        return None

    source_image_dir = source_root / layout["image_rel"]
    source_gt_dir = source_root / layout["gt_rel"]
    if not source_image_dir.is_dir() or not source_gt_dir.is_dir():
        return None

    selected_stems: List[str] = []
    for gt in sorted(source_gt_dir.glob(f"*.{layout['gt_ext']}")):
        stem = gt.stem
        image_file = source_image_dir / f"{stem}.{layout['image_ext']}"
        if not image_file.exists():
            continue
        selected_stems.append(stem)
        if len(selected_stems) >= max_samples:
            break

    if not selected_stems:
        return None

    if subset_root.exists():
        shutil.rmtree(subset_root)

    subset_image_dir = subset_root / layout["image_rel"]
    subset_gt_dir = subset_root / layout["gt_rel"]
    subset_image_dir.mkdir(parents=True, exist_ok=True)
    subset_gt_dir.mkdir(parents=True, exist_ok=True)

    for stem in selected_stems:
        src_image = source_image_dir / f"{stem}.{layout['image_ext']}"
        src_gt = source_gt_dir / f"{stem}.{layout['gt_ext']}"
        dst_image = subset_image_dir / src_image.name
        dst_gt = subset_gt_dir / src_gt.name
        try:
            os.symlink(src_image, dst_image)
            os.symlink(src_gt, dst_gt)
        except OSError:
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_gt, dst_gt)

    return subset_root


def _resolve_datasets_root_for_eval() -> Path:
    configured = Path(os.environ.get("DETECTRON2_DATASETS", str(_workspace_root() / "datasets")))
    if configured.is_absolute():
        return configured
    return (_workspace_root() / configured).resolve()


def _run_single_eval(
    project_root: Path,
    entrypoint: str,
    config_file: Path,
    output_dir: Path,
    weights: str,
    class_json: str,
    dataset_name: str,
    eval_sliding_window: bool,
    eval_max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    env_overrides: Dict[str, str] = {}
    if eval_max_samples is not None:
        source_datasets_root = _resolve_datasets_root_for_eval()
        subset_root = output_dir.resolve() / "_subset_datasets"
        built_subset = _build_eval_subset_root(
            dataset_name=dataset_name,
            source_root=source_datasets_root,
            subset_root=subset_root,
            max_samples=eval_max_samples,
        )
        if built_subset is not None:
            env_overrides["DETECTRON2_DATASETS"] = str(built_subset)

    cmd = [
        "python",
        entrypoint,
        "--config-file",
        str(config_file),
        "--num-gpus",
        "1",
        "--dist-url",
        "auto",
        "--eval-only",
        "OUTPUT_DIR",
        str(output_dir),
        "MODEL.WEIGHTS",
        str(weights),
        "TEST.SLIDING_WINDOW",
        "True" if eval_sliding_window else "False",
        "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
        "[1,1]",
        "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON",
        class_json,
        "DATASETS.TEST",
        f'("{dataset_name}",)',
    ]
    _run_d2_command(
        cmd=cmd,
        cwd=project_root,
        on_output=progress_callback,
        env_overrides=env_overrides or None,
    )
    return _extract_eval_metrics(eval_output=output_dir, project_root=project_root, class_json=class_json)


def _extract_train_records(output_dir: Path, resume_task: int, num_tasks: int) -> int:
    source = output_dir / "metrics.json"
    target = output_dir / "metrics.jsonl"
    if not source.exists():
        append_metrics_jsonl(
            target,
            {
                "task": float(resume_task + num_tasks),
                "phase": "d2_train",
                "loss": 0.0,
                "engine": "d2",
            },
        )
        return 1

    num_records = 0
    phase_names = ["phase1", "phase2", "phase3", "phase4"]
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        loss = record.get("total_loss")
        if loss is None:
            continue
        iter_value = int(record.get("iteration", num_records))
        append_metrics_jsonl(
            target,
            {
                "task": float(resume_task + num_tasks),
                "phase": phase_names[iter_value % 4],
                "iter": float(iter_value),
                "loss": float(loss),
                "engine": "d2",
            },
        )
        num_records += 1

    if num_records == 0:
        append_metrics_jsonl(
            target,
            {
                "task": float(resume_task + num_tasks),
                "phase": "d2_train",
                "loss": 0.0,
                "engine": "d2",
            },
        )
        return 1
    return num_records


def _lookup_override_value(overrides: List[str], key: str) -> Optional[str]:
    for idx in range(len(overrides) - 2, -1, -2):
        if overrides[idx] == key:
            return overrides[idx + 1]
    return None


def run_detectron2_train(
    config_path: str,
    output_dir: Path,
    seed: int,
    resume_task: int,
    max_tasks: Optional[int],
    seg_network: Optional[str] = None,
    extra_overrides: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    if not detectron2_available():
        raise RuntimeError("Detectron2 is not available in this environment")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = _d2_project_root()
    entrypoint = _d2_entrypoint()
    resolved_config = _resolve_d2_config(config_path, project_root=project_root, seg_network=seg_network)
    explicit_overrides = list(extra_overrides or [])
    if resume_task > 0 and _lookup_override_value(explicit_overrides, "MODEL.WEIGHTS") is None:
        raise RuntimeError(
            "Detectron2 training with resume_task > 0 requires explicit MODEL.WEIGHTS in extra_overrides."
        )

    base_cmd = [
        "python",
        entrypoint,
        "--config-file",
        str(resolved_config),
        "--num-gpus",
        "1",
        "--dist-url",
        "auto",
        "--resume",
        "OUTPUT_DIR",
        str(output_dir),
        "SEED",
        str(seed),
    ]
    default_overrides = _default_stable_train_overrides()
    cmd = base_cmd + default_overrides + explicit_overrides
    force_low_mem = os.environ.get("COVL_SEG_D2_FORCE_LOW_MEM", "0") == "1"
    if force_low_mem:
        last_exc: Optional[RuntimeError] = None
        for idx, profile in enumerate(_low_mem_override_profiles(), start=1):
            attempt_cmd = base_cmd + profile + explicit_overrides
            try:
                _run_d2_command(cmd=attempt_cmd, cwd=project_root, on_output=progress_callback)
                break
            except RuntimeError as exc:
                if not (_is_cuda_oom_error(exc) or _is_nan_loss_error(exc)):
                    raise
                last_exc = exc
                if idx == len(_low_mem_override_profiles()):
                    raise last_exc
    else:
        try:
            _run_d2_command(cmd=cmd, cwd=project_root, on_output=progress_callback)
        except RuntimeError as exc:
            if _is_cuda_oom_error(exc):
                last_exc = exc
                for idx, profile in enumerate(_low_mem_override_profiles(), start=1):
                    retry_cmd = base_cmd + profile + explicit_overrides
                    try:
                        _run_d2_command(cmd=retry_cmd, cwd=project_root, on_output=progress_callback)
                        break
                    except RuntimeError as retry_exc:
                        if not _is_cuda_oom_error(retry_exc):
                            raise
                        last_exc = retry_exc
                        if idx == len(_low_mem_override_profiles()):
                            raise last_exc
            elif _is_nan_loss_error(exc):
                last_exc = exc
                for idx, profile in enumerate(_stability_override_profiles(), start=1):
                    retry_cmd = base_cmd + profile + explicit_overrides
                    try:
                        _run_d2_command(cmd=retry_cmd, cwd=project_root, on_output=progress_callback)
                        break
                    except RuntimeError as retry_exc:
                        if not _is_nan_loss_error(retry_exc):
                            raise
                        last_exc = retry_exc
                        if idx == len(_stability_override_profiles()):
                            raise last_exc
            else:
                raise

    num_tasks = max_tasks if max_tasks is not None else 1

    num_phase_records = _extract_train_records(
        output_dir=output_dir,
        resume_task=resume_task,
        num_tasks=num_tasks,
    )

    last_task = resume_task + num_tasks
    checkpoint_payload = {
        "config": str(resolved_config),
        "seed": seed,
        "resume_task": resume_task,
        "last_task": last_task,
        "num_phase_records": num_phase_records,
        "engine": "d2",
        "backend": "d2_train_entrypoint",
    }
    loaded_weights = _lookup_override_value(explicit_overrides, "MODEL.WEIGHTS")
    if loaded_weights is not None:
        checkpoint_payload["loaded_weights"] = loaded_weights
    ckpt = output_dir / f"checkpoint_task_{last_task:03d}.json"
    ckpt.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")
    return {
        "num_tasks": num_tasks,
        "num_phase_records": num_phase_records,
        "last_task": last_task,
    }


def run_detectron2_eval(
    config_path: str,
    output_dir: Path,
    resume_task: int,
    checkpoint: Optional[str],
    open_vocab: bool,
    seg_network: Optional[str] = None,
    eval_sliding_window: bool = True,
    eval_max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not detectron2_available():
        raise RuntimeError("Detectron2 is not available in this environment")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = _d2_project_root()
    entrypoint = _d2_entrypoint()
    resolved_config = _resolve_d2_config(config_path, project_root=project_root, seg_network=seg_network)
    weights = checkpoint or str(output_dir / "model_final.pth")
    base_spec = _select_base_eval_spec(config_path)

    base_metrics = _run_single_eval(
        project_root=project_root,
        entrypoint=entrypoint,
        config_file=resolved_config,
        output_dir=output_dir / "eval" / base_spec["alias"],
        weights=str(weights),
        class_json=base_spec["class_json"],
        dataset_name=base_spec["dataset_name"],
        eval_sliding_window=eval_sliding_window,
        eval_max_samples=eval_max_samples,
        progress_callback=progress_callback,
    )
    miou = base_metrics.get("mIoU_all")

    payload: Dict[str, Any] = {
        "mIoU_all": None if miou is None else float(miou),
        "mIoU_old": None,
        "mIoU_new": None,
        "BG-mIoU": None,
        "resume_task": float(resume_task),
        "engine": "d2",
    }
    payload.update(base_metrics)

    if open_vocab:
        ov_specs = [
            ("pc59", "datasets/pc59.json", "context_59_test_sem_seg"),
            ("pc459", "datasets/pc459.json", "context_459_test_sem_seg"),
            ("voc20", "datasets/voc20.json", "voc_2012_test_sem_seg"),
        ]
        for alias, class_json, dataset_name in ov_specs:
            metrics = _run_single_eval(
                project_root=project_root,
                entrypoint=entrypoint,
                config_file=resolved_config,
                output_dir=output_dir / "eval" / alias,
                weights=str(weights),
                class_json=class_json,
                dataset_name=dataset_name,
                eval_sliding_window=eval_sliding_window,
                eval_max_samples=eval_max_samples,
                progress_callback=progress_callback,
            )
            ov_value = metrics.get("mIoU_all")
            payload[f"ov_mIoU_{alias}"] = None if ov_value is None else float(ov_value)

    payload["checkpoint"] = str(weights)
    payload["config"] = str(resolved_config)
    (output_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
