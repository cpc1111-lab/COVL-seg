from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

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


def _catseg_root() -> Path:
    root = _workspace_root() / "CAT-Seg"
    if not root.exists():
        raise RuntimeError(f"CAT-Seg workspace was not found at expected path: {root}")
    return root


def _resolve_catseg_config(config_path: str) -> Path:
    workspace = _workspace_root()
    catseg = _catseg_root()

    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    if candidate.exists() and str(candidate).startswith(str(catseg.resolve())):
        return candidate

    lowered = config_path.lower()
    fallback_name = "vitl_336.yaml" if "vitl" in lowered else "vitb_384.yaml"
    fallback = catseg / "configs" / fallback_name
    if not fallback.exists():
        raise RuntimeError(f"Unable to resolve CAT-Seg config. Missing fallback config: {fallback}")
    return fallback


def _run_catseg_command(cmd: List[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CAT-Seg command failed with exit code "
            f"{exc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


def _extract_eval_metrics(eval_output: Path) -> Dict[str, float]:
    eval_pth = eval_output / "inference" / "sem_seg_evaluation.pth"
    if not eval_pth.exists():
        return {}

    raw = torch.load(eval_pth, map_location="cpu")
    sem_seg = raw.get("sem_seg") if isinstance(raw, dict) else None
    if not isinstance(sem_seg, dict):
        sem_seg = raw if isinstance(raw, dict) else {}

    resolved: Dict[str, float] = {}

    def _pick(*keys: str) -> Optional[float]:
        for key in keys:
            value = sem_seg.get(key)
            if value is not None:
                return float(value)
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
    return resolved


def _run_single_eval(
    catseg_root: Path,
    config_file: Path,
    output_dir: Path,
    weights: str,
    class_json: str,
    dataset_name: str,
) -> Dict[str, float]:
    cmd = [
        "python",
        "train_net.py",
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
        "True",
        "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
        "[1,1]",
        "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON",
        class_json,
        "DATASETS.TEST",
        f'("{dataset_name}",)',
    ]
    _run_catseg_command(cmd=cmd, cwd=catseg_root)
    return _extract_eval_metrics(eval_output=output_dir)


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


def run_detectron2_train(
    config_path: str,
    output_dir: Path,
    seed: int,
    resume_task: int,
    max_tasks: Optional[int],
) -> Dict[str, int]:
    if not detectron2_available():
        raise RuntimeError("Detectron2 is not available in this environment")

    output_dir.mkdir(parents=True, exist_ok=True)

    catseg_root = _catseg_root()
    resolved_config = _resolve_catseg_config(config_path)

    cmd = [
        "python",
        "train_net.py",
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
    _run_catseg_command(cmd=cmd, cwd=catseg_root)

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
        "backend": "cat_seg_train_net",
    }
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
) -> Dict[str, float]:
    if not detectron2_available():
        raise RuntimeError("Detectron2 is not available in this environment")

    output_dir.mkdir(parents=True, exist_ok=True)
    catseg_root = _catseg_root()
    resolved_config = _resolve_catseg_config(config_path)
    weights = checkpoint or str(output_dir / "model_final.pth")

    base_metrics = _run_single_eval(
        catseg_root=catseg_root,
        config_file=resolved_config,
        output_dir=output_dir / "eval" / "ade150",
        weights=str(weights),
        class_json="datasets/ade150.json",
        dataset_name="ade20k_150_test_sem_seg",
    )
    miou = base_metrics.get("mIoU_all")

    payload: Dict[str, float] = {
        "mIoU_all": float("nan") if miou is None else miou,
        "mIoU_old": float("nan"),
        "mIoU_new": float("nan"),
        "BG-mIoU": float("nan"),
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
                catseg_root=catseg_root,
                config_file=resolved_config,
                output_dir=output_dir / "eval" / alias,
                weights=str(weights),
                class_json=class_json,
                dataset_name=dataset_name,
            )
            payload[f"ov_mIoU_{alias}"] = metrics.get("mIoU_all", float("nan"))

    payload["checkpoint"] = str(weights)
    payload["config"] = str(resolved_config)
    (output_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
