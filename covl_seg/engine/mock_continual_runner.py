from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from covl_seg.continual.task_partition import build_task_plan
from covl_seg.engine.evaluator import compute_basic_miou
from covl_seg.engine.hooks import append_metrics_jsonl
from covl_seg.engine.open_vocab_eval import OpenVocabEvaluator
from covl_seg.model import BoundaryDetector, COVLSegModel, ContinualBackbone, FusionHead, HCIBAHead


def infer_num_classes_from_config(config_path: str) -> int:
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parents[2] / config_path).resolve()

    lowered = config_path.lower()
    if candidate.exists():
        lowered = f"{lowered}\n{candidate.read_text(encoding='utf-8').lower()}"

    if "ade20k_15" in lowered or "ade20k_100" in lowered:
        return 150
    if "coco" in lowered:
        return 164
    return 20


def _build_model(num_classes: int, text_dim: int, seed: int) -> COVLSegModel:
    torch.manual_seed(seed)
    return COVLSegModel(
        backbone=ContinualBackbone(in_channels=3, hidden_dim=32, out_dim=64),
        hciba_head=HCIBAHead(in_dim=64, out_dim=text_dim),
        boundary_detector=BoundaryDetector(threshold=0.15),
        fusion_head=FusionHead(alpha=0.5, tau=1.0),
        num_classes=num_classes,
        text_dim=text_dim,
    )


def _make_class_tables(num_classes: int, text_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 1024)
    palette = torch.rand(num_classes, 3, generator=generator) * 2.0 - 1.0
    text_embeddings = torch.randn(num_classes, text_dim, generator=generator)
    text_embeddings = nn.functional.normalize(text_embeddings, dim=1)
    return palette, text_embeddings


def _sample_batch(
    class_ids: Sequence[int],
    palette: torch.Tensor,
    batch_size: int,
    image_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not class_ids:
        raise ValueError("class_ids must be non-empty")

    class_pool = torch.tensor(list(class_ids), dtype=torch.long)
    sampled = torch.randint(0, class_pool.shape[0], (batch_size, image_size, image_size), generator=generator)
    targets = class_pool[sampled]

    images = palette[targets]  # [B, H, W, 3]
    images = images + 0.08 * torch.randn(images.shape, generator=generator)
    images = images.permute(0, 3, 1, 2).contiguous()
    return images.float(), targets


def _train_phase(
    model: COVLSegModel,
    text_embeddings: torch.Tensor,
    palette: torch.Tensor,
    class_ids: Sequence[int],
    optimizer: torch.optim.Optimizer,
    iters: int,
    batch_size: int,
    image_size: int,
    generator: torch.Generator,
) -> float:
    losses: List[float] = []
    model.train()
    for _ in range(max(iters, 1)):
        images, targets = _sample_batch(
            class_ids=class_ids,
            palette=palette,
            batch_size=batch_size,
            image_size=image_size,
            generator=generator,
        )
        optimizer.zero_grad()
        outputs = model(images=images, text_embeddings=text_embeddings, targets=targets)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def _eval_miou(
    model: COVLSegModel,
    text_embeddings: torch.Tensor,
    palette: torch.Tensor,
    class_ids: Sequence[int],
    num_classes: int,
    batch_size: int,
    image_size: int,
    batches: int,
    generator: torch.Generator,
) -> float:
    if not class_ids:
        return float("nan")

    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(max(batches, 1)):
            images, targets = _sample_batch(
                class_ids=class_ids,
                palette=palette,
                batch_size=batch_size,
                image_size=image_size,
                generator=generator,
            )
            logits = model(images=images, text_embeddings=text_embeddings)["logits"]
            pred = logits.argmax(dim=1)
            total += compute_basic_miou(pred=pred, target=targets, num_classes=num_classes)
    return total / max(batches, 1)


def _task_splits(num_classes: int, num_tasks: int, seed: int) -> List[Dict[str, object]]:
    classes_per_task = max(1, (num_classes + num_tasks - 1) // num_tasks)
    plan = build_task_plan(
        task_spec=None,
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        all_classes=list(range(num_classes)),
        seed=seed,
    )
    return [
        {
            "task_id": task.task_id,
            "new_classes": list(task.new_classes),
            "seen_classes": list(task.seen_classes),
            "background_classes": list(task.background_classes),
        }
        for task in plan.tasks
    ]


def train_mock_continual(
    config_path: str,
    output_dir: Path,
    seed: int,
    resume_task: int,
    max_tasks: Optional[int],
) -> Dict[str, int]:
    num_tasks = max_tasks if max_tasks is not None else 1
    num_classes = infer_num_classes_from_config(config_path)
    text_dim = 64
    batch_size = 2
    image_size = 16
    phase1_iters = 4
    phase2_iters = 12

    total_tasks = max(resume_task + num_tasks, 1)
    task_splits = _task_splits(num_classes=num_classes, num_tasks=total_tasks, seed=seed)

    if resume_task > 0:
        resume_ckpt = _resolve_mock_ckpt(output_dir=output_dir, resume_task=resume_task, checkpoint=None)
        if resume_ckpt is None:
            raise FileNotFoundError(
                f"Cannot resume mock training from task {resume_task}: checkpoint not found in {output_dir}."
            )
        prior_state = torch.load(resume_ckpt, map_location="cpu")
        model = _build_model(num_classes=num_classes, text_dim=text_dim, seed=seed)
        model.load_state_dict(prior_state["model_state_dict"])
        palette = torch.tensor(prior_state["palette"], dtype=torch.float32)
        text_embeddings = torch.tensor(prior_state["text_embeddings"], dtype=torch.float32)
        if int(prior_state.get("num_classes", num_classes)) != num_classes:
            raise ValueError("Resume checkpoint class count does not match current config")
    else:
        model = _build_model(num_classes=num_classes, text_dim=text_dim, seed=seed)
        palette, text_embeddings = _make_class_tables(num_classes=num_classes, text_dim=text_dim, seed=seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics_path = output_dir / "metrics.jsonl"
    generator = torch.Generator().manual_seed(seed + 2048)
    phase_records = 0

    for task_id in range(resume_task + 1, resume_task + num_tasks + 1):
        split = task_splits[task_id - 1]
        seen_classes = [int(x) for x in split["seen_classes"]]
        new_classes = [int(x) for x in split["new_classes"]]
        old_classes = [c for c in seen_classes if c not in set(new_classes)]

        phase1_loss = _train_phase(
            model=model,
            text_embeddings=text_embeddings,
            palette=palette,
            class_ids=new_classes,
            optimizer=optimizer,
            iters=phase1_iters,
            batch_size=batch_size,
            image_size=image_size,
            generator=generator,
        )
        phase2_loss = _train_phase(
            model=model,
            text_embeddings=text_embeddings,
            palette=palette,
            class_ids=seen_classes,
            optimizer=optimizer,
            iters=phase2_iters,
            batch_size=batch_size,
            image_size=image_size,
            generator=generator,
        )

        phase3_miou = _eval_miou(
            model=model,
            text_embeddings=text_embeddings,
            palette=palette,
            class_ids=seen_classes,
            num_classes=num_classes,
            batch_size=batch_size,
            image_size=image_size,
            batches=3,
            generator=generator,
        )
        phase4_miou = _eval_miou(
            model=model,
            text_embeddings=text_embeddings,
            palette=palette,
            class_ids=old_classes or seen_classes,
            num_classes=num_classes,
            batch_size=batch_size,
            image_size=image_size,
            batches=3,
            generator=generator,
        )

        records = [
            {"task": float(task_id), "phase": "phase1", "loss": float(phase1_loss)},
            {"task": float(task_id), "phase": "phase2", "loss": float(phase2_loss)},
            {"task": float(task_id), "phase": "phase3", "loss": float(max(0.0, 100.0 - phase3_miou))},
            {"task": float(task_id), "phase": "phase4", "loss": float(max(0.0, 100.0 - phase4_miou))},
        ]
        for record in records:
            append_metrics_jsonl(metrics_path, record)
        phase_records += len(records)

        state_payload = {
            "seed": seed,
            "num_classes": num_classes,
            "text_dim": text_dim,
            "task_splits": task_splits,
            "palette": palette.tolist(),
            "text_embeddings": text_embeddings.tolist(),
            "model_state_dict": model.state_dict(),
        }
        model_ckpt_name = f"checkpoint_task_{task_id:03d}.pt"
        torch.save(state_payload, output_dir / model_ckpt_name)

        checkpoint_payload = {
            "config": config_path,
            "seed": seed,
            "resume_task": task_id - 1,
            "last_task": task_id,
            "num_phase_records": phase_records,
            "engine": "mock",
            "model_checkpoint": model_ckpt_name,
        }
        checkpoint_name = f"checkpoint_task_{task_id:03d}.json"
        (output_dir / checkpoint_name).write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

    last_task = resume_task + num_tasks

    return {
        "num_tasks": num_tasks,
        "num_phase_records": phase_records,
        "last_task": last_task,
    }


def _resolve_mock_ckpt(output_dir: Path, resume_task: int, checkpoint: Optional[str]) -> Optional[Path]:
    if checkpoint:
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = (output_dir / checkpoint).resolve()
        return candidate if candidate.exists() else None

    direct = output_dir / f"checkpoint_task_{resume_task:03d}.pt"
    if direct.exists():
        return direct

    meta_json = output_dir / f"checkpoint_task_{resume_task:03d}.json"
    if not meta_json.exists():
        return None
    try:
        payload = json.loads(meta_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    model_ckpt = payload.get("model_checkpoint")
    if not model_ckpt:
        return None
    candidate = output_dir / str(model_ckpt)
    return candidate if candidate.exists() else None


def eval_mock_continual(
    config_path: str,
    output_dir: Path,
    resume_task: int,
    checkpoint: Optional[str],
    open_vocab: bool,
) -> Dict[str, float]:
    ckpt_path = _resolve_mock_ckpt(output_dir=output_dir, resume_task=resume_task, checkpoint=checkpoint)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Mock checkpoint for task {resume_task} not found under {output_dir}. "
            "Run training first or pass --checkpoint to a valid .pt file."
        )

    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(state["num_classes"])
    text_dim = int(state["text_dim"])
    task_splits = list(state["task_splits"])
    palette = torch.tensor(state["palette"], dtype=torch.float32)
    text_embeddings = torch.tensor(state["text_embeddings"], dtype=torch.float32)

    model = _build_model(num_classes=num_classes, text_dim=text_dim, seed=int(state.get("seed", 0)))
    model.load_state_dict(state["model_state_dict"])

    generator = torch.Generator().manual_seed(int(state.get("seed", 0)) + 4096 + resume_task)

    safe_task = min(max(1, resume_task), len(task_splits))
    split = task_splits[safe_task - 1]
    seen = [int(x) for x in split["seen_classes"]]
    new = [int(x) for x in split["new_classes"]]
    old = [c for c in seen if c not in set(new)]
    bg = [int(x) for x in split["background_classes"]]

    miou_all = _eval_miou(
        model=model,
        text_embeddings=text_embeddings,
        palette=palette,
        class_ids=seen,
        num_classes=num_classes,
        batch_size=2,
        image_size=16,
        batches=8,
        generator=generator,
    )
    miou_old = _eval_miou(
        model=model,
        text_embeddings=text_embeddings,
        palette=palette,
        class_ids=old or seen,
        num_classes=num_classes,
        batch_size=2,
        image_size=16,
        batches=6,
        generator=generator,
    )
    miou_new = _eval_miou(
        model=model,
        text_embeddings=text_embeddings,
        palette=palette,
        class_ids=new,
        num_classes=num_classes,
        batch_size=2,
        image_size=16,
        batches=6,
        generator=generator,
    )
    bg_candidates = bg[: max(1, min(16, len(bg)))]
    bg_miou = _eval_miou(
        model=model,
        text_embeddings=text_embeddings,
        palette=palette,
        class_ids=bg_candidates,
        num_classes=num_classes,
        batch_size=2,
        image_size=16,
        batches=6,
        generator=generator,
    )

    metrics_records = 0
    metrics_file = output_dir / "metrics.jsonl"
    if metrics_file.exists():
        metrics_records = len([line for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()])

    payload: Dict[str, float] = {
        "mIoU_all": float(miou_all),
        "mIoU_old": float(miou_old),
        "mIoU_new": float(miou_new),
        "BG-mIoU": float(bg_miou),
        "resume_task": float(resume_task),
        "train_metric_records": float(metrics_records),
        "checkpoint": str(ckpt_path.name),
        "config": config_path,
        "engine": "mock",
    }

    if open_vocab:
        ov_class_count = min(20, max(3, len(seen)))
        ov_classes = seen[:ov_class_count]
        if len(ov_classes) < 3:
            ov_classes = list(range(min(3, num_classes)))
        model.eval()
        with torch.no_grad():
            images, local_targets = _sample_batch(
                class_ids=ov_classes,
                palette=palette,
                batch_size=2,
                image_size=16,
                generator=generator,
            )
            logits = model(images=images, text_embeddings=text_embeddings)["logits"]
            class_index = torch.tensor(ov_classes, dtype=torch.long)
            logits_ov = logits[:, class_index, :, :]

            remapped = torch.zeros_like(local_targets)
            for idx, class_id in enumerate(ov_classes):
                remapped = torch.where(local_targets == class_id, torch.tensor(idx, dtype=remapped.dtype), remapped)

            ov = OpenVocabEvaluator(dataset_aliases={"pc59": "pascal_context_59"})
            payload.update(ov.evaluate_dataset(dataset_key="pc59", logits=logits_ov, targets=remapped))

    (output_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
