import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from covl_seg.engine.detectron2_runner import detectron2_available, run_detectron2_train
from covl_seg.engine.mock_continual_runner import train_mock_continual
from covl_seg.scripts.bootstrap_coco_train import (
    _resolve_d2_runtime_root,
    ensure_coco_stuff_ready_for_training,
    resolve_datasets_root,
)


def _emit_d2_progress(line: str) -> None:
    print(line, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train COVL-Seg continual model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resume-task", type=int, default=0, help="Resume from task index")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on number of tasks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--engine",
        choices=["auto", "mock", "d2"],
        default="auto",
        help="Training engine backend",
    )
    parser.add_argument("--smoke", action="store_true", help="Run short smoke-mode schedule")
    parser.add_argument(
        "--seg-net",
        choices=["vitb", "vitl", "r50", "r101", "swin_t", "swin_b"],
        default=None,
        help="Optional segmentation network preset for Detectron2 backend",
    )
    parser.add_argument(
        "--datasets-root",
        default=None,
        help="Dataset root; missing COCO-Stuff will be auto-downloaded here in d2 mode",
    )
    return parser


def resolve_engine(requested: str, detectron2_ready: bool) -> str:
    if requested == "mock":
        return "mock"
    if requested == "d2":
        if not detectron2_ready:
            raise RuntimeError("Detectron2 backend was requested, but Detectron2 is unavailable")
        return "d2"
    if requested == "auto":
        return "d2" if detectron2_ready else "mock"
    raise ValueError(f"Unsupported engine: {requested}")


def run_smoke_once(output_dir: Path, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_tasks = 1
    num_iters = 5
    losses = []
    has_nan = False

    weight = torch.randn(16, 16)
    for _ in range(num_iters):
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)
        pred = x @ weight
        loss = ((pred - y) ** 2).mean()
        losses.append(float(loss.item()))
        if not torch.isfinite(loss):
            has_nan = True

    payload = {
        "num_tasks": num_tasks,
        "num_iters": num_iters,
        "has_nan": has_nan,
        "losses": losses,
    }
    (output_dir / "smoke_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_train_once(
    config_path: str,
    output_dir: Path,
    seed: int,
    resume_task: int,
    max_tasks: Optional[int],
    engine: str = "auto",
    seg_net: Optional[str] = None,
    datasets_root: Optional[str] = None,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_engine(requested=engine, detectron2_ready=detectron2_available())

    if backend == "d2":
        repo_root = Path(__file__).resolve().parents[2]
        resolved_datasets_root = resolve_datasets_root(datasets_root, repo_root=repo_root)
        runtime_root = _resolve_d2_runtime_root(repo_root=repo_root)
        ensure_coco_stuff_ready_for_training(
            datasets_root=resolved_datasets_root,
            runtime_root=runtime_root,
        )
        return run_detectron2_train(
            config_path=config_path,
            output_dir=output_dir,
            seed=seed,
            resume_task=resume_task,
            max_tasks=max_tasks,
            seg_network=seg_net,
            progress_callback=_emit_d2_progress,
        )

    return train_mock_continual(
        config_path=config_path,
        output_dir=output_dir,
        seed=seed,
        resume_task=resume_task,
        max_tasks=max_tasks,
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "mode": "train",
        "config": args.config,
        "output_dir": str(out_dir),
        "resume_task": args.resume_task,
        "max_tasks": args.max_tasks,
        "seed": args.seed,
        "engine": args.engine,
        "seg_net": args.seg_net,
        "datasets_root": args.datasets_root,
        "smoke": args.smoke,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    if args.smoke:
        smoke = run_smoke_once(output_dir=out_dir, seed=args.seed)
        print(f"Smoke run complete: iters={smoke['num_iters']}, has_nan={smoke['has_nan']}")
        return

    payload = run_train_once(
        config_path=args.config,
        output_dir=out_dir,
        seed=args.seed,
        resume_task=args.resume_task,
        max_tasks=args.max_tasks,
        engine=args.engine,
        seg_net=args.seg_net,
        datasets_root=args.datasets_root,
    )
    print(
        "Train run complete: "
        f"tasks={payload['num_tasks']}, "
        f"phase_records={payload['num_phase_records']}, "
        f"last_task={payload['last_task']}"
    )


if __name__ == "__main__":
    main()
