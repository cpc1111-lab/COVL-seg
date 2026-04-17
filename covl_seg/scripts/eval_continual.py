import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from covl_seg.engine.evaluator import compute_basic_miou, summarize_metrics
from covl_seg.engine.detectron2_runner import detectron2_available, run_detectron2_eval
from covl_seg.engine.open_vocab_eval import OpenVocabEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate COVL-Seg continual model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resume-task", type=int, default=0, help="Evaluate checkpoint from task index")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override")
    parser.add_argument(
        "--engine",
        choices=["auto", "mock", "d2"],
        default="auto",
        help="Evaluation engine backend",
    )
    parser.add_argument("--open-vocab", action="store_true", help="Run PC59/PC459/VOC20 evaluations")
    parser.add_argument(
        "--seg-net",
        choices=["vitb", "vitl", "r50", "r101", "swin_t", "swin_b"],
        default=None,
        help="Optional segmentation network preset for Detectron2 backend",
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


def _read_metrics_jsonl(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines)


def run_eval_once(
    config_path: str,
    output_dir: Path,
    resume_task: int,
    checkpoint: Optional[str],
    open_vocab: bool,
    engine: str = "auto",
    seg_net: Optional[str] = None,
) -> Dict[str, float]:
    backend = resolve_engine(requested=engine, detectron2_ready=detectron2_available())
    if backend == "d2":
        return run_detectron2_eval(
            config_path=config_path,
            output_dir=output_dir,
            resume_task=resume_task,
            checkpoint=checkpoint,
            open_vocab=open_vocab,
            seg_network=seg_net,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_records = _read_metrics_jsonl(output_dir / "metrics.jsonl")

    torch.manual_seed(100 + resume_task)
    logits = torch.randn(1, 4, 16, 16)
    targets = torch.randint(0, 4, (1, 16, 16))
    pred = logits.argmax(dim=1)

    miou_all = compute_basic_miou(pred=pred, target=targets, num_classes=4)
    base = summarize_metrics(
        miou_all=miou_all,
        miou_old=max(miou_all - 1.0, 0.0),
        miou_new=min(miou_all + 1.0, 100.0),
        bg_miou=max(miou_all - 2.0, 0.0),
    )
    base["resume_task"] = float(resume_task)
    base["train_metric_records"] = float(metrics_records)
    if checkpoint is not None:
        base["checkpoint"] = checkpoint
    else:
        base["checkpoint"] = f"checkpoint_task_{resume_task:03d}.json"
    base["config"] = config_path
    base["engine"] = "mock"

    if open_vocab:
        ov = OpenVocabEvaluator(dataset_aliases={"pc59": "pascal_context_59"})
        ov_metrics = ov.evaluate_dataset(dataset_key="pc59", logits=logits[:, :3], targets=targets.clamp_max(2))
        base.update(ov_metrics)

    (output_dir / "eval_summary.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "mode": "eval",
        "config": args.config,
        "output_dir": str(out_dir),
        "resume_task": args.resume_task,
        "checkpoint": args.checkpoint,
        "engine": args.engine,
        "seg_net": args.seg_net,
        "open_vocab": args.open_vocab,
    }
    (out_dir / "eval_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    payload = run_eval_once(
        config_path=args.config,
        output_dir=out_dir,
        resume_task=args.resume_task,
        checkpoint=args.checkpoint,
        open_vocab=args.open_vocab,
        engine=args.engine,
        seg_net=args.seg_net,
    )
    print(
        "Eval run complete: "
        f"mIoU_all={payload['mIoU_all']:.3f}, "
        f"resume_task={int(payload['resume_task'])}"
    )


if __name__ == "__main__":
    main()
