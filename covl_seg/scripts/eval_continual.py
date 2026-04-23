import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from covl_seg.engine.detectron2_runner import detectron2_available, run_detectron2_eval
from covl_seg.engine.mock_continual_runner import eval_mock_continual


def _parse_bool_flag(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
    parser.add_argument(
        "--eval-sliding-window",
        type=_parse_bool_flag,
        default=True,
        help="Use sliding-window inference for Detectron2 eval (true/false)",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Optional cap for eval sample count",
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


def run_eval_once(
    config_path: str,
    output_dir: Path,
    resume_task: int,
    checkpoint: Optional[str],
    open_vocab: bool,
    engine: str = "auto",
    seg_net: Optional[str] = None,
    eval_sliding_window: bool = True,
    eval_max_samples: Optional[int] = None,
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
            eval_sliding_window=eval_sliding_window,
            eval_max_samples=eval_max_samples,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return eval_mock_continual(
        config_path=config_path,
        output_dir=output_dir,
        resume_task=resume_task,
        checkpoint=checkpoint,
        open_vocab=open_vocab,
    )


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
        "eval_sliding_window": args.eval_sliding_window,
        "eval_max_samples": args.eval_max_samples,
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
        eval_sliding_window=args.eval_sliding_window,
        eval_max_samples=args.eval_max_samples,
    )
    print(
        "Eval run complete: "
        f"mIoU_all={payload['mIoU_all']:.3f}, "
        f"resume_task={int(payload['resume_task'])}"
    )


if __name__ == "__main__":
    main()
