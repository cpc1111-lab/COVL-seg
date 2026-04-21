import argparse
import json
from pathlib import Path

from covl_seg.engine.open_continual_trainer import OpenContinualTrainer
from covl_seg.scripts.bootstrap_coco_train import (
    _resolve_d2_runtime_root,
    ensure_coco_stuff_ready_for_training,
    resolve_datasets_root,
)
from covl_seg.scripts.bootstrap_open_vocab_data import ensure_open_vocab_eval_data_ready


def _parse_bool_flag(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train continual open-learning model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine", choices=["auto", "mock", "d2"], default="auto")
    parser.add_argument(
        "--datasets-root",
        default=None,
        help="Dataset root; missing COCO-Stuff will be auto-downloaded here in d2 mode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-task", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)

    parser.add_argument("--col-method", choices=["covl", "none", "replay", "ewc"], default="covl")
    parser.add_argument("--task-spec", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--classes-per-task", type=int, default=None)
    parser.add_argument("--task-seed", type=int, default=0)

    parser.add_argument("--clip-finetune", choices=["none", "attention", "full"], default="attention")
    parser.add_argument(
        "--seg-net",
        choices=["vitb", "vitl", "r50", "r101", "swin_t", "swin_b"],
        default=None,
        help="Optional segmentation network preset for Detectron2 backend",
    )
    parser.add_argument("--open-vocab", action="store_true")
    parser.add_argument(
        "--skip-per-task-eval",
        action="store_true",
        help="Skip per-task Detectron2 evaluation during training",
    )
    parser.add_argument(
        "--eval-sliding-window",
        type=_parse_bool_flag,
        default=True,
        help="Use sliding-window inference for per-task eval (true/false)",
    )
    parser.add_argument(
        "--eval-max-samples-per-task",
        type=int,
        default=None,
        help="Optional cap for per-task eval sample count",
    )

    parser.add_argument("--n-pre", type=int, default=200)
    parser.add_argument("--n-main", type=int, default=4000)
    parser.add_argument("--eps-f", type=float, default=0.05)
    parser.add_argument("--t-mem", default="all")
    parser.add_argument("--mix-ratio", nargs=2, type=int, default=[3, 1])
    parser.add_argument("--m-max-total", type=int, default=4000)
    parser.add_argument("--m-max-per-class", type=int, default=200)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--ewc-topk", type=int, default=8)
    parser.add_argument("--ewc-iters", type=int, default=25)

    parser.add_argument("--enable-ciba", dest="enable_ciba", action="store_true")
    parser.add_argument("--disable-ciba", dest="enable_ciba", action="store_false")
    parser.set_defaults(enable_ciba=True)
    parser.add_argument("--enable-ctr", dest="enable_ctr", action="store_true")
    parser.add_argument("--disable-ctr", dest="enable_ctr", action="store_false")
    parser.set_defaults(enable_ctr=True)
    parser.add_argument("--enable-spectral-ogp", dest="enable_spectral_ogp", action="store_true")
    parser.add_argument("--disable-spectral-ogp", dest="enable_spectral_ogp", action="store_false")
    parser.set_defaults(enable_spectral_ogp=True)
    parser.add_argument("--enable-sacr", dest="enable_sacr", action="store_true")
    parser.add_argument("--disable-sacr", dest="enable_sacr", action="store_false")
    parser.set_defaults(enable_sacr=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "d2":
        repo_root = Path(__file__).resolve().parents[2]
        resolved_datasets_root = resolve_datasets_root(args.datasets_root, repo_root=repo_root)
        runtime_root = _resolve_d2_runtime_root(repo_root=repo_root)
        ensure_coco_stuff_ready_for_training(
            datasets_root=resolved_datasets_root,
            runtime_root=runtime_root,
        )
        if args.open_vocab:
            ensure_open_vocab_eval_data_ready(
                datasets_root=resolved_datasets_root,
                runtime_root=runtime_root,
            )

    run_cfg = vars(args).copy()
    (out_dir / "open_run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    trainer = OpenContinualTrainer.from_args(args)
    result = trainer.run(max_tasks=args.max_tasks)
    print(
        "Open continual run complete: "
        f"tasks_executed={int(result['tasks_executed'])}, "
        f"last_task={int(result['last_task'])}"
    )


if __name__ == "__main__":
    main()
