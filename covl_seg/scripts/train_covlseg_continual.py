import argparse
import json
import logging
from pathlib import Path

import yaml

from covl_seg.engine.open_continual_trainer import OpenContinualTrainer
from covl_seg.data.download import ensure_dataset

_log = logging.getLogger(__name__)

_YAML_TO_ARG_MAP = {
    ("model", "clip_model_name"): "clip_model_name",
    ("model", "dino_model_name"): "dino_model_name",
    ("model", "clip_finetune"): "clip_finetune",
    ("model", "num_classes"): None,
    ("model", "out_dim"): None,
    ("dataset", "name"): None,
    ("dataset", "root"): "dataset_root",
    ("dataset", "split"): None,
    ("training", "batch_size"): "batch_size",
    ("training", "num_workers"): None,
    ("training", "learning_rate"): "learning_rate",
    ("training", "text_learning_rate"): "text_learning_rate",
    ("training", "weight_decay"): None,
    ("training", "n_main"): "n_main",
    ("training", "use_real_training"): "use_real_training",
    ("training", "image_size"): "image_size",
    ("training", "lr_scheduler"): "lr_scheduler",
    ("continual", "method"): "col_method",
    ("continual", "num_tasks"): "num_tasks",
    ("continual", "classes_per_task"): "classes_per_task",
    ("continual", "task_seed"): "task_seed",
    ("continual", "ewc_lambda"): "ewc_lambda",
    ("continual", "ewc_topk"): "ewc_topk",
    ("continual", "ewc_iters"): "ewc_iters",
    ("continual", "balanced_profile"): "balanced_profile",
    ("continual", "eps_f"): "eps_f",
    ("continual", "t_mem"): "t_mem",
    ("continual", "mix_ratio"): "mix_ratio",
    ("continual", "m_max_total"): "m_max_total",
    ("continual", "m_max_per_class"): "m_max_per_class",
    ("continual", "enable_ciba"): "enable_ciba",
    ("continual", "enable_ctr"): "enable_ctr",
    ("continual", "enable_spectral_ogp"): "enable_spectral_ogp",
    ("continual", "enable_sacr"): "enable_sacr",
}

_ARG_DEFAULTS = {
    "dataset_root": "datasets/ADE20K",
    "n_main": 10000,
    "clip_model_name": "ViT-B-16",
    "dino_model_name": "dinov2_vitb14",
    "clip_finetune": "attention",
    "batch_size": 4,
    "learning_rate": 1e-4,
    "text_learning_rate": 1e-5,
    "use_real_training": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train COVL-Seg continual open-vocabulary segmentation model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-task", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)

    parser.add_argument("--col-method", choices=["covl", "none", "replay", "ewc"], default="covl")
    parser.add_argument("--task-spec", default=None)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--classes-per-task", type=int, default=None)
    parser.add_argument("--task-seed", type=int, default=0)

    parser.add_argument("--clip-finetune", choices=["none", "attention", "v_only", "full"], default="attention")
    parser.add_argument("--clip-model-name", default="ViT-B-16")
    parser.add_argument("--dino-model-name", default="dinov2_vitb14")

    parser.add_argument("--n-pre", type=int, default=200)
    parser.add_argument("--n-main", type=int, default=10000)
    parser.add_argument("--eps-f", type=float, default=0.05)
    parser.add_argument("--t-mem", default="all")
    parser.add_argument("--mix-ratio", nargs=2, type=int, default=[3, 1])
    parser.add_argument("--m-max-total", type=int, default=4000)
    parser.add_argument("--m-max-per-class", type=int, default=200)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--ewc-topk", type=int, default=8)
    parser.add_argument("--ewc-iters", type=int, default=25)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--eval-max-samples", type=int, default=500)

    parser.add_argument("--balanced-profile", choices=["off", "balanced"], default="off")
    parser.add_argument("--target-delta-new", type=float, default=0.30)
    parser.add_argument("--epsilon-old", type=float, default=0.20)
    parser.add_argument("--epsilon-all", type=float, default=0.15)
    parser.add_argument("--epsilon-ov", type=float, default=0.20)

    parser.add_argument("--enable-ciba", dest="enable_ciba", action="store_true")
    parser.add_argument("--disable-ciba", dest="enable_ciba", action="store_false")
    parser.set_defaults(enable_ciba=True)
    parser.add_argument("--enable-ctr", dest="enable_ctr", action="store_true")
    parser.add_argument("--disable-ctr", dest="enable_ctr", action="store_false")
    parser.set_defaults(enable_ctr=False)
    parser.add_argument("--enable-spectral-ogp", dest="enable_spectral_ogp", action="store_true")
    parser.add_argument("--disable-spectral-ogp", dest="enable_spectral_ogp", action="store_false")
    parser.set_defaults(enable_spectral_ogp=True)
    parser.add_argument("--enable-sacr", dest="enable_sacr", action="store_true")
    parser.add_argument("--disable-sacr", dest="enable_sacr", action="store_false")
    parser.set_defaults(enable_sacr=True)

    parser.add_argument("--use-real-training", dest="use_real_training", action="store_true")
    parser.set_defaults(use_real_training=False)
    parser.add_argument("--engine", default="auto", choices=["auto", "d2", "mock"])

    parser.add_argument("--seg-net", default=None)
    parser.add_argument("--open-vocab", dest="open_vocab", action="store_true")
    parser.set_defaults(open_vocab=False)
    parser.add_argument("--skip-per-task-eval", dest="skip_per_task_eval", action="store_true")
    parser.set_defaults(skip_per_task_eval=False)
    parser.add_argument("--eval-sliding-window", dest="eval_sliding_window", action="store_true")
    parser.set_defaults(eval_sliding_window=True)
    parser.add_argument("--eval-max-samples-per-task", type=int, default=None)
    parser.add_argument("--train-iters-mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--min-iters-per-visible-class", type=int, default=350)
    parser.add_argument("--max-iters-multiplier", type=float, default=2.0)
    parser.add_argument("--lambda-old-kd", type=float, default=1.0)
    parser.add_argument("--lambda-old-clip", type=float, default=0.1)
    parser.add_argument("--lambda-unseen-clip", type=float, default=0.2)
    return parser


def merge_config_into_args(cfg: dict, args: argparse.Namespace) -> None:
    for (section, key), arg_key in _YAML_TO_ARG_MAP.items():
        if arg_key is None:
            continue
        section_data = cfg.get(section, {})
        if not isinstance(section_data, dict):
            continue
        value = section_data.get(key)
        if value is None:
            continue
        current = getattr(args, arg_key, None)
        default = _ARG_DEFAULTS.get(arg_key)
        if current == default or current is None:
            setattr(args, arg_key, value)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    merge_config_into_args(cfg, args)

    if args.dataset_root is None:
        dataset_section = cfg.get("dataset", {})
        if isinstance(dataset_section, dict) and "root" in dataset_section:
            args.dataset_root = dataset_section["root"]
        else:
            args.dataset_root = "datasets/ADE20K"

    run_cfg = vars(args).copy()
    run_cfg.update(cfg)
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    _log.info("dataset_root=%s  n_main=%s  num_tasks=%s  classes_per_task=%s",
              args.dataset_root, args.n_main, args.num_tasks, args.classes_per_task)

    args.dataset_root = ensure_dataset(args.config, args.dataset_root)

    trainer = OpenContinualTrainer.from_args(args)
    result = trainer.run(max_tasks=args.max_tasks)
    print(
        "COVL-Seg continual run complete: "
        f"tasks_executed={int(result['tasks_executed'])}, "
        f"last_task={int(result['last_task'])}"
    )


if __name__ == "__main__":
    main()