import argparse
import json
import shlex
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan balanced-controller ablation runs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-root", required=True, help="Root directory for variant outputs")
    parser.add_argument("--engine", choices=["auto", "mock", "d2"], default="mock")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def variant_definitions() -> dict[str, dict[str, float | str]]:
    return {
        "A0": {"balanced_profile": "off"},
        "A1": {
            "balanced_profile": "balanced",
            "target_delta_new": 0.30,
        },
        "A2": {
            "balanced_profile": "balanced",
            "target_delta_new": 0.35,
            "epsilon_old": 0.22,
        },
        "A3": {
            "balanced_profile": "balanced",
            "target_delta_new": 0.35,
            "epsilon_old": 0.18,
            "epsilon_ov": 0.15,
        },
    }


def build_variant_command(config: str, output_dir: str, engine: str, seed: int, settings: dict[str, float | str]) -> str:
    parts = [
        "python",
        "-m",
        "covl_seg.scripts.train_open_continual",
        "--config",
        config,
        "--output-dir",
        output_dir,
        "--engine",
        engine,
        "--seed",
        str(seed),
    ]
    for key, value in settings.items():
        parts.extend([f"--{key.replace('_', '-')}", str(value)])
    return shlex.join(parts)


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    plan = []
    for variant, settings in variant_definitions().items():
        output_dir = output_root / variant
        command = build_variant_command(
            config=args.config,
            output_dir=str(output_dir),
            engine=args.engine,
            seed=args.seed,
            settings=settings,
        )
        plan.append(
            {
                "variant": variant,
                "output_dir": str(output_dir),
                "settings": settings,
                "command": command,
            }
        )

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
