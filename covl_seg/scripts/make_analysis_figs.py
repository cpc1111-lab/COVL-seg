import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(file_path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _read_task_eval_summaries(run_dir: Path) -> Dict[int, Dict[str, object]]:
    summaries: Dict[int, Dict[str, object]] = {}
    for path in sorted(run_dir.glob("task_*/eval_summary.json")):
        task_dir = path.parent
        task_name = task_dir.name
        if not task_name.startswith("task_"):
            continue
        try:
            task_id = int(task_name.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            summaries[task_id] = payload
    return summaries


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sanitize_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip().lower())
    return cleaned[:80] or "class"


def _build_class_trends(
    eval_by_task: Dict[int, Dict[str, object]]
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
    class_trends: Dict[str, List[Dict[str, object]]] = {}
    group_rows: List[Dict[str, object]] = []

    for task_id in sorted(eval_by_task.keys()):
        payload = eval_by_task[task_id]
        class_iou_all = payload.get("class_iou_all")
        class_iou_old = payload.get("class_iou_old")
        class_iou_new = payload.get("class_iou_new")
        class_iou_bg = payload.get("class_iou_bg")
        if not isinstance(class_iou_all, dict):
            continue

        old_names = set(class_iou_old.keys()) if isinstance(class_iou_old, dict) else set()
        new_names = set(class_iou_new.keys()) if isinstance(class_iou_new, dict) else set()
        bg_names = set(class_iou_bg.keys()) if isinstance(class_iou_bg, dict) else set()

        old_vals: List[float] = []
        new_vals: List[float] = []
        bg_vals: List[float] = []
        all_vals: List[float] = []

        for class_name, score in class_iou_all.items():
            val = _safe_float(score)
            if val is None:
                continue
            if class_name in new_names:
                group = "new"
                new_vals.append(val)
            elif class_name in old_names:
                group = "old"
                old_vals.append(val)
            elif class_name in bg_names:
                group = "bg"
                bg_vals.append(val)
            else:
                group = "other"
            all_vals.append(val)

            class_trends.setdefault(class_name, []).append(
                {
                    "task": task_id,
                    "iou": val,
                    "group": group,
                }
            )

        group_rows.append(
            {
                "task": task_id,
                "mean_all": _mean(all_vals),
                "mean_old": _mean(old_vals),
                "mean_new": _mean(new_vals),
                "mean_bg": _mean(bg_vals),
                "count_all": len(all_vals),
                "count_old": len(old_vals),
                "count_new": len(new_vals),
                "count_bg": len(bg_vals),
            }
        )

    return class_trends, group_rows


def _plot_global_trends(group_rows: List[Dict[str, object]], out_png: Path) -> None:
    if not group_rows:
        return
    tasks = [int(row["task"]) for row in group_rows]
    plt.figure(figsize=(10, 5))
    for key, label in [
        ("mean_all", "all"),
        ("mean_old", "old"),
        ("mean_new", "new"),
        ("mean_bg", "bg"),
    ]:
        ys = [row.get(key) for row in group_rows]
        if all(y is None for y in ys):
            continue
        plt.plot(tasks, ys, marker="o", label=label)
    plt.xlabel("Task")
    plt.ylabel("mIoU")
    plt.title("Group mIoU Trends")
    plt.grid(alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_class_trends(class_trends: Dict[str, List[Dict[str, object]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name, points in class_trends.items():
        if not points:
            continue
        points_sorted = sorted(points, key=lambda x: int(x["task"]))
        xs = [int(p["task"]) for p in points_sorted]
        ys = [float(p["iou"]) for p in points_sorted]
        colors = ["tab:blue" if p["group"] == "old" else "tab:orange" if p["group"] == "new" else "tab:green" if p["group"] == "bg" else "tab:gray" for p in points_sorted]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, linewidth=1.5, color="0.5")
        plt.scatter(xs, ys, c=colors, s=18)
        plt.ylim(0, 100)
        plt.xlabel("Task")
        plt.ylabel("IoU")
        plt.title(f"Class IoU Trend: {class_name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{_sanitize_filename(class_name)}.png", dpi=130)
        plt.close()


def generate_analysis_artifacts(metrics_jsonl: Path, output_dir: Path, run_dir: Optional[Path] = None) -> Dict[str, int]:
    records = _read_jsonl(metrics_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_run_dir = run_dir if run_dir is not None else metrics_jsonl.parent

    summary_json = output_dir / "analysis_summary.json"
    summary_csv = output_dir / "analysis_summary.csv"
    curves_json = output_dir / "analysis_curves.json"
    task_summary_json = output_dir / "analysis_task_summary.json"
    task_summary_csv = output_dir / "analysis_task_summary.csv"
    class_trends_json = output_dir / "analysis_class_iou_trends.json"
    group_trends_json = output_dir / "analysis_group_trends.json"
    group_trends_csv = output_dir / "analysis_group_trends.csv"

    keys = sorted({key for rec in records for key in rec.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    summary_payload = {
        "num_records": len(records),
        "columns": keys,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    proxy_keys = [
        "beta_1_star",
        "beta_2_star",
        "gamma_clip",
        "omega_tau_t",
        "alpha_star",
        "tau_pred",
        "fisher_energy",
        "ctr_loss",
    ]
    curves = {
        key: [
            {
                "task": rec.get("task"),
                "phase": rec.get("phase"),
                "value": rec[key],
            }
            for rec in records
            if key in rec
        ]
        for key in proxy_keys
    }
    curves_json.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    task_summary: Dict[str, Dict[str, float]] = {}
    for rec in records:
        task_val = rec.get("task")
        if task_val is None:
            continue
        task_key = str(int(task_val)) if isinstance(task_val, (int, float)) else str(task_val)
        current = task_summary.setdefault(task_key, {})
        for key in proxy_keys:
            if key in rec:
                current[key] = rec[key]
        if "mIoU_all" in rec:
            current["mIoU_all"] = rec["mIoU_all"]
        if "mIoU_old" in rec:
            current["mIoU_old"] = rec["mIoU_old"]
        if "mIoU_new" in rec:
            current["mIoU_new"] = rec["mIoU_new"]

    task_summary_json.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")

    task_rows = []
    for task_key in sorted(task_summary.keys(), key=lambda x: int(x) if x.isdigit() else x):
        row = {"task": task_key}
        row.update(task_summary[task_key])
        task_rows.append(row)
    task_columns = sorted({key for row in task_rows for key in row.keys()}) if task_rows else ["task"]
    with task_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_columns)
        writer.writeheader()
        for row in task_rows:
            writer.writerow(row)

    eval_by_task = _read_task_eval_summaries(actual_run_dir)
    class_trends, group_rows = _build_class_trends(eval_by_task)
    class_trends_json.write_text(json.dumps(class_trends, indent=2), encoding="utf-8")
    group_trends_json.write_text(json.dumps(group_rows, indent=2), encoding="utf-8")

    with group_trends_csv.open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "task",
            "mean_all",
            "mean_old",
            "mean_new",
            "mean_bg",
            "count_all",
            "count_old",
            "count_new",
            "count_bg",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in group_rows:
            writer.writerow(row)

    _plot_global_trends(group_rows, output_dir / "fig_group_miou_trends.png")
    _plot_class_trends(class_trends, output_dir / "fig_class_trends")

    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate COVL theory-analysis artifacts")
    parser.add_argument("--metrics-jsonl", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON summaries")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional training run root containing task_*/eval_summary.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = generate_analysis_artifacts(
        metrics_jsonl=Path(args.metrics_jsonl),
        output_dir=Path(args.output_dir),
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(f"Generated analysis summaries for {payload['num_records']} records")


if __name__ == "__main__":
    main()
