import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_jsonl(file_path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def generate_analysis_artifacts(metrics_jsonl: Path, output_dir: Path) -> Dict[str, int]:
    records = _read_jsonl(metrics_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "analysis_summary.json"
    summary_csv = output_dir / "analysis_summary.csv"
    curves_json = output_dir / "analysis_curves.json"
    task_summary_json = output_dir / "analysis_task_summary.json"
    task_summary_csv = output_dir / "analysis_task_summary.csv"

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

    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate COVL theory-analysis artifacts")
    parser.add_argument("--metrics-jsonl", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON summaries")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = generate_analysis_artifacts(
        metrics_jsonl=Path(args.metrics_jsonl),
        output_dir=Path(args.output_dir),
    )
    print(f"Generated analysis summaries for {payload['num_records']} records")


if __name__ == "__main__":
    main()
