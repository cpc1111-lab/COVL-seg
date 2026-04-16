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
