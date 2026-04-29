import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_log = logging.getLogger(__name__)


def _read_jsonl(file_path: Path) -> List[Dict[str, float]]:
    if not file_path.exists():
        return []
    records: List[Dict[str, float]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
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


def _has_valid_data(records: List[Dict[str, float]], *keys: str) -> bool:
    if not records:
        return False
    for key in keys:
        values = [_safe_float(rec.get(key)) for rec in records]
        if all(v is None for v in values):
            return False
    return True


def _get_series(records: List[Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for rec in records:
        task_val = rec.get("task")
        if task_val is None:
            continue
        val = _safe_float(rec.get(key))
        if val is None:
            continue
        try:
            xs.append(int(task_val))
        except (ValueError, TypeError):
            continue
        ys.append(val)
    return xs, ys


def _fig_perf_miou_curves_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    if not _has_valid_data(records, "mIoU_all"):
        _log.warning("[report] skipped fig_perf_miou_curves: no data")
        return None

    xs_all, ys_all = _get_series(records, "mIoU_all")
    xs_old, ys_old = _get_series(records, "mIoU_old")
    xs_new, ys_new = _get_series(records, "mIoU_new")
    xs_bg, ys_bg = _get_series(records, "BG-mIoU")

    if not xs_all:
        _log.warning("[report] skipped fig_perf_miou_curves: no valid data")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    if xs_all and ys_all:
        ax.plot(xs_all, ys_all, marker="o", label="mIoU_all", color="tab:blue")
    if xs_old and ys_old:
        ax.plot(xs_old, ys_old, marker="s", label="mIoU_old", color="tab:green")
    if xs_new and ys_new:
        ax.plot(xs_new, ys_new, marker="^", label="mIoU_new", color="tab:orange")
    if xs_bg and ys_bg:
        ax.plot(xs_bg, ys_bg, marker="d", label="BG-mIoU", color="tab:red")
    ax.set_xlabel("Task")
    ax.set_ylabel("mIoU")
    ax.set_title("Segmentation Performance Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out_path = output_dir / "fig_perf_miou_curves.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_perf_forgetting_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    eval_records = [r for r in records if r.get("phase") == "eval"]
    if len(eval_records) < 2:
        _log.warning("[report] skipped fig_perf_forgetting: fewer than 2 eval records")
        return None

    backward_transfer = []
    tasks = []
    cumulative_bwt = []
    cum_sum = 0.0
    prev_miou_old: Optional[float] = None

    for i, rec in enumerate(eval_records):
        task_val = rec.get("task")
        if task_val is None:
            continue
        try:
            task = int(task_val)
        except (ValueError, TypeError):
            continue

        current_miou_new = _safe_float(rec.get("mIoU_new"))
        current_miou_old = _safe_float(rec.get("mIoU_old"))

        if current_miou_new is None or current_miou_old is None:
            balanced_records = [r for r in records if r.get("phase") == "balanced_ctrl" and r.get("task") == task_val]
            if balanced_records:
                br = balanced_records[0]
                delta_new = _safe_float(br.get("delta_new"))
                delta_old = _safe_float(br.get("delta_old"))
                if delta_old is not None:
                    cum_sum += delta_old
                    cumulative_bwt.append(cum_sum)
                    backward_transfer.append(-delta_old)
                    tasks.append(task)
            continue

        if task <= 1:
            prev_miou_old = current_miou_old
            continue

        if prev_miou_old is not None:
            bwt = current_miou_old - prev_miou_old
            backward_transfer.append(bwt)
            cum_sum += bwt
            cumulative_bwt.append(cum_sum)
            tasks.append(task)
        prev_miou_old = current_miou_old

    if not tasks:
        _log.warning("[report] skipped fig_perf_forgetting: no valid data")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(tasks, backward_transfer, color="tab:red", alpha=0.7, label="BWT (mIoU_old(t) - mIoU_old(t-1))")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Backward Transfer", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(tasks, cumulative_bwt, marker="o", color="tab:blue", label="Cumulative BWT")
    ax2.set_ylabel("Cumulative BWT", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Forgetting (Backward Transfer)")
    fig.tight_layout()

    out_path = output_dir / "fig_perf_forgetting.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_perf_stability_plasticity_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    eval_records = [r for r in records if r.get("phase") == "eval"]
    if len(eval_records) < 2:
        _log.warning("[report] skipped fig_perf_stability_plasticity: fewer than 2 eval records")
        return None

    xs_old, ys_old = [], []
    xs_new, ys_new = [], []

    for rec in eval_records:
        task_val = rec.get("task")
        if task_val is None:
            continue
        try:
            task = int(task_val)
        except (ValueError, TypeError):
            continue

        miou_old = _safe_float(rec.get("mIoU_old"))
        miou_new = _safe_float(rec.get("mIoU_new"))

        if miou_old is not None:
            xs_old.append(task)
            ys_old.append(miou_old)
        if miou_new is not None:
            xs_new.append(task)
            ys_new.append(miou_new)

    if not xs_old and not xs_new:
        _log.warning("[report] skipped fig_perf_stability_plasticity: no valid data")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs_old, ys_old, c="tab:blue", marker="s", s=60, label="mIoU_old (Stability)", alpha=0.7)
    ax.scatter(xs_new, ys_new, c="tab:orange", marker="^", s=60, label="mIoU_new (Plasticity)", alpha=0.7)

    ax.set_xlabel("Task")
    ax.set_ylabel("mIoU")
    ax.set_title("Stability (mIoU_old) vs Plasticity (mIoU_new)")
    ax.grid(alpha=0.3)
    ax.legend()
    all_xs = xs_old + xs_new
    if all_xs:
        x_min, x_max = min(all_xs), max(all_xs)
        x_pad = max(0.5, (x_max - x_min) * 0.1)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out_path = output_dir / "fig_perf_stability_plasticity.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_theory_alpha_tau_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    phase3_records = [r for r in records if r.get("phase") == "phase3"]
    if not _has_valid_data(phase3_records, "alpha_star"):
        _log.warning("[report] skipped fig_theory_alpha_tau: no data")
        return None

    xs_alpha, ys_alpha = _get_series(phase3_records, "alpha_star")
    xs_tau, ys_tau = _get_series(phase3_records, "tau_pred")

    if not xs_alpha:
        _log.warning("[report] skipped fig_theory_alpha_tau: no valid data")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = None
    if xs_alpha and ys_alpha:
        ax1.plot(xs_alpha, ys_alpha, marker="o", color="tab:blue", label="alpha*(t)")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("alpha*(t)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if xs_tau and ys_tau:
        ax2 = ax1.twinx()
        ax2.plot(xs_tau, ys_tau, marker="s", color="tab:red", label="tau_pred")
        ax2.set_ylabel("tau_pred", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(alpha=0.3)
    ax1.set_title("Theory Quantities: alpha*(t) and tau_pred")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()

    out_path = output_dir / "fig_theory_alpha_tau.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_theory_iexc_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    phase1_records = [r for r in records if r.get("phase") == "phase1"]
    if not _has_valid_data(phase1_records, "I_exc_C"):
        _log.warning("[report] skipped fig_theory_iexc: no data")
        return None

    xs_c, ys_c = _get_series(phase1_records, "I_exc_C")
    xs_s, ys_s = _get_series(phase1_records, "I_exc_S")

    if not xs_c:
        _log.warning("[report] skipped fig_theory_iexc: no valid data")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    if xs_c and ys_c:
        ax.plot(xs_c, ys_c, marker="o", label="I_exc^C", color="tab:blue")
    if xs_s and ys_s:
        ax.plot(xs_s, ys_s, marker="s", label="I_exc^S", color="tab:orange")

    if xs_c and ys_c and xs_s and ys_s:
        if len(xs_c) == len(xs_s):
            min_len = min(len(xs_c), len(xs_s))
            gap_xs = [xs_c[i] for i in range(min_len)]
            gap_ys = [max(0, ys_s[i] - ys_c[i]) for i in range(min_len)]
            if any(g > 0 for g in gap_ys):
                ax.fill_between(gap_xs, [0] * len(gap_xs), gap_ys, color="tab:green", alpha=0.2, label="Delta_S^C")

    ax.set_xlabel("Task")
    ax.set_ylabel("Mutual Information")
    ax.set_title("Theory Quantities: I_exc")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "fig_theory_iexc.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_theory_spectral_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    phase3_records = [r for r in records if r.get("phase") == "phase3"]
    if not _has_valid_data(phase3_records, "fisher_energy"):
        _log.warning("[report] skipped fig_theory_spectral: no data")
        return None

    xs_fisher, ys_fisher = _get_series(phase3_records, "fisher_energy")
    xs_omega, ys_omega = _get_series(phase3_records, "omega_tau_t")

    if not xs_fisher:
        _log.warning("[report] skipped fig_theory_spectral: no valid data")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = None
    ax1.bar(xs_fisher, ys_fisher, color="tab:blue", alpha=0.7, label="Fisher Energy")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Fisher Energy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if xs_omega and ys_omega:
        ax2 = ax1.twinx()
        ax2.plot(xs_omega, ys_omega, marker="o", color="tab:red", label="omega_tau_t")
        ax2.set_ylabel("omega_tau_t", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(alpha=0.3, axis="y")
    ax1.set_title("Theory Quantities: Fisher Energy and omega")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()

    out_path = output_dir / "fig_theory_spectral.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_bg_ctr_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    phase2_records = [r for r in records if r.get("phase") == "phase2"]
    if not _has_valid_data(phase2_records, "ctr_loss"):
        _log.warning("[report] skipped fig_bg_ctr: no data")
        return None

    xs_loss, ys_loss = _get_series(phase2_records, "ctr_loss")
    xs_gamma, ys_gamma = _get_series(phase2_records, "gamma_clip")

    if not xs_loss:
        _log.warning("[report] skipped fig_bg_ctr: no valid data")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = None
    if xs_loss and ys_loss:
        ys_loss_abs = [abs(y) for y in ys_loss]
        ax1.plot(xs_loss, ys_loss_abs, marker="o", label="|CTR loss|", color="tab:blue")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("|CTR loss|", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if xs_gamma and ys_gamma:
        ax2 = ax1.twinx()
        ax2.plot(xs_gamma, ys_gamma, marker="s", color="tab:red", label="gamma_clip")
        ax2.set_ylabel("gamma_clip", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(alpha=0.3)
    ax1.set_title("Background Entropy Control")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()

    out_path = output_dir / "fig_bg_ctr.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _fig_sacr_replay_run(records: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    phase4_records = [r for r in records if r.get("phase") == "phase4"]
    if not _has_valid_data(phase4_records, "replay_priority_total"):
        _log.warning("[report] skipped fig_sacr_replay: no data")
        return None

    xs_total, ys_total = _get_series(phase4_records, "replay_priority_total")
    xs_selected, ys_selected = _get_series(phase4_records, "replay_selected")

    if not xs_total:
        _log.warning("[report] skipped fig_sacr_replay: no valid data")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = None
    ax1.bar(xs_total, ys_total, color="tab:blue", alpha=0.7, label="replay_priority_total")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("replay_priority_total", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if xs_selected and ys_selected:
        ax2 = ax1.twinx()
        ax2.plot(xs_selected, ys_selected, marker="o", color="tab:red", label="replay_selected")
        ax2.set_ylabel("replay_selected", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.grid(alpha=0.3, axis="y")
    ax1.set_title("SACR Replay")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()

    out_path = output_dir / "fig_sacr_replay.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def generate_report(
    run_dir: Path,
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        _log.warning("[report] skipped: metrics.jsonl not found")
        return {}

    records = _read_jsonl(metrics_path)
    if not records:
        _log.warning("[report] skipped: no records in metrics.jsonl")
        return {}

    if output_dir is None:
        output_dir = run_dir / "analysis"

    generated: Dict[str, Path] = {}

    for name, func in [
        ("fig_perf_miou_curves", _fig_perf_miou_curves_run),
        ("fig_perf_forgetting", _fig_perf_forgetting_run),
        ("fig_perf_stability_plasticity", _fig_perf_stability_plasticity_run),
        ("fig_theory_alpha_tau", _fig_theory_alpha_tau_run),
        ("fig_theory_iexc", _fig_theory_iexc_run),
        ("fig_theory_spectral", _fig_theory_spectral_run),
        ("fig_bg_ctr", _fig_bg_ctr_run),
        ("fig_sacr_replay", _fig_sacr_replay_run),
    ]:
        try:
            result = func(records, output_dir)
            if result is not None:
                generated[name] = result
        except Exception as exc:
            _log.warning("[report] %s failed: %s", name, exc)

    _log.info("[report] generated %d figures → %s", len(generated), output_dir)
    return generated
