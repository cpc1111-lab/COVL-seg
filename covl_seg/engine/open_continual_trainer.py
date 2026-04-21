from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from covl_seg.continual.methods import build_continual_method
from covl_seg.continual.task_partition import TaskDef, TaskPlan, build_task_plan
from covl_seg.engine.balanced_controller import (
    BalancedControllerConfig,
    BalancedControllerState,
    update_controller_state,
)
from covl_seg.engine.detectron2_runner import run_detectron2_eval, run_detectron2_train
from covl_seg.engine.hooks import append_metrics_jsonl
from covl_seg.engine.phase_runner import (
    run_phase1_hciba,
    run_phase2_joint,
    run_phase3_subspace_and_fusion,
    run_phase4_replay_update,
)


def _infer_num_classes_from_config(config_path: str) -> int:
    text = Path(config_path).read_text(encoding="utf-8").lower()
    if "ade20k_15" in text or "ade20k_100" in text:
        return 150
    if "coco" in text:
        return 164
    return 20


def _clip_overrides(mode: str) -> List[str]:
    lowered = mode.lower()
    if lowered not in {"none", "attention", "full"}:
        raise ValueError(f"Unsupported clip finetune mode: {mode}")
    return ["MODEL.SEM_SEG_HEAD.CLIP_FINETUNE", lowered]


def _write_task_class_indexes(task_dir: Path, task: TaskDef) -> Dict[str, Path]:
    split_dir = task_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_indexes = split_dir / "seen_indexes.json"
    test_indexes = split_dir / "unseen_indexes.json"
    train_indexes.write_text(json.dumps(task.seen_classes, indent=2), encoding="utf-8")
    test_indexes.write_text(json.dumps(task.background_classes, indent=2), encoding="utf-8")
    return {"train": train_indexes, "test": test_indexes}


def _build_phase_batch(task: TaskDef) -> Dict[str, object]:
    features_c = [float((c % 17) / 17.0) for c in task.seen_classes[:32]] or [0.0]
    features_s = [float((c % 19) / 19.0) for c in task.new_classes[:32]] or [0.0]
    targets = [float((c % 13) / 13.0) for c in task.seen_classes[:32]] or [0.0]
    bg_logits = [
        [float(((c + i) % 11) / 11.0) for i in range(5)]
        for c in (task.background_classes[:8] or [0])
    ]
    return {
        "features_c": features_c,
        "features_s": features_s,
        "targets": targets,
        "bg_logits": bg_logits,
    }


def _build_phase_batch_from_d2_metrics(task_dir: Path, fallback_batch: Dict[str, object]) -> Dict[str, object]:
    metrics_path = task_dir / "metrics.json"
    if not metrics_path.exists():
        return fallback_batch
    losses: List[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_loss = payload.get("total_loss")
        if total_loss is None:
            continue
        losses.append(float(total_loss))
    if not losses:
        return fallback_batch

    clipped = [max(1e-6, min(10.0, x)) for x in losses[:32]]
    avg = sum(clipped) / len(clipped)
    centered = [x - avg for x in clipped]
    bg_logits = [[x, x * 0.8, x * 0.6, x * 0.4, x * 0.2] for x in clipped[:8]]
    return {
        "features_c": [float(x) for x in clipped],
        "features_s": [float(x) for x in centered],
        "targets": [float(x / (avg + 1e-6)) for x in clipped],
        "bg_logits": bg_logits if bg_logits else fallback_batch["bg_logits"],
    }


def _compute_omega_tau_t(current_basis: List[float], basis_history: List[List[float]]) -> float:
    if not basis_history or not current_basis:
        return 0.0
    curr = torch.tensor(current_basis, dtype=torch.float32)
    curr_norm = torch.norm(curr) + 1e-8
    overlaps = []
    for prev_basis in basis_history:
        prev = torch.tensor(prev_basis, dtype=torch.float32)
        prev_norm = torch.norm(prev) + 1e-8
        cos = torch.dot(prev, curr) / (prev_norm * curr_norm)
        overlaps.append(float(torch.clamp(cos * cos, 0.0, 1.0).item()))
    return max(overlaps) if overlaps else 0.0


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_float_or(value: object, default: float) -> float:
    parsed = _safe_float(value)
    return default if parsed is None else parsed


def _build_d2_progress_callback(task_id: int, total_tasks: int, task_total_iters: int):
    iter_pattern = re.compile(r"iter:\s*(\d+)")
    last_iter = {"value": -1}
    started_at = time.monotonic()

    def _on_output(line: str) -> None:
        match = iter_pattern.search(line)
        if not match:
            return
        current_iter = int(match.group(1))
        if current_iter == last_iter["value"]:
            return
        last_iter["value"] = current_iter
        remaining = max(task_total_iters - current_iter, 0)
        progress_pct = 100.0 * min(max(current_iter, 0), max(task_total_iters, 1)) / max(task_total_iters, 1)
        elapsed = max(1e-6, time.monotonic() - started_at)
        iter_rate = max(current_iter, 1) / elapsed
        eta_sec = int(max(remaining, 0) / max(iter_rate, 1e-6))
        print(
            "[open-continual] "
            f"task {task_id}/{total_tasks} progress | "
            f"iter={current_iter}/{task_total_iters} | "
            f"remaining_task_iters={remaining} | "
            f"task_progress_pct={progress_pct:.1f} | "
            f"task_eta_sec={eta_sec}"
        )

    return _on_output


@dataclass
class OpenContinualTrainer:
    config_path: str
    output_dir: Path
    engine: str
    seed: int
    method_name: str
    clip_finetune: str
    task_spec: Optional[str]
    num_tasks: Optional[int]
    classes_per_task: Optional[int]
    task_seed: int
    n_pre: int
    n_main: int
    eps_f: float
    t_mem: str
    mix_ratio: List[int]
    m_max_total: int
    m_max_per_class: int
    ewc_lambda: float
    ewc_topk: int
    ewc_iters: int
    enable_ciba: bool
    enable_ctr: bool
    enable_spectral_ogp: bool
    enable_sacr: bool
    balanced_profile: str = "off"
    target_delta_new: float = 0.30
    epsilon_old: float = 0.20
    epsilon_all: float = 0.15
    epsilon_ov: float = 0.20
    seg_net: Optional[str] = None
    open_vocab_eval: bool = False
    resume_task: int = 0

    @classmethod
    def from_args(cls, args) -> "OpenContinualTrainer":
        return cls(
            config_path=args.config,
            output_dir=Path(args.output_dir),
            engine=args.engine,
            seed=args.seed,
            method_name=args.col_method,
            clip_finetune=args.clip_finetune,
            seg_net=args.seg_net,
            task_spec=args.task_spec,
            num_tasks=args.num_tasks,
            classes_per_task=args.classes_per_task,
            task_seed=args.task_seed,
            n_pre=args.n_pre,
            n_main=args.n_main,
            eps_f=args.eps_f,
            t_mem=args.t_mem,
            mix_ratio=list(args.mix_ratio),
            m_max_total=args.m_max_total,
            m_max_per_class=args.m_max_per_class,
            ewc_lambda=args.ewc_lambda,
            ewc_topk=args.ewc_topk,
            ewc_iters=args.ewc_iters,
            balanced_profile=args.balanced_profile,
            target_delta_new=args.target_delta_new,
            epsilon_old=args.epsilon_old,
            epsilon_all=args.epsilon_all,
            epsilon_ov=args.epsilon_ov,
            enable_ciba=args.enable_ciba,
            enable_ctr=args.enable_ctr,
            enable_spectral_ogp=args.enable_spectral_ogp,
            enable_sacr=args.enable_sacr,
            open_vocab_eval=args.open_vocab,
            resume_task=args.resume_task,
        )

    def _build_task_plan(self) -> TaskPlan:
        total_classes = _infer_num_classes_from_config(self.config_path)
        return build_task_plan(
            task_spec=self.task_spec,
            num_tasks=self.num_tasks,
            classes_per_task=self.classes_per_task,
            all_classes=list(range(total_classes)),
            seed=self.task_seed,
        )

    def _state_path(self) -> Path:
        return self.output_dir / "continual_state.json"

    def _metrics_path(self) -> Path:
        return self.output_dir / "metrics.jsonl"

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_state(self) -> dict:
        state_path = self._state_path()
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _validate_resume(self, state: dict) -> None:
        if not state:
            return
        prior_method = state.get("method")
        if prior_method is not None and prior_method != self.method_name:
            raise ValueError(
                f"Resume method mismatch: existing={prior_method}, requested={self.method_name}"
            )

    def run(self, max_tasks: Optional[int] = None) -> Dict[str, float]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        method = build_continual_method(
            name=self.method_name,
            config={
                "enable_ciba": self.enable_ciba,
                "enable_ctr": self.enable_ctr,
                "enable_spectral_ogp": self.enable_spectral_ogp,
                "enable_sacr": self.enable_sacr,
                "ewc_lambda": self.ewc_lambda,
                "ewc_topk": self.ewc_topk,
                "ewc_iters": self.ewc_iters,
            },
        )
        plan = self._build_task_plan()
        self._write_json(self.output_dir / "task_plan.json", plan.to_dict())

        state = self._load_state()
        self._validate_resume(state)

        alpha_star_history = list(state.get("alpha_star_history", []))
        tau_pred_history = list(state.get("tau_pred_history", []))
        basis_history = list(state.get("basis_history", []))
        balanced_prev_eval = dict(state.get("balanced_prev_eval", {}))

        balanced_enabled = self.balanced_profile == "balanced"
        balanced_cfg = BalancedControllerConfig(
            epsilon_ov=self.epsilon_ov,
            target_delta_new=self.target_delta_new,
        )
        balanced_state_payload = state.get("balanced_controller", {}) if balanced_enabled else {}
        balanced_state = BalancedControllerState(
            alpha_floor=_safe_float_or(balanced_state_payload.get("alpha_floor"), 0.0),
            g_stab=_safe_float_or(balanced_state_payload.get("g_stab"), 0.0),
            rho_old=_safe_float_or(balanced_state_payload.get("rho_old"), 0.0),
            rho_new=_safe_float_or(balanced_state_payload.get("rho_new"), 0.0),
            w_ctr=_safe_float_or(balanced_state_payload.get("w_ctr"), 0.0),
            ov_guard_triggered=bool(balanced_state_payload.get("ov_guard_triggered", False)),
        )

        task_cfg = {
            "n_pre": self.n_pre,
            "n_main": self.n_main,
            "eps_f": self.eps_f,
            "t_mem": self.t_mem,
            "mix_ratio": self.mix_ratio,
            "m_max_total": self.m_max_total,
            "m_max_per_class": self.m_max_per_class,
            "ewc_lambda": self.ewc_lambda,
            "ewc_topk": self.ewc_topk,
            "ewc_iters": self.ewc_iters,
            "enable_ciba": self.enable_ciba,
            "enable_ctr": self.enable_ctr,
            "enable_spectral_ogp": self.enable_spectral_ogp,
            "enable_sacr": self.enable_sacr,
        }
        clip_overrides = _clip_overrides(self.clip_finetune)

        completed = 0
        for task in plan.tasks:
            if task.task_id <= self.resume_task:
                continue
            if max_tasks is not None and completed >= max_tasks:
                break

            task_dir = self.output_dir / f"task_{task.task_id:03d}"
            total_tasks = len(plan.tasks)
            remaining_tasks = max(0, total_tasks - task.task_id)
            print(
                "[open-continual] "
                f"task {task.task_id}/{total_tasks} start | "
                f"task_iters={self.n_main} | remaining_task_iters={self.n_main} | "
                f"remaining_tasks={remaining_tasks}"
            )
            method.before_task(task_state={"task_id": task.task_id})
            method_phase_cfg = method.phase_overrides()
            balanced_knobs = {
                "balanced_w_ctr": 1.0,
                "balanced_w_oldfix": 0.0,
                "balanced_g_stab": 0.0,
                "balanced_alpha_floor": 0.0,
                "balanced_rho_new": 0.0,
                "balanced_rho_old": 0.0,
            }
            if balanced_enabled:
                balanced_knobs = {
                    "balanced_w_ctr": float(balanced_state.w_ctr),
                    "balanced_w_oldfix": float(balanced_state.rho_old),
                    "balanced_g_stab": float(balanced_state.g_stab),
                    "balanced_alpha_floor": float(balanced_state.alpha_floor),
                    "balanced_rho_new": float(balanced_state.rho_new),
                    "balanced_rho_old": float(balanced_state.rho_old),
                }
            phase_cfg = {**task_cfg, **method_phase_cfg, **balanced_knobs}
            batch = _build_phase_batch(task)

            p1 = run_phase1_hciba(task.task_id, phase_cfg, batch=batch)
            p2 = run_phase2_joint(task.task_id, phase_cfg, batch=batch)
            p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=batch)
            p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=batch)

            current_basis = p3.get("subspace_basis", [])
            p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))

            class_index_paths = _write_task_class_indexes(task_dir=task_dir, task=task)
            task_overrides = clip_overrides + [
                "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES",
                str(class_index_paths["train"]),
                "MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES",
                str(class_index_paths["test"]),
                "SOLVER.MAX_ITER",
                str(self.n_main),
                "TEST.EVAL_PERIOD",
                str(max(1, self.n_main)),
            ]

            eval_payload = None
            if self.engine == "d2":
                run_detectron2_train(
                    config_path=self.config_path,
                    output_dir=task_dir,
                    seed=self.seed,
                    resume_task=task.task_id - 1,
                    max_tasks=1,
                    seg_network=self.seg_net,
                    extra_overrides=task_overrides,
                    progress_callback=_build_d2_progress_callback(
                        task_id=task.task_id,
                        total_tasks=total_tasks,
                        task_total_iters=self.n_main,
                    ),
                )
                d2_batch = _build_phase_batch_from_d2_metrics(task_dir=task_dir, fallback_batch=batch)
                p1 = run_phase1_hciba(task.task_id, phase_cfg, batch=d2_batch)
                p2 = run_phase2_joint(task.task_id, phase_cfg, batch=d2_batch)
                p3 = run_phase3_subspace_and_fusion(task.task_id, phase_cfg, batch=d2_batch)
                p4 = run_phase4_replay_update(task.task_id, phase_cfg, batch=d2_batch)
                current_basis = p3.get("subspace_basis", [])
                p3["omega_tau_t"] = float(_compute_omega_tau_t(current_basis, basis_history))

                for record in (p1, p2, p3, p4):
                    record["proxy_source"] = "d2_metrics"
                    append_metrics_jsonl(self._metrics_path(), record)

                eval_payload = run_detectron2_eval(
                    config_path=self.config_path,
                    output_dir=task_dir,
                    resume_task=task.task_id,
                    checkpoint=None,
                    open_vocab=self.open_vocab_eval,
                    seg_network=self.seg_net,
                )
                append_metrics_jsonl(
                    self._metrics_path(),
                    {
                        "task": float(task.task_id),
                        "phase": "eval",
                        "mIoU_all": _safe_float(eval_payload.get("mIoU_all")),
                        "mIoU_old": _safe_float(eval_payload.get("mIoU_old")),
                        "mIoU_new": _safe_float(eval_payload.get("mIoU_new")),
                        "BG-mIoU": _safe_float(eval_payload.get("BG-mIoU")),
                        "engine": "d2",
                    },
                )
            else:
                for record in (p1, p2, p3, p4):
                    record["proxy_source"] = "synthetic"
                    append_metrics_jsonl(self._metrics_path(), record)

            if balanced_enabled:
                current_eval = {
                    "mIoU_all": _safe_float((eval_payload or {}).get("mIoU_all")),
                    "mIoU_old": _safe_float((eval_payload or {}).get("mIoU_old")),
                    "mIoU_new": _safe_float((eval_payload or {}).get("mIoU_new")),
                }
                prev_eval = {
                    "mIoU_all": _safe_float(balanced_prev_eval.get("mIoU_all")),
                    "mIoU_old": _safe_float(balanced_prev_eval.get("mIoU_old")),
                    "mIoU_new": _safe_float(balanced_prev_eval.get("mIoU_new")),
                }

                required_metrics = ("mIoU_all", "mIoU_old", "mIoU_new")
                has_finite_current = all(current_eval[key] is not None for key in required_metrics)
                has_finite_prev = all(prev_eval[key] is not None for key in required_metrics)

                delta_new = None
                delta_old = None
                delta_all = None
                ov_min_delta = None

                if has_finite_current and has_finite_prev:
                    delta_new = float(current_eval["mIoU_new"] - prev_eval["mIoU_new"])
                    delta_old = float(current_eval["mIoU_old"] - prev_eval["mIoU_old"])
                    delta_all = float(current_eval["mIoU_all"] - prev_eval["mIoU_all"])
                    ov_min_delta = float(min(delta_old, delta_all))
                    signals = {
                        "delta_new": delta_new,
                        "delta_old": delta_old,
                        "delta_all": delta_all,
                        "ov_min_delta": ov_min_delta,
                        "old_constraint_violated": delta_old < -self.epsilon_old,
                        "all_constraint_violated": delta_all < -self.epsilon_all,
                    }
                    balanced_state = update_controller_state(
                        state=balanced_state,
                        cfg=balanced_cfg,
                        signals=signals,
                    )

                append_metrics_jsonl(
                    self._metrics_path(),
                    {
                        "task": float(task.task_id),
                        "phase": "balanced_ctrl",
                        "delta_new": delta_new,
                        "delta_old": delta_old,
                        "delta_all": delta_all,
                        "ov_min_delta": ov_min_delta,
                        "alpha_floor": _safe_float(balanced_state.alpha_floor),
                        "ov_guard_triggered": bool(balanced_state.ov_guard_triggered),
                        "ov_guard_state": bool(balanced_state.ov_guard_state),
                    },
                )

                if has_finite_current:
                    balanced_prev_eval = current_eval

            method_state = method.after_task(task_state={"task_id": task.task_id})
            alpha_star_history.append(float(p3["alpha_star"]))
            tau_pred_history.append(float(p3["tau_pred"]))
            if current_basis:
                basis_history.append(current_basis)
            state = {
                "current_task": task.task_id,
                "method": method.name,
                "clip_finetune": self.clip_finetune,
                "alpha_star": p3["alpha_star"],
                "tau_pred": p3["tau_pred"],
                "alpha_star_history": alpha_star_history,
                "tau_pred_history": tau_pred_history,
                "basis_history": basis_history,
                "method_state": method_state.values,
                "last_task_dir": str(task_dir),
            }
            if balanced_enabled:
                state["balanced_controller"] = {
                    "alpha_floor": _safe_float(balanced_state.alpha_floor),
                    "g_stab": _safe_float(balanced_state.g_stab),
                    "rho_old": _safe_float(balanced_state.rho_old),
                    "rho_new": _safe_float(balanced_state.rho_new),
                    "w_ctr": _safe_float(balanced_state.w_ctr),
                    "ov_guard_triggered": bool(balanced_state.ov_guard_triggered),
                    "ov_guard_state": bool(balanced_state.ov_guard_state),
                }
                state["balanced_prev_eval"] = {
                    "mIoU_all": _safe_float(balanced_prev_eval.get("mIoU_all")),
                    "mIoU_old": _safe_float(balanced_prev_eval.get("mIoU_old")),
                    "mIoU_new": _safe_float(balanced_prev_eval.get("mIoU_new")),
                }
            self._write_json(self._state_path(), state)
            print(
                "[open-continual] "
                f"task {task.task_id}/{total_tasks} done | remaining_task_iters=0 | "
                f"remaining_tasks={remaining_tasks}"
            )
            completed += 1

        return {
            "tasks_executed": float(completed),
            "last_task": float(state.get("current_task", self.resume_task)),
        }
