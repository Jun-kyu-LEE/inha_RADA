"""추론 파이프라인 — 단일 진입점."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from .ensemble import EnsembleScorer
from .history import history_manager
from .rule_based import rule_based_detect
from .scoring import combine_risk, ml_risk_score
from .slot_classifier import classify_slot
from .config_cache import load_yaml
from .trainer import trainer_state
from ..feature.training_adapter import extract_one, apply_weights
from ..storage import pc_history_store

logger = logging.getLogger(__name__)


_scorer: EnsembleScorer | None = None
_scorer_version: str = ""


def reset_scorer_cache() -> None:
    global _scorer, _scorer_version
    _scorer = None
    _scorer_version = ""


def _get_scorer() -> EnsembleScorer | None:
    global _scorer, _scorer_version
    if not trainer_state.is_ready:
        return None
    if _scorer_version != trainer_state.version:
        _scorer = EnsembleScorer(trainer_state.models)
        _scorer_version = trainer_state.version
    return _scorer


def _flat_usage_boost(
    pc_id: str,
    slot: str,
    cfg: dict,
    rule_score: float,
    flag_scores: dict,
    anomaly_types: list,
) -> tuple[float, dict, list]:
    """pc_history_store 기반 평탄 고부하 → rule_score 가산."""
    flat_cfg = (cfg.get("flat_usage") or {}).get((slot or "free").lower())
    if not flat_cfg:
        return rule_score, flag_scores, anomaly_types

    history = pc_history_store.pc_history.get(pc_id)
    if not history or len(history) < 10:
        return rule_score, flag_scores, anomaly_types

    last_n = min(120, len(history))
    rows = list(history)[-last_n:]
    mean_th = float(flat_cfg.get("mean_threshold", 30.0))
    cv_th = float(flat_cfg.get("cv_threshold", 0.08))
    weight = float(flat_cfg.get("weight", 0.25))

    for feat in ("cpu_percent", "gpu_percent"):
        vals = [float(r.get(feat) or 0.0) for r in rows]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / (mean + 0.001)
        if mean > mean_th and cv < cv_th:
            flag_scores = dict(flag_scores)
            flat_flag = "Flat_Usage"
            flag_scores[flat_flag] = weight
            if flat_flag not in anomaly_types:
                anomaly_types = list(anomaly_types) + [flat_flag]
            return min(1.0, rule_score + weight), flag_scores, anomaly_types

    return rule_score, flag_scores, anomaly_types


def _ml_branch(
    snapshot: dict,
    pc_id: str,
    slot: str,
    cfg: dict,
    scorer: EnsembleScorer,
) -> tuple[float, float, dict[str, float], float, str]:
    X_raw = extract_one(snapshot)
    X_scaled = trainer_state.scaler.transform(X_raw)
    X = apply_weights(X_scaled, slot=slot)

    ml = scorer.compute(pc_id, X, slot=slot)
    ensemble = ml["ensemble_score"]
    threshold = ml["threshold"]
    risk = ml_risk_score(ensemble, threshold, cfg.get("ml_risk_mapping"))

    model_scores = {k: v for k, v in ml["raw_scores"].items()}
    if_score = ml.get("if_score", ensemble)
    return risk, ensemble, model_scores, if_score, trainer_state.version


def run_inference_snapshot(
    snapshot: dict,
    pc_id: str,
    timestamp: datetime,
    *,
    append_history: bool = True,
    slot_override: str | None = None,
) -> dict:
    if append_history:
        history_manager.append(pc_id, snapshot)

    slot = slot_override or classify_slot(timestamp)
    cfg = load_yaml(_thresholds_path())

    rule_result = rule_based_detect(snapshot, slot)
    rule_score = float(rule_result["risk_score"])
    anomaly_types: list[str] = list(rule_result["anomaly_types"])
    flag_scores: dict[str, float] = dict(rule_result.get("flag_scores", {}))

    rule_score, flag_scores, anomaly_types = _flat_usage_boost(
        pc_id, slot, cfg, rule_score, flag_scores, anomaly_types,
    )

    ml_risk = 0.0
    ensemble_score = 0.0
    if_score = 0.0
    model_scores: dict[str, float] = {}
    model_version = "rule_based"

    scorer = _get_scorer()
    if scorer is not None:
        try:
            ml_risk, ensemble_score, model_scores, if_score, model_version = _ml_branch(
                snapshot, pc_id, slot, cfg, scorer,
            )
        except Exception as e:
            logger.warning("ML 추론 실패 — rule-based 단독으로 진행: %s", e)
            scorer = None

    total_risk, is_anomaly = combine_risk(
        ml_risk, rule_score, cfg, model_ready=scorer is not None,
    )

    if is_anomaly and not anomaly_types:
        anomaly_types = ["Pattern"]

    rule_flag_th = cfg.get("rule_based_flag_threshold", 0.3)

    return {
        "pc_id":           pc_id,
        "slot":            slot,
        "is_anomaly":      bool(is_anomaly),
        "total_risk":      round(float(total_risk), 4),
        "rule_score":      round(float(rule_score), 4),
        "ml_risk":         round(float(ml_risk), 4),
        "ensemble_score":  round(float(ensemble_score), 4),
        "if_score":        round(float(if_score), 4),
        "model_scores":    model_scores,
        "anomaly_types":   anomaly_types,
        "flag_scores":     flag_scores,
        "model_version":   model_version,
        "rule_based":      rule_score > rule_flag_th,
        "ml_ready":        scorer is not None,
        "known_miners":    rule_result.get("known_miners", []),
        "mining_pool_ip":  bool(rule_result.get("mining_pool_ip")),
    }


def _thresholds_path():
    from pathlib import Path
    return Path(__file__).parent.parent / "config" / "thresholds.yaml"
