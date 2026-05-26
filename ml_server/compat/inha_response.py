"""inha_RADA / Spring 호환 — /analyze 응답 형태 어댑터.

판정 코어(inference) 출력을 레거시 isolation_forest·score_breakdown 형태로 맞춘다.
"""
from __future__ import annotations

from typing import Any

# inha Grafana 9키 (retrieval 없으면 0 placeholder)
_BREAKDOWN_KEYS = (
    "resource", "network", "process", "episode",
    "correlation", "ml", "retrieval", "context_discount", "final",
)

_COMPUTE_TYPES = frozenset({
    "GPU_MINING", "CPU_ONLY_MINING",
    "GPU_CPU_IMBALANCE", "CPU_GPU_IMBALANCE", "HIGH_GPU", "HIGH_CPU",
    "Flat_Usage", "FLAT_USAGE",
})
_NETWORK_TYPES = frozenset({
    "POOL_TRAFFIC", "OUTBOUND_DOMINANT", "HIGH_OUTBOUND",
    "MANY_EXTERNAL", "STRATUM_PATTERN",
})
_PROCESS_TYPES = frozenset({"CONFIRMED_MINING"})


def normalize_verdict_severity(verdict: str, overall_severity: str) -> str:
    """OBSERVE → LOW (Spring AlertService 가 NORMAL 은 DB 스킵)."""
    if verdict == "OBSERVE" and overall_severity == "NORMAL":
        return "LOW"
    return overall_severity


def _sum_flag_scores(flag_scores: dict, types: frozenset) -> float:
    return round(
        sum(float(flag_scores.get(t, 0.0)) for t in types if t in flag_scores) * 100.0,
        2,
    )


def enrich_scores_for_inha(
    scores: dict,
    infer: dict,
    retrieval_evidence: dict | None = None,
) -> dict:
    """scores 에 score_breakdown 8키(+ retrieval) 추가 (기존 gpu_mining 등 키 유지)."""
    flag_scores = infer.get("flag_scores") or {}
    out = dict(scores)
    resource = _sum_flag_scores(flag_scores, _COMPUTE_TYPES)
    network = _sum_flag_scores(flag_scores, _NETWORK_TYPES)
    process = _sum_flag_scores(flag_scores, _PROCESS_TYPES)
    ml_pts = round(float(infer.get("ml_risk", 0.0)) * 100.0, 2)
    final = float(out.get("final", 0.0))
    retrieval_pts = 0.0
    if retrieval_evidence:
        try:
            retrieval_pts = float(retrieval_evidence.get("retrieval_score", 0) or 0)
        except (TypeError, ValueError):
            retrieval_pts = 0.0
    out["score_breakdown"] = {
        "resource":         resource,
        "network":          network,
        "process":          process,
        "episode":          0.0,
        "correlation":      round(max(0.0, resource + network + process - final * 0.3), 2),
        "ml":               ml_pts,
        "retrieval":        retrieval_pts,
        "context_discount": 0.0,
        "final":            final,
    }
    return out


def build_isolation_forest_block(
    infer: dict,
    *,
    history_size: int,
    metrics_boxplot: dict | None = None,
) -> dict[str, Any]:
    """레거시 anomaly_predictor.predict_anomaly() 형태."""
    ml_ready = bool(infer.get("ml_ready"))
    if_score = infer.get("if_score") or 0.0
    ensemble = infer.get("ensemble_score") or 0.0
    lof_score = float(if_score)  # LOF 미사용 앙상블 — 동일 축으로 호환

    bp = metrics_boxplot or {}
    bp_flag = bool(
        bp.get("available")
        and (bp.get("cpu_iqr_outlier") or bp.get("mem_iqr_outlier") or bp.get("cpu_outlier"))
    )

    if not ml_ready:
        sample_count = history_size
        return {
            "available":        False,
            "reason":           f"학습 데이터 수집 중 (pretrain 미완료, history={sample_count})",
            "is_anomaly":       infer.get("is_anomaly"),
            "weighted_score":   round(float(ensemble), 4) if ensemble else None,
            "if_score":         round(float(if_score), 4) if if_score else None,
            "lof_score":        round(lof_score, 4) if if_score else None,
            "if_anomaly":       bool(infer.get("is_anomaly")) if infer.get("is_anomaly") is not None else None,
            "lof_anomaly":      bool(infer.get("is_anomaly")) if infer.get("is_anomaly") is not None else None,
            "boxplot_flag":     bp_flag,
            "sample_count":     sample_count,
            "lof_window_size":  min(sample_count, 240),
            "contamination":    0.05,
            "boxplot_filtered": False,
        }

    is_anom = bool(infer.get("is_anomaly"))
    return {
        "available":        True,
        "is_anomaly":       is_anom,
        "if_score":         round(float(if_score), 4),
        "lof_score":        round(lof_score, 4),
        "weighted_score":   round(float(ensemble), 4),
        "if_anomaly":       is_anom,
        "lof_anomaly":      is_anom,
        "boxplot_flag":     bp_flag,
        "sample_count":     history_size,
        "lof_window_size":  min(history_size, 240),
        "contamination":    0.05,
        "boxplot_filtered": False,
        "model_version":    infer.get("model_version"),
        "model_scores":     infer.get("model_scores"),
    }
