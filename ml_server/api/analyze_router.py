"""POST /analyze — 메인 분석 엔드포인트.

[F-ML+rule 채택] 탐지 코어를 ml_server2 의 rule_based + ML 앙상블 + scoring
조합으로 교체. Spring 측 입력(MetricsRequest)·응답(MlResponse) 계약은 그대로.

응답 키 매핑:
  · ml_server2 anomaly_types / rule_score / ml_risk / total_risk
    → 본 라우터에서 verdict / overall_severity / alerts / scores 로 합성
  · retrieval_evidence, agent, global_hw_degradation 은 기존과 동일

verdict 매핑 (total_risk 기반):
  ≥ 0.85  : HIGH_RISK   (overall_severity = HIGH)
  ≥ 0.60  : SUSPICIOUS  (overall_severity = MEDIUM)
  ≥ 0.30  : OBSERVE     (overall_severity = LOW — inha 호환)
  그 외   : NORMAL
"""
import datetime
import time as _time
from typing import Any, Dict, List
import numpy as np
from fastapi import APIRouter

from ..config import get_timetable_slot
from ..feature.feature_builder import make_snapshot
from ..model.requests import MetricsRequest
from ..storage import pc_history_store
from ..core.inference import run_inference_snapshot
from ..core.online_training import maybe_trigger_from_analyze, track_analyze_sample
from ..core.trainer import train_async
from ..compat import (
    build_isolation_forest_block,
    enrich_scores_for_inha,
    normalize_verdict_severity,
)
from ..detector.global_degradation import detect_global_hw_degradation
from ..agent.runner import run_ai_agent
from ..retrieval import (
    build_segment,
    build_embedding,
    search_similar,
    add_segment,
    build_retrieval_evidence,
)

router = APIRouter()

POLICY_VERSION = "ml-rule-v1.0"

# anomaly_type → (severity, scores key) 매핑. Mining/Compute = HIGH/MEDIUM, Network = LOW.
_ALERT_SEVERITY: Dict[str, str] = {
    "GPU_MINING":         "HIGH",
    "CPU_ONLY_MINING":    "HIGH",
    "CONFIRMED_MINING":   "HIGH",
    "GPU_CPU_IMBALANCE":  "MEDIUM",
    "CPU_GPU_IMBALANCE":  "MEDIUM",
    "HIGH_GPU":           "MEDIUM",
    "HIGH_CPU":           "MEDIUM",
    "POOL_TRAFFIC":       "HIGH",
    "OUTBOUND_DOMINANT":  "MEDIUM",
    "HIGH_OUTBOUND":      "LOW",
    "MANY_EXTERNAL":      "LOW",
    "STRATUM_PATTERN":    "HIGH",
    "Flat_Usage":         "MEDIUM",
    "FLAT_USAGE":         "MEDIUM",
    "Pattern":            "MEDIUM",
}

# anomaly_type → 사람이 읽는 detail 텍스트
_ALERT_DETAIL: Dict[str, str] = {
    "GPU_MINING":         "GPU 고부하 + CPU/GPU 불균형 + 외부 송신 → GPU 채굴 강한 의심",
    "CPU_ONLY_MINING":    "GPU 유휴 + CPU 고부하 + 외부 송신 → CPU(Monero) 채굴 의심",
    "CONFIRMED_MINING":   "알려진 채굴 프로세스 실행 감지",
    "GPU_CPU_IMBALANCE":  "GPU 부하 대비 CPU 점유가 비정상적으로 낮음",
    "CPU_GPU_IMBALANCE":  "CPU 부하 대비 GPU 점유가 비정상적으로 낮음",
    "HIGH_GPU":           "GPU 부하 임계 초과",
    "HIGH_CPU":           "CPU 부하 임계 초과",
    "POOL_TRAFFIC":       "외부 패킷 × 송신량 복합 신호 — 채굴 풀 통신 의심",
    "OUTBOUND_DOMINANT":  "송신량이 수신량을 크게 초과 — 작업 결과 송출 패턴",
    "HIGH_OUTBOUND":      "외부 송신량 임계 초과",
    "MANY_EXTERNAL":      "외부 연결 패킷 수 임계 초과",
    "STRATUM_PATTERN":    "패킷 多 + 소량 송신 — Stratum JSON-RPC 채굴 패턴",
    "Flat_Usage":         "장시간 평탄한 고부하 — 채굴형 지속 패턴",
    "FLAT_USAGE":         "장시간 평탄한 고부하 — 채굴형 지속 패턴",
    "Pattern":            "ML 앙상블 기반 패턴 이상",
}

# 카테고리 게이팅용 플래그 분류 — rule_based.py 의 D/A/B 카테고리와 1:1 매핑.
_RESOURCE_FLAGS = {
    "GPU_MINING", "CPU_ONLY_MINING", "CONFIRMED_MINING",
    "GPU_CPU_IMBALANCE", "CPU_GPU_IMBALANCE", "HIGH_GPU", "HIGH_CPU",
}
_NETWORK_FLAGS = {
    "POOL_TRAFFIC", "OUTBOUND_DOMINANT", "HIGH_OUTBOUND",
    "MANY_EXTERNAL", "STRATUM_PATTERN",
}
_SYSTEM_FLAGS = {"Flat_Usage", "FLAT_USAGE"}

_VERDICT_TO_GATING = {
    "HIGH_RISK":  "DANGEROUS",
    "SUSPICIOUS": "SUSPICIOUS",
    "OBSERVE":    "OBSERVE",
    "NORMAL":     "NORMAL",
}


# scores 매핑 — 기존 mock_agent 가 참조하는 키 (gpu_mining/cpu_mining/...) 를 보존.
_SCORE_KEY: Dict[str, str] = {
    "GPU_MINING":         "gpu_mining",
    "CPU_ONLY_MINING":    "cpu_mining",
    "CONFIRMED_MINING":   "process",
    "GPU_CPU_IMBALANCE":  "gpu_mining",
    "CPU_GPU_IMBALANCE":  "cpu_mining",
    "HIGH_GPU":           "gpu_mining",
    "HIGH_CPU":           "cpu_mining",
    "POOL_TRAFFIC":       "exfil",
    "OUTBOUND_DOMINANT":  "exfil",
    "HIGH_OUTBOUND":      "exfil",
    "MANY_EXTERNAL":      "exfil",
    "STRATUM_PATTERN":    "stealth",
    "Flat_Usage":         "stealth",
    "FLAT_USAGE":         "stealth",
}


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _verdict_from_risk(total_risk: float, is_anomaly: bool) -> tuple[str, str]:
    """(verdict, overall_severity) 반환."""
    if total_risk >= 0.85:
        return "HIGH_RISK", "HIGH"
    if total_risk >= 0.60 or is_anomaly:
        return "SUSPICIOUS", "MEDIUM"
    if total_risk >= 0.30:
        return "OBSERVE", "LOW"
    return "NORMAL", "NORMAL"


def _build_alerts(infer: dict) -> List[dict]:
    alerts: List[dict] = []
    flag_scores = infer.get("flag_scores", {})
    for atype in infer["anomaly_types"]:
        alerts.append({
            "type":     atype,
            "severity": _ALERT_SEVERITY.get(atype, "LOW"),
            "detail":   _ALERT_DETAIL.get(atype, f"{atype} 신호 감지"),
            "score":    round(float(flag_scores.get(atype, 0.0)), 4),
        })
    return alerts


def _build_scores(infer: dict) -> Dict[str, float]:
    """mock_agent / claude_api_agent 가 참조하는 키 셋을 보존하며 합성."""
    flag_scores = infer.get("flag_scores", {})
    bucketed: Dict[str, float] = {
        "gpu_mining": 0.0,
        "cpu_mining": 0.0,
        "exfil":      0.0,
        "stealth":    0.0,
        "dos":        0.0,
        "mem":        0.0,
        "process":    0.0,
    }
    for atype, score in flag_scores.items():
        key = _SCORE_KEY.get(atype)
        if key is None:
            continue
        bucketed[key] = max(bucketed[key], round(float(score) * 100.0, 2))

    return {
        **bucketed,
        "final":              round(infer["total_risk"] * 100.0, 2),
        "rule":               round(infer["rule_score"] * 100.0, 2),
        "ml":                 round(infer["ml_risk"]    * 100.0, 2),
        "ensemble":           infer["ensemble_score"],
        "if_score":           infer["if_score"],
        "context_multiplier": 1.0,
    }


def _build_signals(snapshot: dict, infer: dict) -> Dict[str, Any]:
    """레거시 verdict_classifier signals 슬롯 호환."""
    known = infer.get("known_miners") or []
    return {
        "is_gaming":      False,
        "is_compiling":   False,
        "mining_pool_ip": bool(infer.get("mining_pool_ip")),
        "persistent_miner": len(known) > 0,
        "known_miners":   known,
        "anomaly_types":  infer["anomaly_types"],
        "model_version":  infer["model_version"],
        "ml_ready":       infer["ml_ready"],
    }


def _sustained_minutes(pc_id: str) -> int:
    """pc_history 말단부터 cpu>=30 또는 gpu>=30 인 스냅샷을 연속으로 카운트.
    클라이언트가 분 단위로 보내므로 1 snapshot ≈ 1 minute 로 본다."""
    history = pc_history_store.pc_history.get(pc_id)
    if not history:
        return 0
    streak = 0
    for snap in reversed(history):
        cpu = float(snap.get("cpu_percent") or 0)
        gpu = float(snap.get("gpu_percent") or 0)
        if cpu >= 30.0 or gpu >= 30.0:
            streak += 1
        else:
            break
    return streak


def _build_category_signals(
    infer: dict,
    global_hw: dict,
    pc_id: str,
    verdict: str,
) -> Dict[str, Any]:
    """Spring `MlResponse.categorySignals` 계약에 맞춘 게이팅 신호 합성.

    rule_based 의 D/A/B 카테고리 플래그를 resource/network/system 으로 재그룹화.
    """
    types = set(infer.get("anomaly_types") or [])
    resource_abnormal = bool(types & _RESOURCE_FLAGS)
    network_abnormal  = bool(types & _NETWORK_FLAGS)
    system_abnormal   = bool(types & _SYSTEM_FLAGS) or bool(global_hw.get("detected"))

    return {
        "resource_abnormal":   resource_abnormal,
        "network_abnormal":    network_abnormal,
        "system_abnormal":     system_abnormal,
        "sustained_minutes":   _sustained_minutes(pc_id),
        "triggered_patterns":  list(infer.get("anomaly_types") or []),
        "verdict_from_gating": _VERDICT_TO_GATING.get(verdict, "NORMAL"),
    }


def _signals_missing(snapshot: dict) -> List[str]:
    """F5 호환 — snapshot 에서 누락된 신호 분류."""
    missing: List[str] = []
    if snapshot.get("gpu_percent") is None:
        missing.append("gpu")
    if snapshot.get("outbound_mb") is None and snapshot.get("inbound_mb") is None:
        missing.append("network")
    if not snapshot.get("top_processes"):
        missing.append("process")
    return missing


@router.post("/analyze")
def analyze(metrics: MetricsRequest):
    pc_id = metrics.pc_id
    dt    = datetime.datetime.fromisoformat(metrics.timestamp)
    slot  = get_timetable_slot(dt)

    history = pc_history_store.ensure_pc_history(pc_id)
    snapshot = make_snapshot(metrics)
    history.append(snapshot)
    track_analyze_sample(pc_id, snapshot)
    pc_history_store.update_train_history(pc_id, slot, snapshot)

    pc_history_store.all_pc_latest[pc_id] = {
        "cpu_percent":           metrics.cpu_percent,
        "memory_percent":        metrics.memory_percent,
        "timestamp":             metrics.timestamp,
        "slot":                  slot,
        "inbound_mb":            metrics.inbound_mb,
        "outbound_mb":           metrics.outbound_mb,
        "disk_read_mb":          metrics.disk_read_mb,
        "disk_write_mb":         metrics.disk_write_mb,
        "gpu_percent":           metrics.gpu.load_percent if metrics.gpu else 0.0,
        "external_packet_count": metrics.external_packet_count,
        "_ts":                   _time.time(),
    }

    # ── ML+rule 추론 (단일 진입점) ────────────────────────────────────────
    infer = run_inference_snapshot(
        snapshot, pc_id=pc_id, timestamp=dt,
        append_history=False,   # 위에서 이미 ensure_pc_history 에 append 했음
        slot_override=slot,
    )

    maybe_trigger_from_analyze(train_async)

    verdict, overall_severity = _verdict_from_risk(infer["total_risk"], infer["is_anomaly"])
    overall_severity = normalize_verdict_severity(verdict, overall_severity)
    alerts = _build_alerts(infer)
    if infer.get("known_miners") and not any(a.get("type") == "CONFIRMED_MINING" for a in alerts):
        names = ", ".join(
            (m.get("name") or m.get("process_name") or "?") for m in infer["known_miners"][:3]
        )
        alerts.insert(0, {
            "type":     "CONFIRMED_MINING",
            "severity": "HIGH",
            "detail":   f"채굴 프로세스: {names}",
            "score":    round(float(infer.get("flag_scores", {}).get("CONFIRMED_MINING", 0.8)), 4),
        })
    signals = _build_signals(snapshot, infer)

    # ── Retrieval evidence (segment → embedding → top-k 검색) ────────────
    current_segment = build_segment(pc_id, slot, history, window_size=12)
    retrieval_evidence = None
    current_embedding = None
    if current_segment is not None:
        current_embedding = build_embedding(current_segment)
        retrieved = search_similar(current_segment, current_embedding, top_k=3)
        retrieval_evidence = build_retrieval_evidence(
            current_segment, retrieved,
            peer_latest=pc_history_store.all_pc_latest,
        )

    scores = enrich_scores_for_inha(
        _build_scores(infer), infer, retrieval_evidence=retrieval_evidence,
    )

    pattern_result = {
        "overall_severity":   overall_severity,
        "verdict":            verdict,
        "alerts":             alerts,
        "scores":             scores,
        "signals":            signals,
        "timetable_slot":     slot,
        "policy_version":     POLICY_VERSION,
        "retrieval_evidence": retrieval_evidence,
    }

    # 현재 segment 를 검색 후에 저장 → 자기 자신은 top-k 에 잡히지 않음
    if current_segment is not None and current_embedding is not None:
        try:
            add_segment(
                current_segment, current_embedding,
                verdict=verdict,
                score=float(scores["final"]),
            )
        except Exception:
            pass

    # ── 전체 PC 노후화 ────────────────────────────────────────────────────
    global_hw = detect_global_hw_degradation()
    if global_hw.get("detected"):
        alerts.append({
            "type":     "GLOBAL_HW_DEGRADATION",
            "severity": "MEDIUM",
            "detail":   global_hw["detail"],
        })
        if overall_severity == "NORMAL":
            overall_severity = "MEDIUM"
            pattern_result["overall_severity"] = "MEDIUM"

    # ── AI Agent ─────────────────────────────────────────────────────────
    agent_result = None
    if overall_severity != "NORMAL":
        agent_result = run_ai_agent(metrics, pattern_result, global_hw)

    iso_block = build_isolation_forest_block(
        infer,
        history_size=len(history),
        metrics_boxplot=metrics.boxplot_signal,
    )

    category_signals = _build_category_signals(infer, global_hw, pc_id, verdict)

    return _sanitize({
        "pc_id":               pc_id,
        "timestamp":           metrics.timestamp,
        "timetable_slot":      slot,
        "overall_severity":    overall_severity,
        "verdict":             verdict,
        "policy_version":      POLICY_VERSION,
        "alerts":              alerts,
        "scores":              scores,
        "signals":             signals,
        "history_size":        len(history),
        "isolation_forest":    iso_block,
        "global_hw_degradation": global_hw,
        "agent":               agent_result,
        "retrieval_evidence":  retrieval_evidence,
        "signals_missing":     _signals_missing(snapshot),
        "category_signals":    category_signals,
    })
