"""카테고리 기반 룰 — 11 raw 메트릭만 사용.

  [D] Mining Signature  — MAX (GPU_MINING, CPU_ONLY_MINING, CONFIRMED_MINING*)
  [A] Compute Evidence  — MAX (GPU_CPU_IMBALANCE, CPU_GPU_IMBALANCE, HIGH_GPU, HIGH_CPU)
  [B] Network Evidence  — sum + network_cap

  rule_score = D + A + B (최대 1.0)

  * CONFIRMED_MINING 은 live /analyze 의 top_processes 전용 (CSV 11 raw 외).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import MINING_PROCESSES, MINING_POOL_IPS
from .config_cache import load_yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "thresholds.yaml"


def _normalize_process_name(name: str) -> str:
    return (name or "").strip().lower()


def _is_mining_process(name: str) -> bool:
    n = _normalize_process_name(name)
    if not n:
        return False
    if n in MINING_PROCESSES:
        return True
    return any(n == m or n.endswith(m) for m in MINING_PROCESSES if "." in m or len(m) > 4)


def _find_known_miners(top_processes: list | None) -> list[dict]:
    miners: list[dict] = []
    for proc in top_processes or []:
        name = proc.get("name") or proc.get("process_name") or ""
        if _is_mining_process(name):
            miners.append(proc)
    return miners


def _has_mining_pool_ip(external_connections: list | None) -> bool:
    for conn in external_connections or []:
        ip = str(conn.get("remote_ip") or "")
        if any(ip.startswith(prefix) for prefix in MINING_POOL_IPS):
            return True
    return False


def rule_based_detect(metrics: dict, slot: str, cfg: dict | None = None) -> dict:
    """단일 snapshot dict → 카테고리 플래그 기반 위험 점수(0.0~1.0) 및 타입 반환."""
    if cfg is None:
        cfg = load_yaml(CONFIG_PATH)
    fw = cfg.get("flag_weights", {})
    defaults = cfg.get("default_rules", {})
    slot_rules = cfg.get(f"{slot.lower()}_slot", {}).get("rules", {})
    th = {**defaults, **slot_rules}

    cpu      = float(metrics.get("cpu_percent", 0) or 0)
    gpu      = float(metrics.get("gpu_percent", 0) or 0)
    outbound = float(metrics.get("outbound_mb", 0) or 0)
    inbound  = float(metrics.get("inbound_mb", 0) or 0)
    ext      = float(metrics.get("external_packet_count", 0) or 0)

    d_flags: dict[str, float] = {}

    if gpu > th.get("gpu_mining_gpu_min", 70) and \
       cpu / (gpu + 0.001) < th.get("gpu_mining_ratio_max", 0.30) and \
       (outbound > th.get("outbound_mb", 80) or ext > th.get("external_packet_count", 100)):
        d_flags["GPU_MINING"] = fw.get("GPU_MINING", 0.75)

    if gpu < th.get("cpu_only_gpu_max", 10) and \
       cpu > th.get("cpu_only_cpu_min", 70) and \
       outbound > th.get("cpu_only_outbound_min", 20):
        d_flags["CPU_ONLY_MINING"] = fw.get("CPU_ONLY_MINING", 0.70)

    known_miners = _find_known_miners(metrics.get("top_processes"))
    if known_miners:
        d_flags["CONFIRMED_MINING"] = fw.get("CONFIRMED_MINING", 0.80)

    d_score = max(d_flags.values()) if d_flags else 0.0

    a_flags: dict[str, float] = {}

    if gpu > th.get("gpu_imbalance_gpu_min", 50) and \
       cpu / (gpu + 0.001) < th.get("cpu_gpu_ratio_min", 0.30):
        a_flags["GPU_CPU_IMBALANCE"] = fw.get("GPU_CPU_IMBALANCE", 0.35)

    if cpu > th.get("cpu_only_cpu_min", 70) and gpu < th.get("cpu_only_gpu_max", 10):
        a_flags["CPU_GPU_IMBALANCE"] = fw.get("CPU_GPU_IMBALANCE", 0.30)

    if gpu > th.get("gpu_percent", 85):
        a_flags["HIGH_GPU"] = fw.get("HIGH_GPU", 0.25)

    if cpu > th.get("cpu_percent", 90):
        a_flags["HIGH_CPU"] = fw.get("HIGH_CPU", 0.20)

    a_score = max(a_flags.values()) if a_flags else 0.0

    b_flags: dict[str, float] = {}

    if ext * outbound > th.get("ext_traffic_complex", 1000):
        b_flags["POOL_TRAFFIC"] = fw.get("POOL_TRAFFIC", 0.45)

    if outbound > th.get("outbound_dominant_min_mb", 5) and \
       outbound / (inbound + 0.001) > th.get("traffic_ratio_min", 5.0):
        b_flags["OUTBOUND_DOMINANT"] = fw.get("OUTBOUND_DOMINANT", 0.30)

    if outbound > th.get("outbound_mb", 80):
        b_flags["HIGH_OUTBOUND"] = fw.get("HIGH_OUTBOUND", 0.20)

    if ext > th.get("external_packet_count", 100):
        b_flags["MANY_EXTERNAL"] = fw.get("MANY_EXTERNAL", 0.20)

    if ext > th.get("stratum_ext_min", 50) and \
       outbound < th.get("stratum_outbound_max", 5) and \
       ext / (outbound + 0.5) > th.get("stratum_ratio_min", 30):
        b_flags["STRATUM_PATTERN"] = fw.get("STRATUM_PATTERN", 0.25)

    b_score = min(sum(b_flags.values()), fw.get("network_cap", 0.60))

    risk_score = min(1.0, d_score + a_score + b_score)
    all_flags = {**d_flags, **a_flags, **b_flags}

    return {
        "risk_score":      risk_score,
        "is_anomaly":      risk_score > 0,
        "anomaly_types":   list(all_flags.keys()),
        "flag_scores":     all_flags,
        "known_miners":    known_miners,
        "mining_pool_ip":  _has_mining_pool_ip(metrics.get("external_connections")),
    }
