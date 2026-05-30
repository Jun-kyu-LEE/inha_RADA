"""snapshot/MetricsRequest → 피처 벡터 변환 (외부 진입점).

11 raw + derived(코드 계산) 만 사용. ``training_adapter`` 가 단일 소스.
"""
from typing import Optional

from ..model.requests import MetricsRequest
from .columns import RAW_METRIC_COLS
from .training_adapter import extract_one


def _raw_from_gpu_metrics(metrics: MetricsRequest) -> dict:
    gpu = metrics.gpu
    return {
        "cpu_percent":           metrics.cpu_percent,
        "memory_percent":        metrics.memory_percent,
        "gpu_percent":           gpu.load_percent if gpu else 0.0,
        "gpu_vram_mb":           gpu.memory_used_mb if gpu else 0.0,
        "gpu_total_mb":          gpu.memory_total_mb if gpu else 8192,
        "gpu_power_w":           gpu.power_draw_w if gpu else 0.0,
        "inbound_mb":            metrics.inbound_mb,
        "outbound_mb":           metrics.outbound_mb,
        "external_packet_count": metrics.external_packet_count,
        "disk_read_mb":          metrics.disk_read_mb,
        "disk_write_mb":         metrics.disk_write_mb,
    }


def build_features(
    cpu: float, memory: float,
    gpu_pct: float, vram_mb: float, gpu_total_mb: float,
    disk_r: float, disk_w: float, power: float,
    inbound: float = 0.0, outbound: float = 0.0, external: float = 0.0,
) -> list:
    snap = {
        "cpu_percent": cpu, "memory_percent": memory,
        "gpu_percent": gpu_pct, "gpu_vram_mb": vram_mb,
        "gpu_total_mb": gpu_total_mb if gpu_total_mb else 8192,
        "disk_read_mb": disk_r, "disk_write_mb": disk_w,
        "gpu_power_w": power,
        "inbound_mb": inbound, "outbound_mb": outbound,
        "external_packet_count": external,
    }
    return extract_one(snap)[0].tolist()


def extract_features_from_snapshot(snap: dict) -> Optional[list]:
    try:
        return extract_one(snap)[0].tolist()
    except Exception:
        return None


def extract_features_from_metrics(metrics: MetricsRequest) -> list:
    return extract_one(_raw_from_gpu_metrics(metrics))[0].tolist()


def make_snapshot(metrics: MetricsRequest) -> dict:
    """analyze 라우터 snapshot — 11 raw + API 보조(top_processes, external_connections)."""
    snap = _raw_from_gpu_metrics(metrics)
    snap["timestamp"] = metrics.timestamp
    snap["top_processes"] = metrics.top_processes
    snap["external_connections"] = metrics.external_connections
    # None → 0.0 (룰/ML 공통)
    for key in RAW_METRIC_COLS:
        if snap.get(key) is None:
            snap[key] = 0.0
    if snap.get("gpu_percent") is None:
        snap["gpu_percent"] = 0.0
    return snap
