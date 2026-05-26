"""End-to-end 통합 테스트용 시나리오 페이로드.

ML 서버 /analyze 입력(Spring MetricsRequest 페이로드 호환) 4종.
"""
from __future__ import annotations
import datetime as _dt


def _ts(offset_sec: int = 0) -> str:
    return (_dt.datetime.now() + _dt.timedelta(seconds=offset_sec)).strftime("%Y-%m-%dT%H:%M:%S")


def normal_payload(pc_id: str = "PC-NORMAL", offset: int = 0) -> dict:
    """저부하·정상 트래픽 → verdict=NORMAL 기대."""
    return {
        "pc_id": pc_id,
        "timestamp": _ts(offset),
        "cpu_percent": 8.0,
        "cpu_core_count": 8,
        "memory_percent": 32.0,
        "memory_used_gb": 5.1,
        "memory_total_gb": 16.0,
        "inbound_mb": 0.4,
        "outbound_mb": 0.2,
        "inbound_total_mb": 102.4,
        "outbound_total_mb": 88.7,
        "external_packet_count": 12,
        "external_connection_count": 2,
        "external_connections": [],
        "active_ports": [443, 80],
        "disk_read_mb": 1.0,
        "disk_write_mb": 0.5,
        "gpu": {
            "name": "RTX 3060",
            "load_percent": 3.0,
            "memory_used_mb": 200,
            "memory_total_mb": 12000,
            "memory_percent": 1.6,
            "temperature": 45.0,
        },
        "top_processes": [
            {"name": "chrome.exe", "cpu_percent": 4.0, "memory_mb": 1200},
        ],
        "loop_elapsed": 0.5,
        "local_alerts": [],
        "boxplot_signal": {},
    }


def gpu_mining_payload(pc_id: str = "PC-GPU-MINING", offset: int = 0) -> dict:
    """high GPU + 외부 송신 ↑ → GPU_MINING/POOL_TRAFFIC HIGH_RISK 기대."""
    return {
        "pc_id": pc_id,
        "timestamp": _ts(offset),
        "cpu_percent": 12.0,
        "cpu_core_count": 8,
        "memory_percent": 41.0,
        "memory_used_gb": 6.5,
        "memory_total_gb": 16.0,
        "inbound_mb": 0.2,
        "outbound_mb": 6.0,
        "inbound_total_mb": 55.0,
        "outbound_total_mb": 980.0,
        "external_packet_count": 950,
        "external_connection_count": 24,
        "external_connections": [
            {"remote_ip": "185.10.68.55", "remote_port": 3333, "proc": "xmrig.exe"},
            {"remote_ip": "104.21.5.10",  "remote_port": 4444, "proc": "xmrig.exe"},
        ],
        "active_ports": [3333, 4444, 8333],
        "disk_read_mb": 0.1,
        "disk_write_mb": 0.1,
        "gpu": {
            "name": "RTX 3080",
            "load_percent": 98.0,
            "memory_used_mb": 9500,
            "memory_total_mb": 12000,
            "memory_percent": 79.2,
            "temperature": 82.0,
            "power_draw_w": 320.0,
        },
        "top_processes": [
            {"name": "xmrig.exe",  "cpu_percent": 5.0,  "memory_mb": 220},
            {"name": "nbminer.exe","cpu_percent": 8.0,  "memory_mb": 320},
        ],
        "loop_elapsed": 0.6,
        "local_alerts": [],
        "boxplot_signal": {"gpu_outlier": True, "outbound_outlier": True},
    }


def cpu_only_mining_payload(pc_id: str = "PC-CPU-MINING", offset: int = 0) -> dict:
    """low GPU + high CPU + stratum 외부 송신 → CPU_ONLY_MINING/STRATUM_PATTERN 기대."""
    return {
        "pc_id": pc_id,
        "timestamp": _ts(offset),
        "cpu_percent": 94.0,
        "cpu_core_count": 8,
        "memory_percent": 64.0,
        "memory_used_gb": 10.2,
        "memory_total_gb": 16.0,
        "inbound_mb": 0.1,
        "outbound_mb": 0.6,
        "inbound_total_mb": 20.5,
        "outbound_total_mb": 410.0,
        "external_packet_count": 480,
        "external_connection_count": 6,
        "external_connections": [
            {"remote_ip": "51.79.45.20", "remote_port": 5555, "proc": "minerd.exe"},
        ],
        "active_ports": [5555, 14444],
        "disk_read_mb": 0.2,
        "disk_write_mb": 0.1,
        "gpu": {
            "name": "RTX 3060",
            "load_percent": 4.0,
            "memory_used_mb": 250,
            "memory_total_mb": 12000,
            "memory_percent": 2.0,
            "temperature": 50.0,
        },
        "top_processes": [
            {"name": "minerd.exe", "cpu_percent": 88.0, "memory_mb": 180},
            {"name": "xmrig.exe",  "cpu_percent": 5.0,  "memory_mb": 140},
        ],
        "loop_elapsed": 0.7,
        "local_alerts": [],
        "boxplot_signal": {"cpu_outlier": True, "packet_outlier": True},
    }


def network_anomaly_payload(pc_id: str = "PC-NET-ANOM", offset: int = 0) -> dict:
    """저부하지만 송신·외부연결 ↑ → OUTBOUND_DOMINANT/MANY_EXTERNAL."""
    return {
        "pc_id": pc_id,
        "timestamp": _ts(offset),
        "cpu_percent": 10.0,
        "cpu_core_count": 8,
        "memory_percent": 38.0,
        "memory_used_gb": 6.1,
        "memory_total_gb": 16.0,
        "inbound_mb": 0.3,
        "outbound_mb": 12.0,
        "inbound_total_mb": 80.0,
        "outbound_total_mb": 2200.0,
        "external_packet_count": 1500,
        "external_connection_count": 80,
        "external_connections": [
            {"remote_ip": f"203.0.113.{i}", "remote_port": 443} for i in range(40)
        ],
        "active_ports": [443, 8080],
        "disk_read_mb": 0.5,
        "disk_write_mb": 0.3,
        "gpu": {
            "name": "RTX 3060",
            "load_percent": 6.0,
            "memory_used_mb": 300,
            "memory_total_mb": 12000,
            "memory_percent": 2.4,
            "temperature": 48.0,
        },
        "top_processes": [
            {"name": "rclone.exe", "cpu_percent": 9.0, "memory_mb": 220},
        ],
        "loop_elapsed": 0.5,
        "local_alerts": [],
        "boxplot_signal": {"outbound_outlier": True},
    }


SCENARIOS = {
    "normal":           normal_payload,
    "gpu_mining":       gpu_mining_payload,
    "cpu_only_mining":  cpu_only_mining_payload,
    "network_anomaly":  network_anomaly_payload,
}
