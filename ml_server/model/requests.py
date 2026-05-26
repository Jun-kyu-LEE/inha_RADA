"""요청 모델 (Spring Boot 페이로드 호환 — 키 변경 금지)."""
import datetime as _dt
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class GpuMetrics(BaseModel):
    name:               str
    load_percent:       float
    memory_used_mb:     float
    memory_total_mb:    float
    memory_percent:     float
    temperature:        Optional[float] = None
    sm_utilization:     Optional[float] = None
    tensor_core_active: Optional[int]   = None
    power_draw_w:       Optional[float] = None


class MetricsRequest(BaseModel):
    pc_id:                  str
    timestamp:              str
    cpu_percent:            float
    cpu_core_count:         int   = 1
    memory_percent:         float
    memory_used_gb:         float = 0
    memory_total_gb:        float = 0
    inbound_mb:             float
    outbound_mb:            float
    inbound_total_mb:       float = 0
    outbound_total_mb:      float = 0
    external_packet_count:  int
    external_connection_count: int = 0
    external_connections:   List[dict] = Field(default_factory=list)
    active_ports:           List[int]  = Field(default_factory=list)
    disk_read_mb:           float = 0
    disk_write_mb:          float = 0
    gpu:                    Optional[GpuMetrics] = None
    top_processes:          List[dict] = Field(default_factory=list)
    loop_elapsed:           float = 0.0
    local_alerts:           List[dict] = Field(default_factory=list)
    boxplot_signal:         dict       = Field(default_factory=dict)
    derived_features:       Optional[Dict[str, Any]] = None

    @field_validator("timestamp")
    @classmethod
    def _validate_iso(cls, v: str) -> str:
        try:
            _dt.datetime.fromisoformat(v)
        except (ValueError, TypeError):
            raise ValueError("timestamp must be ISO 8601 (e.g. 2026-05-13T12:34:56)")
        return v
