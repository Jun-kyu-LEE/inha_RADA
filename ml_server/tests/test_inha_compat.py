"""inha_RADA /analyze 응답 호환 회귀 (ml_server 내부).

inha tests/integration/test_response_schema.py 와 동일 계약 검증.
"""
from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient

from ml_server.main import app
from ml_server.storage import pc_history_store, model_store, score_history_store
from ml_server.core.slot_classifier import classify_slot

pytestmark = pytest.mark.unit

REQUIRED_TOP_KEYS = {
    "pc_id", "timestamp", "timetable_slot",
    "overall_severity", "verdict",
    "alerts", "scores", "signals",
    "history_size", "isolation_forest",
    "global_hw_degradation", "agent",
}

REQUIRED_IF_KEYS = {
    "if_score", "lof_score", "weighted_score",
    "is_anomaly", "if_anomaly", "lof_anomaly",
    "boxplot_flag", "sample_count", "lof_window_size",
    "contamination", "boxplot_filtered",
}

SCORE_BREAKDOWN_KEYS = {
    "resource", "network", "process", "episode",
    "correlation", "ml", "context_discount", "final",
}

SCORE_BREAKDOWN_WITH_RETRIEVAL = SCORE_BREAKDOWN_KEYS | {"retrieval"}

FORBIDDEN_PHRASES = ["확실시", "EDR", "오탐 제거"]

VERDICT_VALS = {"NORMAL", "OBSERVE", "SUSPICIOUS", "HIGH_RISK"}
SEVERITY_VALS = {"NORMAL", "LOW", "MEDIUM", "HIGH"}
HW_VALS = {"NONE", "SUSPECTED", "CONFIRMED"}


@pytest.fixture(autouse=True)
def _reset_stores():
    pc_history_store.pc_history.clear()
    pc_history_store.pc_train_history.clear()
    pc_history_store.all_pc_latest.clear()
    model_store.pc_models.clear()
    score_history_store.pc_score_history.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _payload(**overrides):
    base = {
        "pc_id": "pc-test",
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_percent": 96.0,
        "memory_percent": 80.0,
        "inbound_mb": 0.5,
        "outbound_mb": 12.0,
        "external_packet_count": 1200,
        "disk_read_mb": 1.0,
        "disk_write_mb": 1.0,
        "gpu": {
            "name": "GPU",
            "load_percent": 95.0,
            "memory_used_mb": 4000.0,
            "memory_total_mb": 8192.0,
            "memory_percent": 50.0,
            "power_draw_w": 150.0,
        },
        "top_processes": [
            {"name": "xmrig.exe", "cpu_percent": 90.0, "path": r"C:\xmrig.exe"},
        ],
    }
    base.update(overrides)
    return base


def _normal_payload(**overrides):
    base = {
        "cpu_percent": 15.0,
        "memory_percent": 40.0,
        "outbound_mb": 0.2,
        "external_packet_count": 5,
        "gpu": {
            "name": "G",
            "load_percent": 5.0,
            "memory_used_mb": 200.0,
            "memory_total_mb": 8192.0,
            "memory_percent": 5.0,
        },
        "top_processes": [],
    }
    base.update(overrides)
    return _payload(**base)


def test_classify_slot_only_class_or_free():
    sat = datetime.datetime(2026, 5, 16, 12, 0, 0)  # Saturday
    assert classify_slot(sat) == "free"
    wed_class = datetime.datetime(2026, 5, 14, 10, 0, 0)
    assert classify_slot(wed_class) == "class"


def test_analyze_response_schema(client):
    r = client.post("/analyze", json=_payload())
    assert r.status_code == 200
    body = r.json()
    missing = REQUIRED_TOP_KEYS - set(body.keys())
    assert not missing, f"missing top keys: {missing}"
    iso = body["isolation_forest"]
    for k in ("sample_count", "weighted_score", "is_anomaly", "if_score", "lof_score"):
        assert k in iso, f"missing key: {k}"


def test_score_breakdown_eight_keys_and_retrieval(client):
    r = client.post("/analyze", json=_payload())
    body = r.json()
    sb = body["scores"]["score_breakdown"]
    assert SCORE_BREAKDOWN_KEYS.issubset(set(sb.keys()))
    assert "retrieval" in sb
    assert SCORE_BREAKDOWN_WITH_RETRIEVAL.issubset(set(sb.keys()))
    assert sb["final"] == body["scores"]["final"]


def test_insufficient_ml_keys(client):
    """pretrain 미완료 시 isolation_forest.available=False + Spring 호환 키."""
    r = client.post("/analyze", json=_normal_payload(pc_id="pc-cold"))
    body = r.json()
    iso = body["isolation_forest"]
    assert iso["available"] is False
    for k in ("sample_count", "weighted_score", "is_anomaly", "if_score", "lof_score"):
        assert k in iso, f"missing key on insufficient: {k}"


def test_verdict_and_severity_enums(client):
    r = client.post("/analyze", json=_payload())
    body = r.json()
    assert body["verdict"] in VERDICT_VALS
    assert body["overall_severity"] in SEVERITY_VALS
    if body["agent"]:
        assert body["agent"]["hw_degradation"] in HW_VALS


def test_forbidden_expressions(client):
    r = client.post("/analyze", json=_payload())
    body = r.json()
    blobs = []
    for a in body["alerts"]:
        blobs.append(str(a.get("detail", "")))
        blobs.append(str(a.get("type", "")))
    if body["agent"]:
        blobs.append(str(body["agent"].get("reason", "")))
        blobs.append(str(body["agent"].get("action", "")))
    joined = " ".join(blobs)
    for bad in FORBIDDEN_PHRASES:
        assert bad not in joined, f"forbidden phrase '{bad}' in: {joined}"


def test_confirmed_mining_alert(client):
    r = client.post("/analyze", json=_payload())
    body = r.json()
    types = [a["type"] for a in body["alerts"]]
    assert "CONFIRMED_MINING" in types
    assert body["overall_severity"] in ("HIGH", "MEDIUM", "LOW")


def test_observe_severity_low(client):
    """낮은 위험도는 OBSERVE + LOW."""
    r = client.post("/analyze", json=_normal_payload(
        cpu_percent=50.0,
        outbound_mb=0.1,
    ))
    body = r.json()
    if body["verdict"] == "OBSERVE":
        assert body["overall_severity"] == "LOW"
