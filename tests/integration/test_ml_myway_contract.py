"""통합 계약 회귀 — 내 방식(core+models) ML 의 /analyze 응답이
팀원(jjaerud) Spring `MlResponse` + `AlertService` 계약을 만족하는지 검증.

ml_server 를 내 버전으로 교체(integ/ml-myway)한 뒤, Spring 측이 읽는 필드가
모두 존재하고 타입/의미가 맞는지 고정한다.
"""
import pytest
from fastapi.testclient import TestClient

from ml_server.main import app

client = TestClient(app)

_NORMAL = {
    "pc_id": "PC-NORMAL", "timestamp": "2026-05-30T14:00:00",
    "cpu_percent": 12.0, "memory_percent": 45.0,
    "inbound_mb": 0.5, "outbound_mb": 0.3, "external_packet_count": 30,
    "disk_read_mb": 0.1, "disk_write_mb": 0.1,
    "gpu": {"name": "RTX3070", "load_percent": 5.0, "memory_used_mb": 1200,
            "memory_total_mb": 8188, "memory_percent": 14.6, "power_draw_w": 18.0},
    "top_processes": [{"name": "chrome.exe", "cpu_percent": 5}],
}
_MINING = {
    "pc_id": "PC-MINER", "timestamp": "2026-05-30T14:00:00",
    "cpu_percent": 20.0, "memory_percent": 60.0,
    "inbound_mb": 0.5, "outbound_mb": 25.0, "external_packet_count": 1500,
    "disk_read_mb": 0.5, "disk_write_mb": 0.5,
    "gpu": {"name": "RTX3070", "load_percent": 96.0, "memory_used_mb": 7000,
            "memory_total_mb": 8188, "memory_percent": 85.5, "power_draw_w": 165.0},
    "top_processes": [{"name": "t-rex.exe", "cpu_percent": 10}],
}

# 팀원 MlResponse DTO + AlertService 가 읽는 계약 필드
_TOP_FIELDS = ["overall_severity", "verdict", "message", "scores", "alerts",
               "policy_version", "retrieval_evidence", "signals_missing",
               "category_signals", "evidence_meta", "local_evidence"]
_AGENT_FIELDS = ["judgment", "severity", "reason", "action", "hw_degradation"]
_EVMETA_FIELDS = ["active_signal_count", "category_count", "active_categories",
                  "active_signals", "promotion_gated", "promotion_reason", "fast_path_match"]


def _analyze(payload: dict) -> dict:
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    return r.json()


@pytest.mark.parametrize("payload", [_NORMAL, _MINING], ids=["normal", "mining"])
def test_top_level_contract_fields_present(payload):
    body = _analyze(payload)
    missing = [f for f in _TOP_FIELDS if f not in body]
    assert not missing, f"top-level 계약 필드 누락: {missing}"
    assert isinstance(body["scores"], dict)
    assert isinstance(body["alerts"], list)
    assert isinstance(body["evidence_meta"], dict)
    assert isinstance(body["local_evidence"], list)
    assert isinstance(body["message"], str) and body["message"]


@pytest.mark.parametrize("payload", [_NORMAL, _MINING], ids=["normal", "mining"])
def test_evidence_meta_keys(payload):
    body = _analyze(payload)
    missing = [f for f in _EVMETA_FIELDS if f not in body["evidence_meta"]]
    assert not missing, f"evidence_meta 키 누락: {missing}"


def test_normal_is_normal_severity():
    body = _analyze(_NORMAL)
    assert body["overall_severity"] == "NORMAL"
    assert body["verdict"] == "NORMAL"


def test_mining_escalates_and_agent_block_complete():
    body = _analyze(_MINING)
    assert body["overall_severity"] in ("MEDIUM", "HIGH")
    assert body["verdict"] in ("SUSPICIOUS", "HIGH_RISK")
    agent = body.get("agent")
    assert isinstance(agent, dict), "이상 시 agent 블록 필수"
    missing = [f for f in _AGENT_FIELDS if f not in agent]
    assert not missing, f"agent 계약 필드 누락: {missing}"
    assert agent["judgment"] and agent["severity"]
    # 채굴 프로세스 알려진 경우 fast_path 표기
    assert body["evidence_meta"]["fast_path_match"] == "mining_known"
