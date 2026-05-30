"""RADA E2E 통합 테스트 러너.

실행 (PowerShell):
  $env:PYTHONIOENCODING="utf-8"; python -m tests.e2e.run_e2e
실행 (bash):
  PYTHONIOENCODING=utf-8 python -m tests.e2e.run_e2e

전제: `docker compose up -d` 로 postgres / ml-server / spring-server / grafana 가
       모두 healthy. ML 서버에 사전학습된 모델이 없으면 rule-based 단독 모드로
       검증된다 (POST /pretrain/csv 로 모델 학습 후 재실행 권장).

환경변수:
  RADA_E2E_PEPPER     — Spring 의 API_KEY_PEPPER 와 동일해야 함 (기본: dev_pepper_change_me)
  RADA_E2E_API_KEY    — 평문 API key, pc_info.api_key 는 sha256(pepper:rawKey) 로 저장 (기본: e2e_test_apikey_001)
  RADA_E2E_PC_ID      — 등록할 테스트 PC id (기본: PC-E2E-001)
  RADA_E2E_TEARDOWN   — "1"이면 종료 시 테스트 PC 의 metrics/anomaly/judgment 행 삭제 (기본: 1)
  ML_URL / SPRING_URL / GRAFANA_URL — 엔드포인트 오버라이드
"""
from __future__ import annotations
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import urllib.error
import urllib.request

# Windows cp949 콘솔에서 em-dash 등 출력 시 UnicodeEncodeError 회피.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.e2e.scenarios import (  # noqa: E402
    normal_payload, gpu_mining_payload, cpu_only_mining_payload, network_anomaly_payload,
)

ML_URL     = os.environ.get("ML_URL",      "http://localhost:8000")
SPRING_URL = os.environ.get("SPRING_URL",  "http://localhost:8080")
GRAFANA    = os.environ.get("GRAFANA_URL", "http://localhost:3000")

TEST_PC_ID   = os.environ.get("RADA_E2E_PC_ID",  "PC-E2E-001")
TEST_RAW_KEY = os.environ.get("RADA_E2E_API_KEY", "e2e_test_apikey_001")
PEPPER       = os.environ.get("RADA_E2E_PEPPER",  "dev_pepper_change_me")
TEARDOWN     = os.environ.get("RADA_E2E_TEARDOWN", "1") == "1"

PEER_PC_IDS  = ["PC-E2E-PEER-A", "PC-E2E-PEER-B", "PC-E2E-PEER-C"]


def _http(method: str, url: str, body: dict | None = None, headers: dict | None = None,
          timeout: float = 10.0) -> tuple[int, dict | str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def _psql(query: str, mode: str = "-At") -> tuple[int, str, str]:
    p = subprocess.run(
        ["docker", "exec", "rada-postgres", "psql", "-U", "rada", "-d", "pc_monitor",
         mode, "-v", "ON_ERROR_STOP=1", "-c", query],
        capture_output=True, text=True, encoding="utf-8",
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.pass_count = 0
        self.fail_count = 0

    def step(self, name: str) -> None:
        self.lines.append(f"\n=== {name} ===")
        print(f"\n=== {name} ===")

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {label}"
        if detail:
            line += f"  | {detail}"
        self.lines.append(line)
        print(line)
        if ok:
            self.pass_count += 1
        else:
            self.fail_count += 1

    def kv(self, key: str, val: Any) -> None:
        s = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        line = f"    {key}: {s[:240]}"
        self.lines.append(line)
        print(line)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def health_checks(r: Report) -> None:
    r.step("Stage 1: Health checks")
    code, body = _http("GET", f"{ML_URL}/health")
    r.check("ML  /health 200", code == 200, f"code={code} body={body}")
    code, body = _http("GET", f"{ML_URL}/status")
    r.check("ML  /status running", code == 200 and isinstance(body, dict) and body.get("status") == "running",
            f"code={code}")
    code, body = _http("GET", f"{ML_URL}/pretrain/status")
    if code == 200 and isinstance(body, dict):
        mode = body.get("mode", "?")
        r.kv("ML inference mode", mode)
        if mode == "rule_based":
            r.kv("WARN", "saved_models 없음 — rule-based 단독 모드. ML+rule 하이브리드 검증을 원하면 POST /pretrain/csv 후 재실행.")
    code, _ = _http("POST", f"{SPRING_URL}/api/metrics", body={})
    r.check("Spring API 401 w/o key", code == 401, f"code={code}")
    code, _ = _http("GET", f"{GRAFANA}/api/health")
    r.check("Grafana /api/health 200", code == 200, f"code={code}")


def ml_direct_scenarios(r: Report) -> Dict[str, Any]:
    r.step("Stage 2: ML 서버 단독 /analyze 시나리오 4종")
    results: Dict[str, Any] = {}
    cases = [
        ("normal",          normal_payload()),
        ("gpu_mining",      gpu_mining_payload()),
        ("cpu_only_mining", cpu_only_mining_payload()),
        ("network_anomaly", network_anomaly_payload()),
    ]
    for name, payload in cases:
        code, body = _http("POST", f"{ML_URL}/analyze", body=payload, timeout=30)
        ok = code == 200 and isinstance(body, dict) and "verdict" in body
        r.check(f"{name}: HTTP 200 + verdict 존재", ok, f"code={code}")
        if not ok:
            r.kv("body", body)
            continue
        verdict = body["verdict"]
        severity = body.get("overall_severity")
        alerts = body.get("alerts") or []
        atypes = [a.get("type") for a in alerts]
        cat = body.get("category_signals") or {}
        scores = body.get("scores") or {}
        r.kv(f"{name} verdict",          f"{verdict} / sev={severity}")
        r.kv(f"{name} alert_types",      atypes)
        r.kv(f"{name} category_signals", cat)
        r.kv(f"{name} scores.final",     scores.get("final"))

        if name == "normal":
            r.check(f"{name} → verdict=NORMAL or OBSERVE", verdict in ("NORMAL", "OBSERVE"), f"got={verdict}")
        elif name == "gpu_mining":
            mining_types = {"GPU_MINING", "HIGH_GPU", "POOL_TRAFFIC",
                             "OUTBOUND_DOMINANT", "GPU_CPU_IMBALANCE"}
            hit = mining_types.intersection(atypes)
            r.check(f"{name} → mining/network alert 존재",
                    bool(hit) or verdict in ("SUSPICIOUS", "HIGH_RISK"),
                    f"alerts={atypes} verdict={verdict}")
            r.check(f"{name} → category.resource OR network abnormal",
                    bool(cat.get("resource_abnormal") or cat.get("network_abnormal")))
        elif name == "cpu_only_mining":
            cpu_types = {"CPU_ONLY_MINING", "HIGH_CPU", "STRATUM_PATTERN",
                          "CPU_GPU_IMBALANCE", "POOL_TRAFFIC"}
            hit = cpu_types.intersection(atypes)
            r.check(f"{name} → cpu/network alert 존재",
                    bool(hit) or verdict in ("SUSPICIOUS", "HIGH_RISK"),
                    f"alerts={atypes} verdict={verdict}")
        elif name == "network_anomaly":
            net_types = {"OUTBOUND_DOMINANT", "HIGH_OUTBOUND", "MANY_EXTERNAL", "STRATUM_PATTERN"}
            hit = net_types.intersection(atypes)
            r.check(f"{name} → network alert 존재 or OBSERVE 이상",
                    bool(hit) or verdict != "NORMAL",
                    f"alerts={atypes} verdict={verdict}")
        results[name] = body
    return results


def warmup_retrieval_pool(r: Report) -> None:
    """retrieval_evidence 가 활성화되려면 peer segment 가 store 에 쌓여 있어야 한다.
    여러 peer PC 의 NORMAL 스냅샷을 ML 에 흘려보내 segment pool 을 만든다."""
    r.step("Stage 2b: retrieval pool warm-up (peer segments)")
    count = 0
    for pc in PEER_PC_IDS:
        for offset in range(0, 120, 10):  # 12 snapshots × 10s = 1 segment
            code, _ = _http("POST", f"{ML_URL}/analyze",
                            body=normal_payload(pc_id=pc, offset=offset), timeout=10)
            if code == 200:
                count += 1
    r.kv("peer snapshots ingested", f"{count} / {len(PEER_PC_IDS) * 12}")
    r.check("peer warmup ≥ peer_count×12", count >= len(PEER_PC_IDS) * 12)


def register_test_pc(r: Report) -> bool:
    r.step("Stage 3a: 테스트 PC 등록 (pc_info + 해시된 API key)")
    sql = (
        "INSERT INTO pc_monitor.pc_info (pc_id, hostname, api_key, is_active, registered_at) "
        f"VALUES ('{TEST_PC_ID}', 'e2e-host', "
        f"encode(digest('{PEPPER}:{TEST_RAW_KEY}', 'sha256'), 'hex'), true, now()) "
        "ON CONFLICT (pc_id) DO UPDATE SET "
        f"  api_key = encode(digest('{PEPPER}:{TEST_RAW_KEY}', 'sha256'), 'hex'), "
        "  is_active = true;"
    )
    rc, _, err = _psql(sql)
    ok = rc == 0
    r.check(f"pc_info upsert {TEST_PC_ID}", ok, f"rc={rc} err={err[:160]}")
    return ok


def spring_ingest_scenarios(r: Report) -> Dict[str, Any]:
    r.step("Stage 3b: Spring /api/metrics → ML → DB 적재 시나리오")
    headers = {"X-API-Key": TEST_RAW_KEY}
    cases = [
        ("spring_normal",     normal_payload(pc_id=TEST_PC_ID, offset=0)),
        ("spring_gpu_mining", gpu_mining_payload(pc_id=TEST_PC_ID, offset=60)),
        ("spring_cpu_mining", cpu_only_mining_payload(pc_id=TEST_PC_ID, offset=120)),
    ]
    ids: Dict[str, Any] = {}
    for name, payload in cases:
        payload = dict(payload)
        if not payload["timestamp"].endswith(("Z", "+00:00")):
            payload["timestamp"] = payload["timestamp"] + "+09:00"
        code, body = _http("POST", f"{SPRING_URL}/api/metrics", body=payload, headers=headers, timeout=15)
        ok = code == 202 and isinstance(body, dict) and body.get("status") == "accepted"
        r.check(f"{name}: 202 accepted", ok, f"code={code} body={body}")
        if ok:
            ids[name] = body.get("id")
    return ids


def verify_db_state(r: Report) -> None:
    r.step("Stage 4: DB 상태 검증")
    time.sleep(3)  # ML forward 가 @Async — 적재 완료 대기.

    def psql(q: str) -> str:
        rc, out, _ = _psql(q)
        return out if rc == 0 else ""

    metrics_count = psql(f"SELECT COUNT(*) FROM pc_monitor.metrics_history WHERE pc_id='{TEST_PC_ID}'")
    anomaly_count = psql(f"SELECT COUNT(*) FROM pc_monitor.anomaly_history WHERE pc_id='{TEST_PC_ID}'")
    judg_count    = psql(f"SELECT COUNT(*) FROM pc_monitor.ai_judgment_history WHERE pc_id='{TEST_PC_ID}'")
    r.kv("metrics_history rows", metrics_count)
    r.kv("anomaly_history rows", anomaly_count)
    r.kv("ai_judgment_history rows", judg_count)
    r.check("metrics_history ≥ 3", metrics_count.isdigit() and int(metrics_count) >= 3)
    r.check("anomaly_history ≥ 1 (HIGH/SUSPICIOUS 시나리오 1건+)",
            anomaly_count.isdigit() and int(anomaly_count) >= 1)

    sample = psql(
        "SELECT severity || ' / ' || COALESCE(anomaly_type,'') || ' | final=' || "
        "       COALESCE(scores->>'final','-') || ' | cat=' || COALESCE((scores->'category_signals')::text,'-') "
        f"FROM pc_monitor.anomaly_history WHERE pc_id='{TEST_PC_ID}' ORDER BY detected_at DESC LIMIT 1"
    )
    r.kv("anomaly_history latest", sample)
    has_keys = psql(
        "SELECT (scores ? 'category_signals')::text || ',' || "
        "       (scores ? 'final')::text || ',' || "
        "       (scores ? 'retrieval_evidence')::text "
        f"FROM pc_monitor.anomaly_history WHERE pc_id='{TEST_PC_ID}' ORDER BY detected_at DESC LIMIT 1"
    )
    cat_ok, final_ok, retr_ok = (has_keys.split(",") + ["", "", ""])[:3]
    r.kv("scores JSONB 키 존재 (cat/final/retr)", has_keys)
    r.check("scores.category_signals 키 존재", cat_ok in ("t", "true"))
    r.check("scores.final 키 존재",            final_ok in ("t", "true"))
    # retrieval_evidence 는 warmup 이 잘 됐는지에 따라 달라짐 — 정보성 체크.
    if retr_ok in ("t", "true"):
        r.check("scores.retrieval_evidence 키 존재 (warmup 성공)", True)
    else:
        r.kv("INFO", "retrieval_evidence 키 없음 — peer warmup 부족 또는 segment store 미적재")


def grafana_datasource_check(r: Report) -> None:
    r.step("Stage 5: Grafana 데이터소스/대시보드 점검")
    auth_hdr = {"Authorization": "Basic YWRtaW46YWRtaW4="}  # admin/admin
    code, body = _http("GET", f"{GRAFANA}/api/datasources", headers=auth_hdr)
    r.check("GET /api/datasources 200", code == 200, f"code={code}")
    if code != 200 or not isinstance(body, list):
        r.kv("body", body)
        return
    pg = [d for d in body if "postgres" in (d.get("type", "") or "").lower()]
    r.check("Postgres datasource 존재", bool(pg), f"types={[d.get('type') for d in body]}")
    if pg:
        ds_uid = pg[0].get("uid") or pg[0].get("id")
        r.kv("postgres datasource", f"name={pg[0].get('name')} uid={ds_uid}")
        if pg[0].get("uid"):
            code2, body2 = _http("GET", f"{GRAFANA}/api/datasources/uid/{pg[0]['uid']}/health",
                                  headers=auth_hdr, timeout=15)
            r.check("Postgres datasource health OK",
                    code2 == 200 and isinstance(body2, dict) and body2.get("status") == "OK",
                    f"code={code2} body={body2}")
    code, body = _http("GET", f"{GRAFANA}/api/search?type=dash-db", headers=auth_hdr)
    if isinstance(body, list):
        r.kv("provisioned dashboards", [d.get("title") for d in body])
        r.check("최소 1개 대시보드 프로비저닝", len(body) >= 1)


def teardown(r: Report) -> None:
    if not TEARDOWN:
        r.step("Teardown skipped (RADA_E2E_TEARDOWN=0)")
        return
    r.step("Teardown: 테스트 PC/피어 PC 의 적재 행 정리")
    all_pcs = [TEST_PC_ID, *PEER_PC_IDS]
    in_list = "(" + ",".join(f"'{p}'" for p in all_pcs) + ")"
    queries = [
        f"DELETE FROM pc_monitor.ai_judgment_history WHERE pc_id IN {in_list}",
        f"DELETE FROM pc_monitor.anomaly_history     WHERE pc_id IN {in_list}",
        f"DELETE FROM pc_monitor.metrics_history     WHERE pc_id IN {in_list}",
        f"DELETE FROM pc_monitor.pc_info             WHERE pc_id IN {in_list}",
    ]
    total_deleted = 0
    for q in queries:
        rc, out, err = _psql(q + " RETURNING 1;")
        if rc == 0:
            total_deleted += len(out.splitlines())
        else:
            r.kv("teardown error", f"{q} -> {err[:120]}")
    r.kv("deleted rows", total_deleted)


def main() -> int:
    print(f"ML={ML_URL}  SPRING={SPRING_URL}  GRAFANA={GRAFANA}")
    print(f"PC={TEST_PC_ID}  teardown={TEARDOWN}")
    r = Report()
    try:
        health_checks(r)
        ml_direct_scenarios(r)
        warmup_retrieval_pool(r)
        pc_ok = register_test_pc(r)
        if pc_ok:
            spring_ingest_scenarios(r)
            verify_db_state(r)
        grafana_datasource_check(r)
    finally:
        teardown(r)

    r.lines.append("")
    summary = f"\n결과: PASS={r.pass_count}  FAIL={r.fail_count}"
    r.lines.append(summary)
    print(summary)

    out = ROOT / "tests" / "e2e_report.md"
    r.write(out)
    print(f"\n리포트 저장: {out}")
    return 0 if r.fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
