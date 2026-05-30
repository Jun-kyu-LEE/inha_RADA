# RADA End-to-End 통합 테스트

`tests/e2e/run_e2e.py` 는 RADA 전체 스택(client → Spring → ML → Postgres + Grafana)
의 핵심 경로를 한 번에 검증한다.

## 사전 조건

1. **Docker Desktop 기동.**
2. 전체 스택 기동 (프로젝트 루트에서):
   ```powershell
   docker compose up -d --build
   ```
   `docker ps` 로 `rada-postgres` / `rada-ml` / `rada-spring` 이 모두 `healthy`,
   `rada-grafana` 가 `Up` 인지 확인.

3. (선택) **ML 사전학습**. 컨테이너에는 `saved_models/` 볼륨이 없어 첫 기동 시
   ML 서버는 **rule-based 단독 모드**로 동작한다. ML+rule 하이브리드 경로까지
   검증하고 싶다면 학습 CSV 를 준비 후:
   ```powershell
   curl -F "file=@<path/to/train.csv>" http://localhost:8000/pretrain/csv
   ```
   `GET /pretrain/status` 의 `mode` 가 `ml` 로 바뀐 뒤 러너 재실행.

## 실행

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m tests.e2e.run_e2e
```

PowerShell 외 환경:
```bash
PYTHONIOENCODING=utf-8 python -m tests.e2e.run_e2e
```

종료 코드: 모든 단계 PASS 면 `0`, 한 건이라도 FAIL 이면 `1`.
리포트는 `tests/e2e_report.md` 에 누적 없이 매번 새로 작성된다.

## 환경변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RADA_E2E_PEPPER`   | `dev_pepper_change_me` | Spring `API_KEY_PEPPER` 와 일치해야 함 |
| `RADA_E2E_API_KEY`  | `e2e_test_apikey_001`  | 평문 API key (`X-API-Key`). DB 에는 sha256(pepper:rawKey) 가 저장됨 |
| `RADA_E2E_PC_ID`    | `PC-E2E-001`           | 등록 후 사용할 테스트 PC id |
| `RADA_E2E_TEARDOWN` | `1`                    | `0` 으로 끄면 종료 후에도 DB 행을 남김 (디버깅용) |
| `ML_URL`            | `http://localhost:8000`| ML 서버 base URL |
| `SPRING_URL`        | `http://localhost:8080`| Spring API base URL |
| `GRAFANA_URL`       | `http://localhost:3000`| Grafana base URL |

## 검증 단계

| Stage | 내용 |
| --- | --- |
| 1 | ML `/health` + `/status` + `/pretrain/status` (mode 표시), Spring 401 인증 게이트, Grafana `/api/health` |
| 2 | ML `/analyze` 직접 호출 — `normal` / `gpu_mining` / `cpu_only_mining` / `network_anomaly` 4 시나리오의 verdict·alerts·category_signals 검증 |
| 2b | retrieval pool warm-up — peer PC 3 대의 NORMAL 스냅샷을 흘려보내 segment store 충전 |
| 3a | `pc_info` 에 테스트 PC + 해시된 API key UPSERT |
| 3b | Spring `/api/metrics` 로 3 시나리오를 인증 헤더 포함 전송 → 202 accepted 확인 |
| 4  | `metrics_history` / `anomaly_history` / `ai_judgment_history` 행 수 + `scores` JSONB 키 (`category_signals` / `final` / `retrieval_evidence`) 검증 |
| 5  | Grafana Postgres 데이터소스 존재 + health, 프로비저닝된 대시보드 목록 |
| Teardown | 테스트/피어 PC 의 4 개 테이블 행 + `pc_info` 행 제거 |

## 흔한 실패와 원인

- **Spring → ML 422 `inbound_total_mb/outbound_total_mb`** — 클라이언트 페이로드가
  두 필드를 누락하면 Spring DTO 가 `null` 로 직렬화하고 ML Pydantic 이 거부.
  Spring 서버를 거치는 경로에서는 항상 두 필드 포함 필요.
- **DB 행 수 0** — Spring 의 `MlForwardService` 는 `@Async`. 검증 전 sleep 3s 가
  내장돼 있지만 느린 머신이면 늘려야 할 수도 있다.
- **Postgres datasource health FAIL** — `grafana` 컨테이너가 `rada_pg` uid 의
  데이터소스를 프로비저닝하지 못한 경우. `infra/grafana/provisioning-docker/datasources` 점검.
- **`retrieval_evidence` 키 없음** — segment store 가 비어 있는 cold start 시 정상.
  Stage 2b warmup 으로 보강했지만, 그래도 비어 있으면 ML 컨테이너 재기동을 의심.
