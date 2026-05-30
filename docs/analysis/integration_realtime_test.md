# 통합 · 실시간 이상탐지 테스트 리포트 (integ/ml-myway)

> 작성일: 2026-05-30
> 브랜치: `integ/ml-myway` (`Jun-kyu-LEE/inha_RADA`)
> 구성: 팀원 `wip/phase1-2-deploy` base + `ml_server/` 만 core+models 하이브리드로 교체

## 1. 무엇을 했나
- ML 서버를 core+models 하이브리드(rule_based + IF/ECOD/Maha 앙상블)로 **교체**, 나머지 팀 스택(Spring/DB/Grafana/client)은 팀원 것 유지.
- `/analyze` 응답에 `message`/`evidence_meta`/`local_evidence` 추가 → 팀원 `MlResponse`/`AlertService`/Grafana **계약 호환**.
- 빌드 인프라 수정: `server-spring/Dockerfile` 에 gradlew CRLF strip (Windows 체크아웃에서 `./gradlew: not found` 빌드 실패 해결).

## 2. 어떻게 테스트했나
- **풀스택 e2e**: `docker compose up -d --build` → `python -m tests.e2e.run_e2e`.
- **실시간**: 실제 테스트 PC 지표를 5초 간격 스트리밍(client) + 채굴 패턴 주입.
- **ML 앙상블**: `POST /pretrain/csv` (phase1) 후 라이브 추론, ML 점수 실시간 관찰.

## 3. 결과
- ✅ **풀스택 e2e 26 PASS / 0 FAIL** — Grafana datasource(OK) + 대시보드 3종 + 실데이터 조회 확인.
- ✅ **실데이터 계약**: `anomaly_history.scores` JSONB 에 `category_signals`/`evidence_meta` 저장, `ai_judgment_history` 에 에이전트 판정 저장. 팀원 Spring-side FP 필터(P0-1 NORMAL skip, P1-3 60s 쿨다운) 그대로 작동.
- ✅ **실시간 채굴 탐지**: 정상 구간 NORMAL → 채굴 주입 시 즉시 **HIGH_RISK** (`CPU_ONLY_MINING`/`CONFIRMED_MINING`/`POOL_TRAFFIC` …), DB·Grafana 반영. ML ensemble 점수가 정상 -0.1대 → 채굴 -79~-158 로 실시간 급락.
  - 참고: 채굴 직후 몇 사이클은 SUSPICIOUS 유지 — `ensemble.py` 의 PC별 슬라이딩 윈도우(최근 5건 가중평균) 때문(지속 공격 강조 / 단발 스파이크 완화 설계). ~5사이클 후 NORMAL 회복.

## 4. ⚠️ 핵심 발견 — OOD(분포 밖) 오탐과 GPU 미탐지
phase1(실습실 PC, GPU 탑재)만 학습한 모델이 **GPU 없는 노트북을 SUSPICIOUS 오탐**.

### Ablation — GPU 유무만 바꿔 비교 (phase1-only 모델, 동일 스냅샷 cpu20/mem78)
| | GPU 미탐지(노트북) | GPU 탐지(15%) |
|---|---|---|
| ensemble | **-0.488** (임계 -0.5 턱걸이) | **-0.069** (여유 정상) |
| ECOD | **-1.134** | -0.306 |
| IF | -0.057 | +0.077 |
| 극단(OOD) feature 수 | **5** | 1 |

GPU=0 일 때 극단값이 되는 feature(전부 GPU 파생): `gpu_percent`/`gpu_vram_mb`/`gpu_power_w`/`gpu_vram_complex` (모두 phase1 0.0%ile), **`cpu_gpu_ratio`=200 (99.9%ile)** — `cpu/(gpu+0.001)` 클립 때문. GPU 를 잡으면 `cpu_gpu_ratio` 200→1.33, ECOD -1.13→-0.31 로 정상화.

> 단독 원인은 아님: 메모리 96%(96.8%ile)도 함께 기여. **"GPU=0(파생 5개 극단) + 높은 메모리"의 복합**.

### 교정 — 해당 PC 정상 baseline 을 학습에 포함
phase1 + 노트북 정상 데이터(12%)로 재학습 → **동일 실시간 데이터가 NORMAL 로 판정** (ensemble -0.58→-0.12, ECOD -1.33→-0.49, IF 음수→양수).

**결론: FP 는 모델 결함이 아니라 학습 분포가 배포 PC 를 대표하느냐에 의존.** 실전 배포 시 각 PC baseline 포함 / 온라인 재학습 필요. (특히 GPU 없는 기기가 fleet 에 섞이면 GPU 파생 feature 가 구조적 OOD 가 됨 → 그런 기기군 데이터로 학습 보강 필요.)

## 5. shadow 모델 비교 (`docs/analysis/shadow_model_comparison.md`)
LOF 는 CPU 채굴 37% 놓침(단독 부적합), ECOD/HBOS/KNN 은 IF 와 비슷. 앙상블이 ECOD idle 약점 상쇄 → **현재 IF+ECOD+Maha 유지 권장**.

## 6. ⚠️ 논의 필요
이 교체는 팀원이 46커밋에 걸쳐 튜닝한 ML-side 로직(`scorer/`+`policy/`: risk_vector, signal_quality, promotion gating 등)을 rule_based+ensemble 로 **대체**함. Spring-side FP 필터는 유지되나 ML 내부 튜닝은 빠짐 → 판정 로직 방향 결정 필요.

## 7. 재현
```bash
git fetch && git checkout integ/ml-myway
docker compose up -d --build              # 4서비스 healthy 대기
python -m tests.e2e.run_e2e               # 풀스택 e2e
# (선택) ML 앙상블: curl -F file=@phase1_features.csv localhost:8000/pretrain/csv
python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv
# Grafana: http://localhost:3000 (admin/admin)
```
