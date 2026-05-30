# Shadow 모델 비교 — IF/LOF vs ECOD/HBOS/KNN

> 작성일: 2026-05-30
> 대상 브랜치: `integ/ml-myway` (팀원 base + core+models ML)
> 하니스: `ml_server/eval/shadow_model_compare.py`
> 데이터: `phase1_features.csv` (로컬 수집 정상 14,942행) + 합성 채굴 injection

## 1. 배경 / 목적

피드백 반영: **IF/LOF 는 최종 판단 경로 유지**, ECOD/HBOS/KNN 은 최종 verdict 에
반영하지 않고 **점수·탐지만 따로 기록(shadow)** 하여 기존 IF/LOF 와 비교한다.
성능이 괜찮으면 도입을 검토한다. 본 평가는 `rule_based`/`ensemble` verdict 로직을
전혀 건드리지 않는 **순수 오프라인 비교**다.

핵심 관심사(피드백):
1. 로컬 PC 수집 정상 데이터를 이상으로 **오탐(FP)** 하지 않는지? 오탐률은?
   특히 **딥러닝 학습 환경(GPU 풀로드 정상)** 을 채굴로 오탐하는지.
2. 채굴 데이터를 **injection 했을 때 잘 탐지**하는지?

## 2. 방법

학습 파이프라인과 동일하게 처리(공정 비교):

1. `parse_csv` → 추가/미사용 컬럼 제거, `extract_features` → **원래 13 feature**
   (enabled_raw 7 + derived 6). CSV 의 미리계산 derived 는 무시하고 raw 에서 재계산.
2. 정상 데이터를 시나리오 stratified 로 train 70% / test 30% 분할.
3. `RobustScaler` fit(train) → transform, `apply_weights(free)` 적용.
4. 각 모델 fit(train). **threshold 를 train 점수의 contamination(기본 5%) 분위로 보정**
   (점수는 낮을수록 이상). → 잘 일반화하면 test 정상 FP ≈ 5% 가 기대값.
5. test 정상으로 FP(시나리오별 포함), 합성 채굴로 탐지율 측정.

> 모델 파라미터는 `config/models.yaml` 과 동일:
> IF(contamination=0.03, n=200) · LOF(k=20, novelty) · ECOD · HBOS(bins=20) ·
> KNN(k=20) · Mahalanobis(ridge=1e-4).

### 채굴 injection (합성)
- **GPU 채굴**(t-rex 류): gpu 88–99%, power 150–175W, vram 7GB, outbound 15–40MB, 패킷 多.
- **CPU 채굴**(xmrig/Monero): cpu 88–99%, gpu 유휴, outbound 10–25MB, 패킷 多.
- 타입별 300 샘플.

## 3. 결과 (contamination=5%, seed=42)

정상 데이터 분포: office 7,040 · dl_learning 7,013 · idle(`idel`) 889.
train=10,459 / test(정상)=4,483 / mining=600.

### 3-A. 개별 모델 — baseline(IF/LOF) vs shadow(ECOD/HBOS/KNN)

| 모델 | 그룹 | FP 전체 | **dl_learning** | idle | office | 탐지 GPU | 탐지 CPU |
|---|---|---|---|---|---|---|---|
| IsolationForest | baseline | 4.3% | 7.9% | 0.0% | 1.3% | 100% | 100% |
| **LOF** | baseline | 3.1% | **2.8%** | 7.1% | 2.8% | 99.7% | **63.0%** ⚠️ |
| ECOD | shadow | 4.2% | 6.7% | **9.4%** | 1.1% | 100% | 100% |
| HBOS | shadow | 4.3% | 7.4% | 0.0% | 1.8% | 100% | 100% |
| KNN | shadow | 4.5% | 7.5% | 0.0% | 2.1% | 100% | 100% |

### 3-B. 앙상블 변형 — 현재(IF+ECOD+Maha)에서 ECOD 자리 교체 (weight 0.4/0.3/0.3)

| 앙상블 | FP 전체 | dl_learning | idle | office | 탐지 GPU | 탐지 CPU |
|---|---|---|---|---|---|---|
| **IF+ECOD+Maha (현재)** | 4.1% | 7.4% | 0.4% | 1.2% | 100% | 100% |
| IF+HBOS+Maha | 4.3% | 7.7% | 0.0% | 1.5% | 100% | 100% |
| IF+KNN+Maha | 4.4% | 7.7% | 0.0% | 1.7% | 100% | 100% |

## 4. 핵심 발견

1. **LOF 는 CPU 채굴을 37% 놓침**(탐지 63%). 단독 baseline 판정자로는 CPU(Monero)
   채굴에 취약 → 레포가 LOF 를 비활성화한 결정을 데이터로 뒷받침.
2. **shadow 3종 모두 GPU·CPU 채굴 100% 탐지** + FP 는 IF 와 비슷. IF 를 결정적으로
   이기지 못함(HBOS·KNN ≈ IF). **ECOD 만 개별로는 idle FP 9.4%** 로 튐.
3. **앙상블에서는 ECOD 의 idle 약점이 IF/Maha 에 의해 상쇄**(개별 9.4% → 앙상블 0.4%).
   ECOD→HBOS/KNN 교체는 idle 을 0% 로 낮추지만 dl/office 가 미세하게 올라
   **사실상 wash** — 교체 이득 없음.
4. **dl_learning(GPU 풀로드 정상)이 모든 모델의 최대 오탐원**(7~8%). 단, threshold 를
   5% 로 고정한 상대 비교이며, 실제 production 은 `rule_based` 게이팅이 더 얹혀 FP↓.

## 5. 결론 / 권장

- **현재 IF+ECOD+Maha 유지 권장.** 앙상블 단위에서 이미 GPU·CPU 채굴 100% 탐지 +
  idle 0.4% 로 우수하며, ECOD→HBOS/KNN 교체로 얻는 실익이 없다.
- **LOF 를 최종 판단에 단독 투입하지 말 것** (CPU 채굴 miss).
- shadow 3종은 "도입해도 손해는 없지만 결정적 이득도 없는" 수준 → 굳이 활성 앙상블을
  바꿀 근거는 약함. 필요 시 HBOS/KNN 을 idle-민감 환경의 ECOD 대체 후보로만 보관.

## 6. 한계 / 후속

- 채굴이 **합성 injection** 이라 정상 분포와 멀어 탐지율이 낙관적(100%)일 수 있음.
  **실제 격리 채굴 데이터**(behavior-only/stealth 포함)로 재검증 필요.
- threshold 5% 고정 비교 — 절대 FP 보다 **시나리오별 패턴·탐지율 차이**가 해석 핵심.
- feature-space 만 평가 — 실제 verdict 는 rule 게이팅·sustained·retrieval 이 추가됨.
- `idel` 시나리오 라벨 오타(=idle, 889행). 본 하니스는 라벨 문자열 그대로 집계.

## 7. 재현

```bash
# ddr_project 환경, repo 루트에서
python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv
python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv --contamination 0.03
python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv --mining-n 500
```
