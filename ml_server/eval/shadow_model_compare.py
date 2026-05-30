"""Shadow 모델 비교 — baseline(IF/LOF) vs shadow(ECOD/HBOS/KNN).

피드백 반영: IF/LOF 는 최종 판단 유지, ECOD/HBOS/KNN 은 점수/탐지만 따로 기록해
기존 IF/LOF 와 비교한다. 최종 verdict 로직(rule_based/ensemble)은 건드리지 않는
순수 오프라인 평가다.

평가 방법
---------
1. 정상 데이터(phase1_features.csv: office/dl_learning/idle)를 학습 파이프라인과
   동일하게 처리: parse_csv → extract_features(원래 13 feature) → RobustScaler →
   apply_weights(free).
2. 정상을 train/test 로 분할(시나리오 stratified). 각 모델을 train 으로 fit.
3. 각 모델 threshold 를 train 점수의 contamination 분위로 보정(낮을수록 이상).
4. test 정상으로 **오탐률(FP)** — 시나리오별 포함 — 측정.
5. 합성 채굴 injection(GPU 채굴 / CPU 채굴)으로 **탐지율** 측정.

실행:
  python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv
  python -m ml_server.eval.shadow_model_compare --csv phase1_features.csv --contamination 0.05
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.csv_parser import parse_csv
from ..feature.training_adapter import extract_features, FeatureScaler, apply_weights
from ..models.isolation_forest import IsolationForestModel
from ..models.lof import LOFModel
from ..models.ecod import ECODModel
from ..models.hbos import HBOSModel
from ..models.knn import KNNModel
from ..models.mahalanobis import MahalanobisModel

BASELINE = ["IsolationForest", "LOF"]
SHADOW = ["ECOD", "HBOS", "KNN"]

# 앙상블 변형 — 현재 활성(IF+ECOD+Maha)의 ECOD 자리를 HBOS/KNN 으로 교체 비교.
# weight 는 models.yaml 과 동일(0.4/0.3/0.3).
ENSEMBLES = {
    "IF+ECOD+Maha (현재)": [("IsolationForest", 0.4), ("ECOD", 0.3), ("Mahalanobis", 0.3)],
    "IF+HBOS+Maha":        [("IsolationForest", 0.4), ("HBOS", 0.3), ("Mahalanobis", 0.3)],
    "IF+KNN+Maha":         [("IsolationForest", 0.4), ("KNN",  0.3), ("Mahalanobis", 0.3)],
}


def _make_models() -> dict:
    """models.yaml 과 동일 파라미터로 모델 인스턴스 (개별 5종 + Maha)."""
    return {
        "IsolationForest": IsolationForestModel(contamination=0.03, n_estimators=200),
        "LOF":             LOFModel(n_neighbors=20, novelty=True),
        "ECOD":            ECODModel(),
        "HBOS":            HBOSModel(n_bins=20),
        "KNN":             KNNModel(n_neighbors=20),
        "Mahalanobis":     MahalanobisModel(ridge=1.0e-4),
    }


def _metrics(s_tr, s_te, s_mn, contamination, scen_te, scen_mine, test_scenarios):
    """train 점수로 threshold 보정 후 FP/탐지율 산출. (낮을수록 이상)"""
    thr = float(np.quantile(s_tr, contamination))
    fp_overall = float((s_te < thr).mean())
    fp_by_scen = {sc: float((s_te[scen_te == sc] < thr).mean()) for sc in test_scenarios}
    det_gpu = float((s_mn[scen_mine == "gpu_mining"] < thr).mean())
    det_cpu = float((s_mn[scen_mine == "cpu_mining"] < thr).mean())
    return fp_overall, fp_by_scen, det_gpu, det_cpu


def _mining_injection(n_each: int, seed: int) -> pd.DataFrame:
    """합성 채굴 raw 스냅샷 → extract_features 가 읽는 11 raw 컬럼 구성."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_each):  # GPU 채굴 (t-rex 류)
        rows.append({
            "scenario": "gpu_mining",
            "cpu_percent": rng.uniform(15, 35), "memory_percent": 60.0,
            "gpu_percent": rng.uniform(88, 99), "gpu_vram_mb": 7000, "gpu_total_mb": 8188,
            "gpu_power_w": rng.uniform(150, 175),
            "inbound_mb": 0.5, "outbound_mb": rng.uniform(15, 40),
            "external_packet_count": rng.randint(800, 2000),
            "disk_read_mb": 0.5, "disk_write_mb": 0.5,
        })
    for _ in range(n_each):  # CPU 채굴 (xmrig/Monero)
        rows.append({
            "scenario": "cpu_mining",
            "cpu_percent": rng.uniform(88, 99), "memory_percent": 70.0,
            "gpu_percent": rng.uniform(0, 8), "gpu_vram_mb": 200, "gpu_total_mb": 8188,
            "gpu_power_w": rng.uniform(15, 25),
            "inbound_mb": 0.3, "outbound_mb": rng.uniform(10, 25),
            "external_packet_count": rng.randint(500, 1500),
            "disk_read_mb": 0.2, "disk_write_mb": 0.2,
        })
    return pd.DataFrame(rows)


def _split(n: int, scenarios: np.ndarray, test_frac: float, seed: int):
    """시나리오 stratified train/test 인덱스."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for sc in np.unique(scenarios):
        idx = np.where(scenarios == sc)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - test_frac))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(test_idx)


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow 모델 비교 (IF/LOF vs ECOD/HBOS/KNN)")
    ap.add_argument("--csv", type=Path, default=Path("phase1_features.csv"))
    ap.add_argument("--contamination", type=float, default=0.05,
                    help="threshold 보정 분위 (train 정상의 이 비율을 이상으로 간주)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--mining-n", type=int, default=300, help="채굴 타입별 합성 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV 없음: {args.csv}")
        return 1

    # ── 1. 정상 데이터 로드 (원래 13 feature) ────────────────────────────────
    df_norm = parse_csv(args.csv.read_bytes())
    scen_norm = df_norm["scenario"].astype(str).values if "scenario" in df_norm else \
        np.array(["normal"] * len(df_norm))
    X_norm_raw = extract_features(df_norm)
    print(f"정상 데이터: {len(df_norm)}행, feature {X_norm_raw.shape[1]}차원")
    for sc, c in pd.Series(scen_norm).value_counts().items():
        print(f"  {sc}: {c}")

    # ── 2. 채굴 injection ────────────────────────────────────────────────────
    df_mine = _mining_injection(args.mining_n, args.seed)
    scen_mine = df_mine["scenario"].values
    X_mine_raw = extract_features(df_mine)

    # ── 3. train/test 분할 + 스케일 (학습 파이프라인 동일) ───────────────────
    tr, te = _split(len(df_norm), scen_norm, args.test_frac, args.seed)
    scaler = FeatureScaler()
    X_tr = apply_weights(scaler.fit_transform(X_norm_raw[tr]), slot="free")
    X_te = apply_weights(scaler.transform(X_norm_raw[te]), slot="free")
    X_mn = apply_weights(scaler.transform(X_mine_raw), slot="free")
    scen_te = scen_norm[te]
    print(f"\ntrain={len(tr)}  test(정상)={len(te)}  mining={len(df_mine)}  "
          f"contamination={args.contamination}\n")

    # ── 4. 모델 fit + 점수 캐시 (개별 + 앙상블 공용) ─────────────────────────
    test_scenarios = list(pd.unique(scen_te))
    scores = {}  # name -> (s_tr, s_te, s_mn)
    for name, model in _make_models().items():
        model.fit(X_tr)
        scores[name] = (model.score(X_tr), model.score(X_te), model.score(X_mn))

    def _row(name_or_label, group, triple):
        s_tr, s_te, s_mn = triple
        fp, fp_scen, dg, dc = _metrics(
            s_tr, s_te, s_mn, args.contamination, scen_te, scen_mine, test_scenarios)
        return {"label": name_or_label, "group": group,
                "fp_overall": fp, "fp_by_scen": fp_scen, "det_gpu": dg, "det_cpu": dc}

    # ── 5. 개별 모델 (IF/LOF baseline vs ECOD/HBOS/KNN shadow) ───────────────
    indiv = [_row(n, "baseline" if n in BASELINE else "shadow", scores[n])
             for n in BASELINE + SHADOW]

    # ── 6. 앙상블 변형 (ECOD 자리 → HBOS/KNN 교체) ──────────────────────────
    def _ensemble_triple(spec):
        agg = [np.zeros(len(X_tr)), np.zeros(len(X_te)), np.zeros(len(X_mn))]
        for mname, w in spec:
            for i in range(3):
                agg[i] = agg[i] + scores[mname][i] * w
        return tuple(agg)
    ens = [_row(label, "ensemble", _ensemble_triple(spec)) for label, spec in ENSEMBLES.items()]

    # ── 7. 출력 ──────────────────────────────────────────────────────────────
    def _print_table(title, rows, label_w):
        scen_hdr = "".join(f"{sc[:9]:>11}" for sc in test_scenarios)
        print(f"\n=== {title} ===")
        print(f"{'':<{label_w}}{'group':<10}{'FP전체':>9}{scen_hdr}{'탐지GPU':>10}{'탐지CPU':>10}")
        print("-" * (label_w + 39 + 11 * len(test_scenarios)))
        for r in rows:
            cells = "".join(f"{r['fp_by_scen'][sc]:>10.1%} " for sc in test_scenarios)
            print(f"{r['label']:<{label_w}}{r['group']:<10}{r['fp_overall']:>8.1%} {cells}"
                  f"{r['det_gpu']:>9.1%} {r['det_cpu']:>9.1%}")

    _print_table("개별 모델 — baseline(IF/LOF) vs shadow(ECOD/HBOS/KNN)", indiv, 16)
    _print_table("앙상블 변형 — 현재(IF+ECOD+Maha)에서 ECOD 교체", ens, 22)

    print("\n[해석]")
    print(f"  · threshold 를 train 정상의 {args.contamination:.0%} 분위로 보정 → test FP ≈ "
          f"{args.contamination:.0%} 가 기대값. 시나리오별 패턴·탐지율 차이가 비교 핵심.")
    print("  · dl_learning FP = GPU 풀로드 정상을 채굴로 오탐하는지 (핵심 지표).")
    print("  · 탐지GPU/CPU = 합성 채굴 injection 을 잡은 비율 (높을수록 좋음).")
    print("  · 앙상블: ECOD→HBOS/KNN 교체 시 FP↓(특히 idle) 면서 탐지 유지면 교체 가치.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
