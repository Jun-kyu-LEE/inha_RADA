"""탐지 성능 오프라인 평가.

1) 합성 패널 — 정상 / 채굴 / CPU채굴 (ground truth 확실)
2) CSV (phase1_features 등) — scenario 컬럼 기반 proxy 라벨

사용 (ML 컨테이너 / conda, repo 루트에서):

  python -m ml_server.eval.run_offline_eval
  python -m ml_server.eval.run_offline_eval --csv data/phase1_features.csv --sample 800

판정 기준 (알림 발생 = positive):
  overall_severity not in (NORMAL,)  OR  verdict in (SUSPICIOUS, HIGH_RISK)
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.inference import run_inference_snapshot
from ..core.trainer import load_latest_model

# scenario → 0=정상 기대, 1=이상 기대 (proxy — CSV 수집 시 라벨)
SCENARIO_EXPECT_ANOMALY: set[str] = set()
SCENARIO_EXPECT_NORMAL: set[str] = {
    "office",
    "dl_learning",
    "idle",
    "normal",
    "browser",
}


@dataclass
class EvalRow:
    name: str
    expected: int  # 1=anomaly
    predicted: int
    verdict: str
    severity: str
    total_risk: float
    types: list[str]


def _is_alert(verdict: str, severity: str, total_risk: float, *, risk_th: float = 0.60) -> bool:
    if verdict in ("SUSPICIOUS", "HIGH_RISK"):
        return True
    if severity not in (None, "", "NORMAL"):
        return True
    return total_risk >= risk_th


def _snapshot_from_row(row: pd.Series) -> dict:
    return {
        "timestamp":             str(row.get("timestamp", "")),
        "cpu_percent":           float(row.get("cpu_percent", 0)),
        "memory_percent":        float(row.get("memory_percent", 0)),
        "gpu_percent":           float(row.get("gpu_percent", 0)),
        "gpu_vram_mb":           float(row.get("gpu_vram_mb", 0)),
        "gpu_total_mb":          float(row.get("gpu_total_mb", 8188)),
        "gpu_power_w":           float(row.get("gpu_power_w", 0)),
        "inbound_mb":            float(row.get("inbound_mb", 0)),
        "outbound_mb":           float(row.get("outbound_mb", 0)),
        "external_packet_count": float(row.get("external_packet_count", 0)),
        "disk_read_mb":          float(row.get("disk_read_mb", 0)),
        "disk_write_mb":         float(row.get("disk_write_mb", 0)),
        "top_processes":         [],
    }


def _infer_snapshot(snap: dict, pc_id: str, ts: datetime) -> EvalRow:
    out = run_inference_snapshot(
        snap, pc_id=pc_id, timestamp=ts, append_history=False, slot_override="free",
    )
    verdict = "HIGH_RISK" if out["total_risk"] >= 0.85 else (
        "SUSPICIOUS" if out["total_risk"] >= 0.60 or out["is_anomaly"] else (
            "OBSERVE" if out["total_risk"] >= 0.30 else "NORMAL"
        )
    )
    severity = "HIGH" if verdict == "HIGH_RISK" else (
        "MEDIUM" if verdict == "SUSPICIOUS" else (
            "LOW" if verdict == "OBSERVE" else "NORMAL"
        )
    )
    pred = int(_is_alert(verdict, severity, out["total_risk"]))
    return EvalRow(
        name=pc_id,
        expected=-1,
        predicted=pred,
        verdict=verdict,
        severity=severity,
        total_risk=float(out["total_risk"]),
        types=list(out.get("anomaly_types", [])),
    )


def _synthetic_panels() -> list[tuple[str, dict, int]]:
    """(label, snapshot, expected_anomaly)"""
    rng = random.Random(42)
    panels: list[tuple[str, dict, int]] = []

    for i in range(20):
        panels.append(("normal", {
            "cpu_percent": rng.uniform(8, 35),
            "memory_percent": rng.uniform(40, 70),
            "gpu_percent": rng.uniform(0, 25),
            "gpu_vram_mb": 1500, "gpu_total_mb": 8188, "gpu_power_w": 15,
            "inbound_mb": rng.uniform(0.1, 2), "outbound_mb": rng.uniform(0.1, 1),
            "external_packet_count": rng.randint(10, 80),
            "disk_read_mb": 0.1, "disk_write_mb": 0.1,
            "top_processes": [{"name": "chrome.exe", "cpu_percent": 5}],
        }, 0))

    for i in range(20):
        panels.append(("gpu_mining", {
            "cpu_percent": rng.uniform(15, 35),
            "memory_percent": 60,
            "gpu_percent": rng.uniform(88, 99),
            "gpu_vram_mb": 7000, "gpu_total_mb": 8188, "gpu_power_w": 160,
            "inbound_mb": 0.5, "outbound_mb": rng.uniform(15, 40),
            "external_packet_count": rng.randint(800, 2000),
            "disk_read_mb": 0.5, "disk_write_mb": 0.5,
            "top_processes": [{"name": "t-rex.exe", "cpu_percent": 10}],
        }, 1))

    for i in range(20):
        panels.append(("cpu_mining_xmrig", {
            "cpu_percent": rng.uniform(88, 99),
            "memory_percent": 70,
            "gpu_percent": rng.uniform(0, 8),
            "gpu_vram_mb": 200, "gpu_total_mb": 8188, "gpu_power_w": 20,
            "inbound_mb": 0.3, "outbound_mb": rng.uniform(10, 25),
            "external_packet_count": rng.randint(500, 1500),
            "disk_read_mb": 0.2, "disk_write_mb": 0.2,
            "top_processes": [{"name": "xmrig.exe", "cpu_percent": 90}],
        }, 1))

    for i in range(20):
        panels.append(("dual_load", {
            "cpu_percent": rng.uniform(90, 98),
            "memory_percent": 75,
            "gpu_percent": rng.uniform(90, 98),
            "gpu_vram_mb": 7500, "gpu_total_mb": 8188, "gpu_power_w": 170,
            "inbound_mb": 0.5, "outbound_mb": rng.uniform(8, 20),
            "external_packet_count": rng.randint(600, 1800),
            "disk_read_mb": 0.3, "disk_write_mb": 0.3,
            "top_processes": [{"name": "xmrig.exe", "cpu_percent": 88}],
        }, 1))

    return panels


def _metrics(rows: list[EvalRow]) -> dict:
    tp = sum(1 for r in rows if r.expected == 1 and r.predicted == 1)
    tn = sum(1 for r in rows if r.expected == 0 and r.predicted == 0)
    fp = sum(1 for r in rows if r.expected == 0 and r.predicted == 1)
    fn = sum(1 for r in rows if r.expected == 1 and r.predicted == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(rows) if rows else 0.0
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
    }


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  n={m['n']}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"  precision={m['precision']:.2%}  recall={m['recall']:.2%}  "
          f"f1={m['f1']:.2%}  accuracy={m['accuracy']:.2%}")


def run_synthetic() -> list[EvalRow]:
    rows: list[EvalRow] = []
    ts = datetime(2026, 5, 20, 14, 0, 0)
    for label, snap, exp in _synthetic_panels():
        snap = dict(snap)
        snap["timestamp"] = ts.isoformat()
        row = _infer_snapshot(snap, f"syn-{label}", ts)
        row.expected = exp
        row.name = label
        rows.append(row)
    return rows


def run_csv(path: Path, sample: int, seed: int) -> list[EvalRow]:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        raise ValueError("CSV 에 scenario 컬럼이 없습니다.")

    counts = df["scenario"].value_counts()
    print("\n[CSV scenario 분포]")
    for sc, n in counts.items():
        tag = "→ 이상 기대" if sc in SCENARIO_EXPECT_ANOMALY else (
            "→ 정상 기대" if sc in SCENARIO_EXPECT_NORMAL else "→ 라벨 미정")
        print(f"  {sc}: {n}{tag}")

    labeled = df[df["scenario"].isin(SCENARIO_EXPECT_ANOMALY | SCENARIO_EXPECT_NORMAL)]
    if labeled.empty:
        print("  (라벨된 scenario 없음 — 합성 패널만 신뢰)")
        return []

    if len(labeled) > sample:
        labeled = labeled.sample(n=sample, random_state=seed)

    rows: list[EvalRow] = []
    for i, (_, r) in enumerate(labeled.iterrows()):
        sc = str(r["scenario"])
        exp = 1 if sc in SCENARIO_EXPECT_ANOMALY else 0
        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        if pd.isna(ts):
            ts = datetime(2026, 5, 20, 12, 0, 0)
        snap = _snapshot_from_row(r)
        row = _infer_snapshot(snap, f"csv-{i}", ts.to_pydatetime())
        row.expected = exp
        row.name = sc
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ML 탐지 오프라인 평가")
    parser.add_argument("--csv", type=Path, default=None, help="라벨 CSV (scenario 컬럼)")
    parser.add_argument("--sample", type=int, default=500, help="CSV 최대 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = load_latest_model()
    print(f"모델 로드: {'OK' if loaded else 'rule-only (pretrain 없음)'}")

    syn_rows = run_synthetic()
    _print_metrics("합성 패널 (정상20 + GPU채굴20 + CPU채굴20 + dual20)", _metrics(syn_rows))

    fails = [r for r in syn_rows if r.expected != r.predicted]
    if fails:
        print("\n[합성 오분류 샘플 (최대 10)]")
        for r in fails[:10]:
            print(f"  {r.name}: exp={r.expected} pred={r.predicted} "
                  f"risk={r.total_risk:.3f} verdict={r.verdict} types={r.types}")

    if args.csv and args.csv.exists():
        csv_rows = run_csv(args.csv, args.sample, args.seed)
        if csv_rows:
            _print_metrics(f"CSV proxy ({args.csv.name})", _metrics(csv_rows))
    elif args.csv:
        print(f"\nCSV 없음: {args.csv}")

    print("\n[해석 가이드]")
    print("  · 합성 recall — 채굴 패턴을 얼마나 잡는지 (높을수록 좋음)")
    print("  · 합성 precision — 정상을 오탐하지 않는지 (높을수록 좋음)")
    print("  · phase1 CSV는 office/dl_learning 위주면 proxy 정확도는 참고용")
    print("  · 실전: python tools/anomaly_trigger.py + Grafana anomaly_history 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
