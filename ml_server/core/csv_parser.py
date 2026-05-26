"""CSV 업로드 파서 — bytes → 검증된 DataFrame.

사전학습(/pretrain/csv) · 히스토리(/upload/csv) 공통.
11 개 raw 만 유지하고, 미리 계산된 derived·미사용 컬럼은 제거한다.
derived 는 ``training_adapter.extract_features()`` 가 raw 에서 계산.
"""
from __future__ import annotations

import io

import pandas as pd

from ..feature.columns import DROP_ON_IMPORT_COLS, OPTIONAL_LABEL_COLS, RAW_METRIC_COLS
from .history import history_manager

REQUIRED_COLUMNS = ["timestamp", "pc_id"]


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """CSV bytes → 11 raw (+ pc_id, timestamp, scenario?) DataFrame."""
    df = pd.read_csv(io.BytesIO(file_bytes), on_bad_lines="skip")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 누락: '{col}'")

    drop = [c for c in DROP_ON_IMPORT_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    for col in RAW_METRIC_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df[RAW_METRIC_COLS] = df[RAW_METRIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    keep = REQUIRED_COLUMNS + RAW_METRIC_COLS + [
        c for c in OPTIONAL_LABEL_COLS if c in df.columns
    ]
    df = df[keep].sort_values("timestamp").reset_index(drop=True)
    return df


def parse_and_inject(file_bytes: bytes) -> dict:
    """CSV bytes → 파싱 → history_manager 에 PC 별 주입. /upload/csv 전용."""
    errors: list[str] = []
    try:
        df = parse_csv(file_bytes)
    except ValueError as e:
        return {"injected": 0, "pcs": [], "errors": [str(e)]}

    injected = 0
    pcs: list[str] = []

    for pc_id, group in df.groupby("pc_id"):
        records = group.drop(columns=["pc_id"], errors="ignore").to_dict(orient="records")
        history_manager.bulk_append(str(pc_id), records)
        injected += len(records)
        pcs.append(str(pc_id))

    return {"injected": injected, "pcs": pcs, "errors": errors}
