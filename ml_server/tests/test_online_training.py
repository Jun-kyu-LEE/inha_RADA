"""온라인 재학습 — pretrain+online 혼합, 360건 트리거, ON/OFF."""
from __future__ import annotations

import pandas as pd
import pytest

from ml_server.core import online_training
from ml_server.core.history import history_manager
from ml_server.feature.columns import RAW_METRIC_COLS

pytestmark = pytest.mark.unit


def _snap(i: int, pc: str = "PC-001") -> dict:
    row = {c: float(i % 100) for c in RAW_METRIC_COLS}
    row["timestamp"] = f"2026-05-21T12:{i % 60:02d}:00"
    return row


def _pretrain_df(n: int = 1000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = {c: float(i % 50) for c in RAW_METRIC_COLS}
        r["timestamp"] = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=i * 5)
        r["pc_id"] = f"PC-{i % 10:03d}"
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset():
    history_manager.clear()
    online_training.set_enabled(False)
    online_training.set_pretrain_cache(_pretrain_df(1000))
    online_training.mark_retrain_done()
    yield
    history_manager.clear()


def test_mixed_ratio_7_3():
    for i in range(360):
        history_manager.append("PC-001", _snap(i))

    mixed = online_training.build_mixed_training_dataframe()
    assert mixed is not None
    assert len(mixed) == 360 + int(360 * 0.7 / 0.3)  # 360 + 840


def test_trigger_requires_enabled_and_360_samples():
    online_training.set_enabled(True)
    triggered = []

    def _fake_train():
        triggered.append(True)

    for i in range(359):
        online_training.track_analyze_sample("PC-001", _snap(i))
        online_training.maybe_trigger_from_analyze(_fake_train)
    assert not triggered

    online_training.track_analyze_sample("PC-001", _snap(359))
    online_training.maybe_trigger_from_analyze(_fake_train)
    assert triggered


def test_disabled_never_triggers():
    online_training.set_enabled(False)
    triggered = []

    for i in range(400):
        online_training.track_analyze_sample("PC-001", _snap(i))
        online_training.maybe_trigger_from_analyze(lambda: triggered.append(True))
    assert not triggered


def test_recent_window_caps_per_pc():
    for i in range(500):
        history_manager.append("PC-001", _snap(i))

    df = history_manager.get_recent_training_dataframe(360)
    assert len(df) == 360
