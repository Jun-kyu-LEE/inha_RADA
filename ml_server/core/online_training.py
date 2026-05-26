"""온라인 재학습 — ON/OFF, pretrain+online 비율 혼합, 30분 window.

/analyze → history_manager 적재 → (enabled 시) 간격 충족하면 train_async().
재학습 데이터 = pretrain 캐시 subsample + PC별 최 recent online_window_samples.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import joblib
import pandas as pd

from .config_cache import load_yaml
from .history import history_manager

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "online_training.yaml"
SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"
PRETRAIN_CACHE_PATH = SAVED_MODELS_DIR / "_pretrain_cache.joblib"

_lock = threading.Lock()
_pretrain_cache: pd.DataFrame | None = None
_samples_since_retrain: int = 0
_enabled_override: bool | None = None


def _cfg() -> dict:
    return load_yaml(CONFIG_PATH) or {}


def is_enabled() -> bool:
    if _enabled_override is not None:
        return _enabled_override
    return bool(_cfg().get("enabled", False))


def set_enabled(enabled: bool) -> None:
    global _enabled_override
    _enabled_override = enabled
    logger.info("온라인 재학습 %s", "ON" if enabled else "OFF")


def get_status() -> dict:
    cfg = _cfg()
    cache = _pretrain_cache
    return {
        "enabled":                 is_enabled(),
        "enabled_from_yaml":       bool(cfg.get("enabled", False)),
        "enabled_override":        _enabled_override,
        "pretrain_cache_rows":     len(cache) if cache is not None else 0,
        "pretrain_cache_loaded":   cache is not None,
        "samples_since_retrain":   _samples_since_retrain,
        "online_window_samples":   int(cfg.get("online_window_samples", 360)),
        "pretrain_ratio":          float(cfg.get("pretrain_ratio", 0.7)),
        "online_ratio":            float(cfg.get("online_ratio", 0.3)),
        "retrain_interval_samples": int(cfg.get("retrain_interval_samples", 360)),
        "history_sizes":           history_manager.size(),
    }


def set_pretrain_cache(df: pd.DataFrame) -> None:
    """pretrain 직후 호출 — 메모리 + parquet 저장."""
    global _pretrain_cache
    with _lock:
        _pretrain_cache = df.copy()
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(_pretrain_cache, PRETRAIN_CACHE_PATH)
    logger.info("pretrain 캐시 저장: %d 행", len(df))


def load_pretrain_cache() -> bool:
    """서버 시작 시 디스크에서 pretrain 캐시 복원."""
    global _pretrain_cache
    if not PRETRAIN_CACHE_PATH.exists():
        return False
    try:
        with _lock:
            _pretrain_cache = joblib.load(PRETRAIN_CACHE_PATH)
        logger.info("pretrain 캐시 로드: %d 행", len(_pretrain_cache))
        return True
    except Exception as e:
        logger.warning("pretrain 캐시 로드 실패: %s", e)
        return False


def has_pretrain_cache() -> bool:
    return _pretrain_cache is not None and not _pretrain_cache.empty


def track_analyze_sample(pc_id: str, snapshot: dict) -> None:
    """/analyze 마다 history_manager 적재 + 재학습 카운터."""
    history_manager.append(pc_id, snapshot)
    global _samples_since_retrain
    _samples_since_retrain += 1


def mark_retrain_done() -> None:
    global _samples_since_retrain
    _samples_since_retrain = 0


def _online_window() -> int:
    return int(_cfg().get("online_window_samples", 360))


def _retrain_interval() -> int:
    return int(_cfg().get("retrain_interval_samples", 360))


def is_ready_for_retrain() -> bool:
    """pretrain 캐시 + online window 분량 이상."""
    if not has_pretrain_cache():
        return False
    online_df = history_manager.get_recent_training_dataframe(_online_window())
    min_online = min(_online_window(), _retrain_interval())
    return len(online_df) >= min_online


def should_trigger_from_analyze() -> bool:
    if not is_enabled():
        return False
    if not has_pretrain_cache():
        return False
    if _samples_since_retrain < _retrain_interval():
        return False
    return is_ready_for_retrain()


def maybe_trigger_from_analyze(train_async_fn) -> bool:
    """조건 충족 시 train_async_fn() 호출. True = 트리거됨."""
    if not should_trigger_from_analyze():
        return False
    train_async_fn()
    logger.info(
        "온라인 재학습 트리거 (since_last=%d, interval=%d)",
        _samples_since_retrain, _retrain_interval(),
    )
    return True


def build_mixed_training_dataframe() -> pd.DataFrame | None:
    """pretrain subsample + online 최근 window → 혼합 DataFrame."""
    if not has_pretrain_cache():
        logger.warning("pretrain 캐시 없음 — 혼합 재학습 불가")
        return None

    cfg = _cfg()
    online_ratio = float(cfg.get("online_ratio", 0.3))
    pretrain_ratio = float(cfg.get("pretrain_ratio", 0.7))
    pretrain_max = int(cfg.get("pretrain_max_samples", 10000))
    window = _online_window()

    online_df = history_manager.get_recent_training_dataframe(window)
    if online_df.empty:
        return None

    online_n = len(online_df)
    if online_ratio <= 0:
        pretrain_n = pretrain_max
    else:
        pretrain_n = int(online_n * pretrain_ratio / online_ratio)
    pretrain_n = max(1, min(pretrain_n, pretrain_max, len(_pretrain_cache)))

    pretrain_sample = _pretrain_cache.sample(n=pretrain_n, random_state=42)
    mixed = pd.concat([pretrain_sample, online_df], ignore_index=True)
    mixed = mixed.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "혼합 학습 세트: pretrain=%d + online=%d → total=%d (window=%d)",
        pretrain_n, online_n, len(mixed), window,
    )
    return mixed
