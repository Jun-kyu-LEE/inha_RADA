"""여러 모델의 점수를 가중 합산 → 앙상블 최종 점수.

PC 별 슬라이딩 윈도우(최근 5 건) 가중 평균 (linspace 0.5→1.0).
recent-heavy 가중치로 단일 스파이크 노이즈는 부드럽게, 지속 패턴은 강화.
"""
import logging
from collections import deque
from pathlib import Path

import numpy as np

from .config_cache import load_yaml
from ..models.base import BaseAnomalyModel

logger = logging.getLogger(__name__)

THRESHOLDS_CONFIG = Path(__file__).parent.parent / "config" / "thresholds.yaml"
WINDOW_SIZE = 5
DEFAULT_ANOMALY_THRESHOLD = -0.5


def _load_ensemble_threshold(slot: str) -> float:
    try:
        cfg = load_yaml(THRESHOLDS_CONFIG)
        return cfg.get(f"{slot.lower()}_slot", {}).get("ensemble_threshold", DEFAULT_ANOMALY_THRESHOLD)
    except Exception:
        return DEFAULT_ANOMALY_THRESHOLD


class EnsembleScorer:

    def __init__(self, models: list[tuple[BaseAnomalyModel, float]]):
        self._models = models
        self._windows: dict[str, deque] = {}

    def compute(self, pc_id: str, X: np.ndarray, slot: str = "free") -> dict:
        """X: (1, n_features) → if/lof/ensemble 점수 + 윈도우 평균 + 이상 여부."""
        scores = {}
        ensemble = 0.0

        for model, weight in self._models:
            name = model.get_name()
            s = float(model.score(X)[0])
            scores[name] = s
            ensemble += s * weight

        if_score  = next((s for n, s in scores.items() if "Isolation" in n), ensemble)
        lof_score = next((s for n, s in scores.items() if "LOF" in n),       ensemble)

        if pc_id not in self._windows:
            self._windows[pc_id] = deque(maxlen=WINDOW_SIZE)
        self._windows[pc_id].append((if_score, lof_score, ensemble))

        window = list(self._windows[pc_id])
        w = np.linspace(0.5, 1.0, len(window))
        w /= w.sum()

        avg_if  = float(np.average([x[0] for x in window], weights=w))
        avg_lof = float(np.average([x[1] for x in window], weights=w))
        avg_ens = float(np.average([x[2] for x in window], weights=w))

        threshold = _load_ensemble_threshold(slot)

        return {
            "if_score":       round(avg_if,  4),
            "lof_score":      round(avg_lof, 4),
            "ensemble_score": round(avg_ens, 4),
            "is_anomaly":     avg_ens < threshold,
            "raw_scores":     scores,
            "threshold":      threshold,
        }
