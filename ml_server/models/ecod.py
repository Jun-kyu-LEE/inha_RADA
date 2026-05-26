"""Empirical CDF Outlier Detection (ECOD).

각 피처의 ECDF 양쪽 tail 확률을 -log 합산 → 차원 간 미세 부조화 포착.
파라미터 free, O(n*d) 매우 빠름.
"""
import numpy as np
import joblib
from .base import BaseAnomalyModel


class ECODModel(BaseAnomalyModel):

    def __init__(self, **kwargs):
        self._X_sorted: np.ndarray | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        n_train = self._X_sorted.shape[0]
        scores = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            r = np.searchsorted(self._X_sorted[:, j], X[:, j], side="right") / (n_train + 1)
            r = np.clip(r, 1e-6, 1 - 1e-6)
            scores += -np.log(np.minimum(r, 1 - r))
        return scores

    def fit(self, X: np.ndarray) -> None:
        self._X_sorted = np.sort(X, axis=0)
        raw = self._raw(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._X_sorted is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._raw(X)
        return -(raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)

    def save(self, path: str) -> None:
        joblib.dump({
            "X_sorted": self._X_sorted,
            "raw_min":  self._raw_min,
            "raw_max":  self._raw_max,
        }, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self._X_sorted = d["X_sorted"]
        self._raw_min  = d["raw_min"]
        self._raw_max  = d["raw_max"]
