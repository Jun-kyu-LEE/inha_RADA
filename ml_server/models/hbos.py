"""Histogram-Based Outlier Score (HBOS).

각 피처를 n_bins 빈으로 나누고 빈 밀도의 -log 합산.
피처 독립 가정 → 상관관계 못 잡지만 매우 빠른 baseline.
"""
import numpy as np
import joblib
from .base import BaseAnomalyModel


class HBOSModel(BaseAnomalyModel):

    def __init__(self, n_bins: int = 20, **kwargs):
        self.n_bins = n_bins
        self._hists: list[np.ndarray] | None = None
        self._edges: list[np.ndarray] | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            idx = np.searchsorted(self._edges[j], X[:, j]) - 1
            idx = np.clip(idx, 0, self.n_bins - 1)
            scores += -np.log(self._hists[j][idx] + 1e-12)
        return scores

    def fit(self, X: np.ndarray) -> None:
        self._hists, self._edges = [], []
        for j in range(X.shape[1]):
            h, e = np.histogram(X[:, j], bins=self.n_bins)
            self._hists.append(h / (h.sum() + 1e-12))
            self._edges.append(e)
        raw = self._raw(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._hists is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._raw(X)
        return -(raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)

    def save(self, path: str) -> None:
        joblib.dump({
            "hists":   self._hists,
            "edges":   self._edges,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
            "n_bins":  self.n_bins,
        }, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self._hists   = d["hists"]
        self._edges   = d["edges"]
        self._raw_min = d["raw_min"]
        self._raw_max = d["raw_max"]
        self.n_bins   = d["n_bins"]
