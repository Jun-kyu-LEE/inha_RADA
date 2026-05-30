"""k-NN 평균 거리 기반 이상 탐지.

학습 분포에서 가까운 점이 적을수록 이상. LOF 와 달리 local density
비교가 아닌 절대 거리 사용.
"""
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from .base import BaseAnomalyModel


class KNNModel(BaseAnomalyModel):

    def __init__(self, n_neighbors: int = 20, **kwargs):
        self.n_neighbors = n_neighbors
        self._nn: NearestNeighbors | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        d, _ = self._nn.kneighbors(X)
        return d.mean(axis=1)

    def fit(self, X: np.ndarray) -> None:
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        raw = self._raw(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._raw(X)
        return -(raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)

    def save(self, path: str) -> None:
        joblib.dump({
            "nn":          self._nn,
            "raw_min":     self._raw_min,
            "raw_max":     self._raw_max,
            "n_neighbors": self.n_neighbors,
        }, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self._nn          = d["nn"]
        self._raw_min     = d["raw_min"]
        self._raw_max     = d["raw_max"]
        self.n_neighbors  = d["n_neighbors"]
