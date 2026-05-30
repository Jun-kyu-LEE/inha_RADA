"""Mahalanobis 거리 기반 이상 탐지.

전체 공분산 + pinv 는 Windows 일부 BLAS 빌드에서 소표본(60×13) 연산 시
네이티브 크래시가 날 수 있어, **대각( feature-wise ) Mahalanobis** 를 사용한다.
d² = Σ ((xᵢ - μᵢ)² / (σᵢ² + ridge)).
"""
import numpy as np
import joblib
from .base import BaseAnomalyModel


class MahalanobisModel(BaseAnomalyModel):

    def __init__(self, ridge: float = 1e-4, **kwargs):
        self.ridge = float(ridge)
        self._mu: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mu
        return np.sum((diff * diff) / self._var, axis=1)

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("MahalanobisModel.fit requires at least 2 samples")
        self._mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=1)
        self._var = np.maximum(var, self.ridge) + self.ridge
        raw = self._raw(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._raw(X)
        return -(raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)

    def save(self, path: str) -> None:
        joblib.dump({
            "mu":      self._mu,
            "var":     self._var,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
            "ridge":   self.ridge,
        }, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self._mu      = d["mu"]
        self._var     = d.get("var")
        if self._var is None and "inv_cov" in d:
            # legacy full-cov 저장본은 per-PC 재학습 권장
            raise RuntimeError("legacy Mahalanobis checkpoint — refit required")
        self._raw_min = d["raw_min"]
        self._raw_max = d["raw_max"]
        self.ridge    = d["ridge"]
