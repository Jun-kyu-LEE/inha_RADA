"""Local Outlier Factor — sklearn novelty 모드 래퍼.

raw decision_function 결과를 [-1, 0] 으로 정규화하여 IF 점수와 스케일 통일.
"""
import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseAnomalyModel


class LOFModel(BaseAnomalyModel):

    def __init__(self, n_neighbors: int = 20, novelty: bool = True, **kwargs):
        self.n_neighbors = n_neighbors
        self.novelty = novelty
        self._model: LocalOutlierFactor | None = None

    def fit(self, X: np.ndarray) -> None:
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=self.novelty,
        )
        self._model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._model.decision_function(X)
        mn, mx = raw.min(), raw.max()
        if mx == mn:
            return np.zeros_like(raw)
        return (raw - mx) / (mx - mn + 1e-9) - 1.0

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)
