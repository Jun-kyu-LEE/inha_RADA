"""Isolation Forest detector — sklearn 래퍼."""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest as SklearnIF
from .base import BaseAnomalyModel


class IsolationForestModel(BaseAnomalyModel):

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, **kwargs):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model: SklearnIF | None = None

    def fit(self, X: np.ndarray) -> None:
        self._model = SklearnIF(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
        )
        self._model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        return self._model.decision_function(X)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)
