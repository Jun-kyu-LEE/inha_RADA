"""sklearn MLPRegressor 기반 자기복원 오토인코더.

입력 = 출력 으로 학습 → 재구성 오차(MSE) 를 이상 점수로 사용.
사용 조건: 학습 데이터 5000 건+ 권장. 적은 데이터에서는 성능 보장 없음.
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from .base import BaseAnomalyModel


class AutoEncoderModel(BaseAnomalyModel):

    def __init__(
        self,
        hidden_layer_sizes=(16, 8, 16),
        max_iter: int = 300,
        learning_rate_init: float = 1e-3,
        random_state: int = 42,
        **kwargs,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self._model: MLPRegressor | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        recon = self._model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    def fit(self, X: np.ndarray) -> None:
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
        )
        self._model.fit(X, X)
        raw = self._raw(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델 미학습 — fit() 먼저 호출")
        raw = self._raw(X)
        return -(raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)

    def save(self, path: str) -> None:
        joblib.dump({
            "model":   self._model,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
        }, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self._model   = d["model"]
        self._raw_min = d["raw_min"]
        self._raw_max = d["raw_max"]
