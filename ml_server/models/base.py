"""
ml_server/models/base.py

모든 이상 탐지 모델이 구현해야 하는 추상 인터페이스.
fit / score / save / load 4 개 메서드 강제.

새 모델 추가:
  1. 이 클래스 상속 + 4 메서드 구현
  2. ml_server/models/ 에 파일 배치
  3. ml_server/config/models.yaml 의 ensemble 목록에 등록
  4. POST /admin/reload-models 호출 (재시작 불필요)
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseAnomalyModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """모델 학습. X: shape (n_samples, n_features)."""
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수. shape (n_samples,). 낮을수록(음수일수록) 이상."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    def get_name(self) -> str:
        return self.__class__.__name__
