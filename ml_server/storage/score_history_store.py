"""ML 점수 / 룰 점수 누적 히스토리 — 가중 평균.

키는 (pc_id, slot) 튜플 — 슬롯 혼용 방지.
"""
from collections import deque
from typing import Dict, Tuple

from ..config import SCORE_WINDOW

# (pc_id, slot) → deque[float]
pc_score_history:   Dict[Tuple[str, str], deque] = {}
rule_score_history: Dict[Tuple[str, str], deque] = {}


def weighted_average(scores: list) -> float:
    """가중 평균: 최근일수록 가중치 1.0, 오래될수록 0.2까지 선형 감소."""
    n = len(scores)
    if n == 0:
        return 0.0
    weights = [max(0.2, 1.0 - 0.2 * (n - 1 - i)) for i in range(n)]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def append_ml_score(pc_id: str, slot: str, score: float) -> float:
    key = (pc_id, slot)
    if key not in pc_score_history:
        pc_score_history[key] = deque(maxlen=SCORE_WINDOW)
    pc_score_history[key].append(score)
    return weighted_average(list(pc_score_history[key]))


def append_rule_score(pc_id: str, slot: str, score: float, maxlen: int = 5) -> float:
    key = (pc_id, slot)
    if key not in rule_score_history:
        rule_score_history[key] = deque(maxlen=maxlen)
    rule_score_history[key].append(score)
    return weighted_average(list(rule_score_history[key]))
