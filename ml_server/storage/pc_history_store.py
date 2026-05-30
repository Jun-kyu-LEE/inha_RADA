"""PC 단기/장기 히스토리 저장소 + 전체 PC 최신 메트릭."""
from collections import deque
from typing import Dict

from ..config import WINDOW_SIZE, TRAIN_WINDOW

# 단기 히스토리 (패턴 분석)
pc_history: Dict[str, deque] = {}

# 장기 히스토리 (학습용) — pc_id → {slot → deque}
pc_train_history: Dict[str, Dict[str, deque]] = {}

# 전체 PC 최신 메트릭 (Cross-PC 비교용)
all_pc_latest: Dict[str, dict] = {}


def ensure_pc_history(pc_id: str) -> deque:
    if pc_id not in pc_history:
        pc_history[pc_id] = deque(maxlen=WINDOW_SIZE)
    return pc_history[pc_id]


def update_train_history(pc_id: str, slot: str, snapshot: dict) -> None:
    if pc_id not in pc_train_history:
        pc_train_history[pc_id] = {}
    if slot not in pc_train_history[pc_id]:
        pc_train_history[pc_id][slot] = deque(maxlen=TRAIN_WINDOW)
    pc_train_history[pc_id][slot].append(snapshot)
