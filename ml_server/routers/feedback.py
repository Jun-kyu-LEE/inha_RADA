"""피드백 라우터 — 관리자 오탐 보고.

POST /feedback : 누적 카운터(전체 + PC 별)가 임계 초과 시 온라인 재학습 트리거.
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from ..core.trainer import train_async
from ..core import online_training

logger = logging.getLogger(__name__)
router = APIRouter()

_feedback_total: int = 0
_feedback_per_pc: dict[str, int] = {}

THRESHOLD_TOTAL  = 10
THRESHOLD_PER_PC = 3


class FeedbackPayload(BaseModel):
    pc_id:           str
    timestamp:       str
    correct_verdict: str   # "normal" | "anomaly"
    memo:            str = ""


@router.post("/feedback")
def receive_feedback(payload: FeedbackPayload):
    global _feedback_total

    _feedback_total += 1
    _feedback_per_pc[payload.pc_id] = _feedback_per_pc.get(payload.pc_id, 0) + 1

    triggered = False
    reason = ""

    if _feedback_total >= THRESHOLD_TOTAL:
        reason = f"전체 피드백 {_feedback_total} 건 초과"
        _trigger_retrain(reason)
        _feedback_total = 0
        triggered = True
    elif _feedback_per_pc.get(payload.pc_id, 0) >= THRESHOLD_PER_PC:
        reason = f"PC {payload.pc_id} 피드백 {_feedback_per_pc[payload.pc_id]} 건 초과"
        _trigger_retrain(reason)
        _feedback_per_pc[payload.pc_id] = 0
        triggered = True

    return {
        "status":           "ok",
        "feedback_total":   _feedback_total,
        "feedback_this_pc": _feedback_per_pc.get(payload.pc_id, 0),
        "train_triggered":  triggered,
        "trigger_reason":   reason,
    }


def _trigger_retrain(reason: str):
    logger.info(f"피드백 기반 온라인 재학습 트리거: {reason}")
    train_async()
