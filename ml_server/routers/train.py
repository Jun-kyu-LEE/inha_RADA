"""온라인 학습 라우터.

POST /train : 메트릭 records 를 history 에 적재 + 학습 가능 조건이면 백그라운드 재학습 트리거.

records 는 make_snapshot() 의 14 키 dict 와 동일 형태를 기대한다.
가장 단순한 방식으로 list[dict] 를 받아 그대로 history_manager 에 누적.
"""
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

from ..core.history import history_manager
from ..core.trainer import train_async
from ..core import online_training

router = APIRouter()


class TrainRequest(BaseModel):
    pc_id:   str
    records: list[dict[str, Any]]


@router.post("/train")
def train(req: TrainRequest):
    history_manager.bulk_append(req.pc_id, req.records)

    triggered = False
    if online_training.is_ready_for_retrain():
        train_async()
        triggered = True

    return {
        "status":          "ok",
        "history_size":    history_manager.size().get(req.pc_id, 0),
        "train_triggered": triggered,
    }
