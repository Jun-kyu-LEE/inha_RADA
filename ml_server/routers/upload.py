"""CSV 업로드 라우터 — 히스토리 보강용 (사전학습 X).

POST /upload/csv : CSV → history_manager 에 적재 + 학습 가능 조건이면 백그라운드 재학습.

사전학습 목적이면 /pretrain/csv 를 사용한다.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..core.csv_parser import parse_and_inject
from ..core.history import history_manager
from ..core.trainer import train_async
from ..core import online_training

router = APIRouter()


@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")

    contents = await file.read()
    result = parse_and_inject(contents)

    if result["errors"]:
        raise HTTPException(status_code=422, detail=result["errors"])

    triggered = False
    if online_training.is_ready_for_retrain():
        train_async()
        triggered = True

    return {
        "status":          "ok",
        "injected":        result["injected"],
        "pcs":             result["pcs"],
        "train_triggered": triggered,
    }
