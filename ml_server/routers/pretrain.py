"""사전학습 라우터 — CSV 업로드 → 동기 학습.

POST /pretrain/csv   : CSV 파일 → 완료까지 대기 후 응답
GET  /pretrain/status: 학습 진행/완료 상태 조회
"""
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from ..core.csv_parser import parse_csv
from ..core.trainer import pretrain, trainer_state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/pretrain/csv")
async def pretrain_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")
    if trainer_state.is_training:
        raise HTTPException(status_code=409, detail="이미 학습 중입니다.")

    contents = await file.read()
    try:
        df = parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(f"사전학습 시작: {len(df)} 행, {df['pc_id'].nunique()} PC")

    try:
        result = pretrain(df)
    except Exception as e:
        logger.exception("사전학습 실패")
        raise HTTPException(status_code=500, detail=f"사전학습 실패: {e}")

    return {
        "status":        "ok",
        "version":       result["version"],
        "train_samples": result["train_samples"],
        "metrics":       result["metrics"],
        "message":       "사전학습 완료. 이제 ML 모드로 추론합니다.",
    }


@router.get("/pretrain/status")
def pretrain_status():
    return {
        "is_training":     trainer_state.is_training,
        "train_type":      trainer_state.train_type,
        "current_version": trainer_state.version,
        "mode":            "ml" if trainer_state.is_ready else "rule_based",
        "last_trained_at": trainer_state.last_trained_at.isoformat()
                           if trainer_state.last_trained_at else None,
    }
