"""관리 라우터 — 헬스체크, 상태, 모델 리로드/재학습/롤백.

- GET  /health              : 단순 헬스체크 (docker healthcheck 가 호출)
- GET  /admin/status        : ML 서버 전체 상태 (모드/버전/히스토리/슬롯)
- GET  /admin/model-info    : 현재 모델 목록 + 가중치 + version.json 메트릭
- POST /admin/reload-models : models.yaml 변경 후 재시작 없이 반영
- POST /admin/retrain       : 온라인 재학습 수동 트리거
- POST /admin/rollback      : 이전 버전으로 롤백
"""
import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.history import history_manager
from ..core.model_loader import load_models_from_config, load_models_from_dir
from ..core.slot_classifier import classify_slot
from ..core.trainer import trainer_state, train_async, SAVED_MODELS_DIR
from ..core import online_training
from ..feature.training_adapter import FeatureScaler

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.get("/admin/status")
def admin_status():
    return {
        "mode":            "ml" if trainer_state.is_ready else "rule_based",
        "model_version":   trainer_state.version,
        "train_type":      trainer_state.train_type,
        "last_trained_at": trainer_state.last_trained_at.isoformat()
                           if trainer_state.last_trained_at else None,
        "is_training":     trainer_state.is_training,
        "models_loaded":   [m.get_name() for m, _ in trainer_state.models],
        "history_size":    history_manager.size(),
        "history_ready":   online_training.is_ready_for_retrain(),
        "online_training": online_training.get_status(),
        "slot_now":        classify_slot(datetime.now()),
    }


class OnlineTrainingToggle(BaseModel):
    enabled: bool


@router.get("/admin/online-training")
def get_online_training():
    return online_training.get_status()


@router.post("/admin/online-training")
def set_online_training(body: OnlineTrainingToggle):
    online_training.set_enabled(body.enabled)
    return online_training.get_status()


@router.post("/admin/reload-models")
def reload_models():
    """models.yaml 변경 → 새 인스턴스 교체 (가중치는 유지)."""
    new_models = load_models_from_config()
    trainer_state.models = new_models
    logger.info(f"모델 설정 리로드: {[m.get_name() for m, _ in new_models]}")
    return {
        "status": "ok",
        "models": [m.get_name() for m, _ in new_models],
    }


@router.post("/admin/retrain")
def retrain():
    if trainer_state.is_training:
        raise HTTPException(status_code=409, detail="이미 학습 중입니다.")
    train_async()
    return {"status": "ok", "message": "온라인 재학습 백그라운드 시작"}


@router.get("/admin/model-info")
def model_info():
    metrics = {}
    version_file = SAVED_MODELS_DIR / trainer_state.version / "version.json"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            info = json.load(f)
            metrics = info.get("metrics", {})

    return {
        "version":    trainer_state.version,
        "train_type": trainer_state.train_type,
        "models": [
            {"name": m.get_name(), "weight": round(w, 4)}
            for m, w in trainer_state.models
        ],
        "metrics":    metrics,
    }


@router.post("/admin/rollback")
def rollback(target_version: str | None = None):
    if not SAVED_MODELS_DIR.exists():
        raise HTTPException(status_code=404, detail="저장된 모델 없음")

    versions = sorted(
        [p for p in SAVED_MODELS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )

    if len(versions) < 2 and target_version is None:
        raise HTTPException(status_code=404, detail="롤백할 이전 버전이 없습니다.")

    if target_version:
        target = SAVED_MODELS_DIR / target_version
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"버전 없음: {target_version}")
    else:
        current = trainer_state.version
        candidates = [v for v in versions if v.name != current]
        if not candidates:
            raise HTTPException(status_code=404, detail="롤백할 버전 없음")
        target = candidates[0]

    try:
        models = load_models_from_dir(str(target))
        scaler = FeatureScaler()
        scaler.load(str(target / "scaler.pkl"))
        trainer_state.update(models, scaler, target.name, train_type="rollback")
        logger.info(f"롤백 완료: {target.name}")
        return {"status": "ok", "rolled_back_to": target.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"롤백 실패: {e}")
