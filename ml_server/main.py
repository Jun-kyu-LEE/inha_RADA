"""FastAPI app 엔트리포인트.

uvicorn ml_server.main:app --reload
또는 uvicorn ml_server:app --reload

[lifespan]
  · saved_models/ 의 최신 버전을 자동 로드 → ML+rule 하이브리드 모드.
  · 저장된 모델이 없으면 Rule-based 단독 (cold-start) 모드로 시작.
  · docker-compose(inha 기준)에는 saved_models 볼륨이 없음 — 컨테이너
    재생성 후 POST /pretrain/csv 로 pretrain 을 다시 실행해야 한다.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.analyze_router import router as analyze_router
from .api.status_router import router as status_router
from .api.clear_router import router as clear_router
from .routers import pretrain as pretrain_router
from .routers import train as train_router
from .routers import upload as upload_router
from .routers import feedback as feedback_router
from .routers import admin as admin_router
from .core.trainer import trainer_state
from .core import online_training

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # saved_models/ 에서 최신 모델 자동 로드 (없으면 rule-based 단독 모드).
    try:
        _load_latest_model()
        online_training.load_pretrain_cache()
    except Exception as e:
        logger.warning("저장된 모델 로드 실패 — Rule-based 단독 모드: %s", e)

    yield


def _load_latest_model() -> bool:
    """ml_server/saved_models/<version>/ 에서 최신 디렉터리를 찾아 로드."""
    from pathlib import Path
    from .core.model_loader import load_models_from_dir
    from .feature.training_adapter import FeatureScaler

    saved_dir = Path(__file__).parent / "saved_models"
    if not saved_dir.exists():
        logger.warning("saved_models/ 없음 → Rule-based 모드로 시작")
        return False

    versions = sorted(
        [p for p in saved_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not versions:
        logger.warning("저장된 모델 버전 없음 → Rule-based 모드로 시작")
        return False

    latest = versions[0]
    models = load_models_from_dir(str(latest))
    if not models:
        logger.warning("최신 버전(%s) 모델 로드 실패 → Rule-based 모드", latest.name)
        return False

    scaler = FeatureScaler()
    scaler_path = latest / "scaler.pkl"
    if not scaler_path.exists():
        logger.warning("scaler.pkl 없음 (%s) → Rule-based 모드", latest)
        return False
    scaler.load(str(scaler_path))

    trainer_state.update(models, scaler, latest.name, train_type="loaded")
    logger.info("모델 로드 완료: %s (%d 개)", latest.name, len(models))
    return True


app = FastAPI(
    title="PC 이상탐지 ML 서버 (ML + Rule-based hybrid)",
    lifespan=lifespan,
)

app.include_router(analyze_router)
app.include_router(status_router)
app.include_router(clear_router)
app.include_router(pretrain_router.router, tags=["사전학습"])
app.include_router(train_router.router,    tags=["온라인학습"])
app.include_router(upload_router.router,   tags=["CSV 업로드"])
app.include_router(feedback_router.router, tags=["피드백"])
app.include_router(admin_router.router,    tags=["어드민"])


@app.get("/health", tags=["헬스체크"])
def health():
    """docker-compose healthcheck 엔드포인트.

    `is_ready` 는 ML 앙상블 활성 여부 — false 이면 rule-based 단독 모드로 동작 중."""
    return {
        "status":   "ok",
        "version":  trainer_state.version,
        "is_ready": trainer_state.is_ready,
    }
