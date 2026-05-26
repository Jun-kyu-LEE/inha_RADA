"""학습 코어 — pretrain (CSV 동기) / online (히스토리 비동기) / load_latest_model.

TrainerState 가 현재 로딩된 모델·스케일러·버전을 전역으로 관리하고
infer 라우터가 이걸 참조한다.

저장 위치: ml_server/saved_models/{pretrain|online}_{YYYYMMDD_HHMMSS}/
  ├── IsolationForestModel.pkl, ECODModel.pkl, MahalanobisModel.pkl, ...
  ├── scaler.pkl
  └── version.json

MAX_VERSIONS 초과 시 가장 오래된 버전 자동 삭제.
"""
import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..feature.training_adapter import extract_features, FeatureScaler, apply_weights
from .history import history_manager
from .model_loader import load_models_from_config, load_models_from_dir
from . import online_training

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).parent.parent / "saved_models"
MAX_VERSIONS = 3


class TrainerState:
    """현재 로딩된 모델·스케일러·버전 전역 관리 (singleton)."""

    def __init__(self):
        self.models: list = []                    # [(BaseAnomalyModel, weight), ...]
        self.scaler: FeatureScaler = FeatureScaler()
        self.version: str = "untrained"
        self.last_trained_at: datetime | None = None
        self.is_training: bool = False
        self.train_type: str = ""                 # "pretrain" / "online" / "loaded" / "rollback"
        self._lock = threading.Lock()

    def update(self, models: list, scaler: FeatureScaler, version: str, train_type: str = ""):
        with self._lock:
            self.models = models
            self.scaler = scaler
            self.version = version
            self.last_trained_at = datetime.now()
            self.train_type = train_type
        # 모델이 교체되면 inference 의 scorer 캐시도 함께 무효화한다.
        # 순환 import 회피를 위해 지연 import.
        from .inference import reset_scorer_cache
        reset_scorer_cache()

    @property
    def is_ready(self) -> bool:
        """추론 가능: 모델 + 스케일러 모두 준비된 경우만 True."""
        return len(self.models) > 0 and self.scaler.is_fitted


trainer_state = TrainerState()


def _fit_and_save(df: pd.DataFrame, version_prefix: str) -> dict:
    """DataFrame → 피처 추출 → 스케일링 → 학습 → 저장 → 메트릭 반환."""
    version = f"{version_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    version_dir = SAVED_MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    X_raw = extract_features(df)

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = apply_weights(X_scaled, slot="free")  # 학습은 free 슬롯 기준

    models = load_models_from_config()
    for model, _ in models:
        model.fit(X)

    metrics = _evaluate(models, X)

    for model, _ in models:
        model.save(str(version_dir / f"{model.get_name()}.pkl"))
    scaler.save(str(version_dir / "scaler.pkl"))

    version_info = {
        "version":       version,
        "type":          version_prefix,
        "trained_at":    datetime.now().isoformat(),
        "train_samples": len(df),
        "feature_count": int(X.shape[1]),
        "models":        [m.get_name() for m, _ in models],
        "metrics":       metrics,
    }
    if version_prefix == "online" and online_training.has_pretrain_cache():
        version_info["mix"] = {
            "pretrain_ratio": online_training.get_status()["pretrain_ratio"],
            "online_ratio":   online_training.get_status()["online_ratio"],
            "online_window":  online_training.get_status()["online_window_samples"],
        }
    with open(version_dir / "version.json", "w", encoding="utf-8") as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)

    logger.info(f"학습 완료 — 버전: {version}, 샘플: {len(df)}, metrics: {metrics}")

    trainer_state.update(models, scaler, version, train_type=version_prefix)
    _cleanup_old_versions()

    return {"version": version, "train_samples": len(df), "metrics": metrics}


def _evaluate(models: list, X: np.ndarray) -> dict:
    """비지도 학습 — IsolationForest 의 predict(-1=이상) 기반 anomaly_ratio 만 기록."""
    try:
        first_model = models[0][0]
        if hasattr(first_model, "_model") and hasattr(first_model._model, "predict"):
            preds = first_model._model.predict(X)
            anomaly_ratio = float((preds == -1).mean())
            return {"anomaly_ratio": round(anomaly_ratio, 4)}
    except Exception:
        pass
    return {}


def _cleanup_old_versions():
    if not SAVED_MODELS_DIR.exists():
        return
    versions = sorted(
        [p for p in SAVED_MODELS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    while len(versions) > MAX_VERSIONS:
        old = versions.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        logger.info(f"오래된 모델 버전 삭제: {old.name}")


def pretrain(df: pd.DataFrame) -> dict:
    """CSV DataFrame → 즉시 동기 사전학습. version prefix: 'pretrain'."""
    if trainer_state.is_training:
        raise RuntimeError("이미 학습 중입니다. 완료 후 재시도하세요.")

    trainer_state.is_training = True
    try:
        result = _fit_and_save(df, version_prefix="pretrain")
        online_training.set_pretrain_cache(df)
        return result
    finally:
        trainer_state.is_training = False


def train_async() -> None:
    """히스토리 기반 온라인 재학습 — 백그라운드 스레드. version prefix: 'online'."""
    if trainer_state.is_training:
        logger.info("이미 학습 중 — 요청 무시")
        return

    threading.Thread(target=_online_train_worker, daemon=True).start()


def _online_train_worker():
    trainer_state.is_training = True
    try:
        df = online_training.build_mixed_training_dataframe()
        if df is None or df.empty:
            df = history_manager.get_recent_training_dataframe(
                online_training.get_status()["online_window_samples"],
            )
            if df.empty:
                logger.warning("온라인 재학습 건너뜀 — history 부족")
                return
            if not online_training.has_pretrain_cache():
                logger.warning(
                    "pretrain 캐시 없음 — online only %d 행으로 재학습", len(df),
                )
        _fit_and_save(df, version_prefix="online")
        online_training.mark_retrain_done()
    except Exception as e:
        logger.exception(f"온라인 재학습 실패: {e}")
    finally:
        trainer_state.is_training = False


def load_latest_model() -> bool:
    """서버 시작 시 saved_models/ 의 최신 버전을 자동 로드.

    저장된 모델이 없으면 False — 호출자는 rule-based 모드로 fallback.
    """
    if not SAVED_MODELS_DIR.exists():
        return False

    versions = sorted(
        [p for p in SAVED_MODELS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not versions:
        return False

    latest = versions[0]
    try:
        models = load_models_from_dir(str(latest))
        if not models:
            logger.warning(f"버전 디렉터리에 유효한 모델 없음: {latest.name}")
            return False
        scaler = FeatureScaler()
        scaler.load(str(latest / "scaler.pkl"))
        trainer_state.update(models, scaler, latest.name, train_type="loaded")
        return True
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return False
