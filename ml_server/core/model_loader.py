"""config/models.yaml 의 ensemble 목록 → 동적 import + 인스턴스화.

- load_models_from_config(): 새 인스턴스 (학습 전)
- load_models_from_dir():    저장된 .pkl 에서 로드 (서버 시작 시 / 롤백)

config 의 module 경로는 ml_server.models.xxx 형식이어야 한다.
"""
import importlib
import logging
from pathlib import Path

import yaml

from ..models.base import BaseAnomalyModel

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"


def _load_config() -> list[dict]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ensemble", [])


def _normalize_weights(entries: list[dict]) -> list[float]:
    total = sum(e.get("weight", 1.0) for e in entries)
    return [e.get("weight", 1.0) / total for e in entries]


def load_models_from_config() -> list[tuple[BaseAnomalyModel, float]]:
    entries = _load_config()
    weights = _normalize_weights(entries)
    result = []

    for entry, weight in zip(entries, weights):
        try:
            module = importlib.import_module(entry["module"])
            cls = getattr(module, entry["class"])
            instance: BaseAnomalyModel = cls(**entry.get("params", {}))
            result.append((instance, weight))
        except Exception as e:
            logger.error(f"모델 로드 실패: {entry.get('name')} — {e}")

    return result


def load_models_from_dir(version_dir: str) -> list[tuple[BaseAnomalyModel, float]]:
    entries = _load_config()
    weights = _normalize_weights(entries)
    base = Path(version_dir)
    result = []

    for entry, weight in zip(entries, weights):
        try:
            module = importlib.import_module(entry["module"])
            cls = getattr(module, entry["class"])
            instance: BaseAnomalyModel = cls(**entry.get("params", {}))

            model_file = base / f"{entry['name']}.pkl"
            if model_file.exists():
                instance.load(str(model_file))
                result.append((instance, weight))
            else:
                logger.warning(f"모델 파일 없음: {model_file}")
        except Exception as e:
            logger.error(f"모델 파일 로드 실패: {entry.get('name')} — {e}")

    return result
