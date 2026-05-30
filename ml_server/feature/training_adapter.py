"""학습/추론용 피처 어댑터 — snapshot/DataFrame → (n, F) 피처 행렬.

[설계 — derived 토글 가능]

  raw / derived 두 종류 피처 모두 ``feature_weights.yaml`` 의
  ``enabled_raw`` / ``enabled_derived`` 리스트에 적힌 키만 추출한다.
  derived 한 개를 끄려면 yaml 의 ``enabled_derived`` 에서 그 줄만 지우고
  재학습하면 됨 (피처 차원이 바뀌므로 기존 saved_models 는 무효화 → 새 학습 필수).

  enabled_* 가 yaml 에 없으면 아래 ``DEFAULT_*`` 가 사용된다.

[지원 derived — _compute_derived() 의 분기 한 곳에서만 정의]

  cpu_gpu_ratio       = clip(cpu / (gpu + 0.001), 0, 200)
  gpu_minus_cpu       = gpu - cpu
  gpu_vram_complex    = gpu * clip(1 - vram/total, 0, 1)
  traffic_direction   = clip(outbound / (inbound + 0.1), 0, 500)
  ext_traffic_complex = min(external_packet_count * outbound, 1e5)
  cpu_mining_index    = clip(cpu * outbound / (gpu + 1), 0, 1e4)

  새 derived 를 늘리려면 _compute_derived() 에 분기 한 줄 + yaml 의
  ``enabled_derived`` 에 키 추가 + ``derived_features`` 에 가중치 한 줄.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ..core.config_cache import load_yaml
from .columns import RAW_METRIC_COLS

WEIGHTS_CONFIG = Path(__file__).parent.parent / "config" / "feature_weights.yaml"

# enabled 리스트 미설정 시 fallback (= columns.RAW_METRIC_COLS 기반 ML raw 7 + network deps)
DEFAULT_RAW: list[str] = list(RAW_METRIC_COLS[:7])  # disk 포함 7 — yaml 과 동기
DEFAULT_DERIVED: list[str] = [
    "cpu_gpu_ratio", "gpu_minus_cpu", "gpu_vram_complex",
    "traffic_direction", "ext_traffic_complex", "cpu_mining_index",
]

# derived 계산에 추가로 필요한 raw 컬럼 (피처 벡터에는 포함되지 않을 수도 있음)
_DERIVED_RAW_DEPS: dict[str, tuple[str, ...]] = {
    "cpu_gpu_ratio":       ("cpu_percent", "gpu_percent"),
    "gpu_minus_cpu":       ("gpu_percent", "cpu_percent"),
    "gpu_vram_complex":    ("gpu_percent", "gpu_vram_mb", "gpu_total_mb"),
    "traffic_direction":   ("outbound_mb", "inbound_mb"),
    "ext_traffic_complex": ("external_packet_count", "outbound_mb"),
    "cpu_mining_index":    ("cpu_percent", "outbound_mb", "gpu_percent"),
}


def _load_enabled() -> tuple[list[str], list[str]]:
    """yaml 에서 enabled_raw / enabled_derived 리스트 로드 (없으면 DEFAULTS).

    config_cache.load_yaml 가 mtime 캐시를 자동 적용하므로 yaml 수정 즉시 반영.
    """
    if not WEIGHTS_CONFIG.exists():
        return list(DEFAULT_RAW), list(DEFAULT_DERIVED)
    cfg = load_yaml(WEIGHTS_CONFIG) or {}
    raw = cfg.get("enabled_raw") or list(DEFAULT_RAW)
    der = cfg.get("enabled_derived") or list(DEFAULT_DERIVED)
    return list(raw), list(der)


def get_feature_order() -> list[str]:
    """학습/추론에 실제 사용되는 피처 이름 순서 (raw + derived)."""
    raw, der = _load_enabled()
    return raw + der


def _compute_derived(name: str, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """ctx: raw key → 1D ndarray. 새 derived 추가 시 분기 한 줄 추가."""
    cpu = ctx.get("cpu_percent")
    gpu = ctx.get("gpu_percent")

    if name == "cpu_gpu_ratio":
        return np.clip(cpu / (gpu + 0.001), 0.0, 200.0)

    if name == "gpu_minus_cpu":
        return gpu - cpu

    if name == "gpu_vram_complex":
        vram = ctx["gpu_vram_mb"]
        total_safe = ctx["_gpu_total_mb_safe"]
        return gpu * np.clip(1.0 - vram / total_safe, 0.0, 1.0)

    if name == "traffic_direction":
        inb = ctx["inbound_mb"]
        out = ctx["outbound_mb"]
        return np.clip(out / (inb + 0.1), 0.0, 500.0)

    if name == "ext_traffic_complex":
        ext = ctx["external_packet_count"]
        out = ctx["outbound_mb"]
        return np.minimum(ext * out, 1.0e5)

    if name == "cpu_mining_index":
        out = ctx["outbound_mb"]
        return np.clip(cpu * out / (gpu + 1.0), 0.0, 1.0e4)

    raise ValueError(
        f"unknown derived feature: {name!r}. "
        f"_compute_derived() 에 분기 추가 필요."
    )


def _collect_ctx(df: pd.DataFrame, raw_names: list[str], der_names: list[str]) -> dict[str, np.ndarray]:
    """피처 계산에 필요한 모든 raw 컬럼을 안전하게 numpy array 로 모은다.
    빠진 컬럼은 0 으로 채운다. gpu_total_mb 는 별도 safe 변환.
    """
    needed: set[str] = set(raw_names)
    for d in der_names:
        needed.update(_DERIVED_RAW_DEPS.get(d, ()))
    # gpu_total_mb 는 derived 계산에 쓰이므로 항상 채움
    needed.add("gpu_total_mb")

    df = df.copy()
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0

    ctx: dict[str, np.ndarray] = {
        col: df[col].fillna(0.0).values.astype(float) for col in needed
    }
    # gpu_total_mb safe: 결측/0 → 관측 vram × 1.5 또는 8GB 중 큰 값
    total = ctx["gpu_total_mb"]
    vram = ctx.get("gpu_vram_mb", np.zeros_like(total))
    ctx["_gpu_total_mb_safe"] = np.maximum.reduce([
        total, vram * 1.5, np.full_like(total, 8192.0),
    ])
    return ctx


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """DataFrame → (n_samples, F) 피처 행렬. F = len(enabled_raw) + len(enabled_derived).

    학습용(csv_parser 결과 DataFrame) / 추론용(extract_one 경유) 둘 다 이 함수가 진입점.
    """
    raw_names, der_names = _load_enabled()
    ctx = _collect_ctx(df, raw_names, der_names)

    blocks: list[np.ndarray] = []
    if raw_names:
        blocks.append(np.column_stack([ctx[n] for n in raw_names]))
    if der_names:
        blocks.append(np.column_stack([_compute_derived(n, ctx) for n in der_names]))

    if not blocks:
        return np.zeros((len(df), 0), dtype=float)

    matrix = np.hstack(blocks)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def extract_one(snapshot: dict) -> np.ndarray:
    """단일 snapshot dict → (1, F) 피처 행렬. 추론 시 사용."""
    return extract_features(pd.DataFrame([snapshot]))


def _load_weights(slot: str = "free") -> np.ndarray:
    """feature_weights.yaml → enabled 순서에 맞춘 (raw+derived) 가중치 배열.

    각 피처의 base 가중치 × 슬롯 승수. yaml 미존재 시 ones.
    """
    raw_names, der_names = _load_enabled()
    feature_order = raw_names + der_names

    if not WEIGHTS_CONFIG.exists():
        return np.ones(len(feature_order))

    cfg = load_yaml(WEIGHTS_CONFIG) or {}
    raw_w = cfg.get("raw_features", {}) or {}
    der_w = cfg.get("derived_features", {}) or {}
    slot_key = (slot or "free").lower()
    mults = (cfg.get("slot_multipliers", {}) or {}).get(slot_key, {}) or {}
    default_mult = mults.get("default", 1.0)

    weights = []
    for feat in feature_order:
        base = raw_w.get(feat, der_w.get(feat, 1.0))
        mult = mults.get(feat, default_mult)
        weights.append(float(base) * float(mult))

    return np.array(weights, dtype=float)


def apply_weights(X: np.ndarray, slot: str = "free") -> np.ndarray:
    """스케일링 완료된 피처 행렬에 슬롯별 가중치 곱셈. X: (n, F)."""
    return X * _load_weights(slot)


class FeatureScaler:
    """RobustScaler 래퍼 — fit_transform/transform/save/load 4 메서드 인터페이스."""

    def __init__(self) -> None:
        self._scaler = RobustScaler()
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._scaler.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("스케일러 미피팅")
        return self._scaler.transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self._scaler, path)

    def load(self, path: str) -> None:
        self._scaler = joblib.load(path)
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ── 하위 호환 alias (이전 RAW_FEATURES/DERIVED_FEATURES/ALL_FEATURES 상수)
# 외부 코드가 import 했을 수 있으니 호출 시점에 yaml 반영해 동적으로 반환한다.
def _proxy_list(getter):
    class _Proxy(list):
        def __iter__(self_inner):
            return iter(getter())
        def __len__(self_inner):
            return len(getter())
        def __getitem__(self_inner, i):
            return getter()[i]
        def __repr__(self_inner):
            return repr(getter())
    return _Proxy()


RAW_FEATURES = _proxy_list(lambda: _load_enabled()[0])
DERIVED_FEATURES = _proxy_list(lambda: _load_enabled()[1])
ALL_FEATURES = _proxy_list(lambda: get_feature_order())
