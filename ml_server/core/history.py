"""PC 별 최근 메트릭을 인메모리 deque 로 관리.

- maxlen=720 → 5 초 주기 × 720 = 1 시간치
- ML 서버 재시작 시 전체 초기화 (→ rule-based 모드 자동 전환)
- is_ready_to_train(): 360 개(30 분치) 이상이면 학습 가능
- get_variance_profile() / is_flat_usage(): 평탄 사용 패턴 탐지 (채굴 특성)

우리 14 키 snapshot 에는 cpu_temp 가 없으므로 variance feature 기본값은
cpu_percent + gpu_percent 로 축소되어 있음 (ml_server2 의 cpu_temp 제외).
"""
from collections import deque
import numpy as np
import pandas as pd

DEFAULT_MAXLEN = 720
MIN_TRAIN_POINTS = 360


class HistoryManager:

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        self._maxlen = maxlen
        self._store: dict[str, deque] = {}

    def append(self, pc_id: str, record: dict) -> None:
        if pc_id not in self._store:
            self._store[pc_id] = deque(maxlen=self._maxlen)
        self._store[pc_id].append(record)

    def bulk_append(self, pc_id: str, records: list[dict]) -> None:
        for r in records:
            self.append(pc_id, r)

    def get_dataframe(self, pc_id: str) -> pd.DataFrame:
        if pc_id not in self._store or not self._store[pc_id]:
            return pd.DataFrame()
        return pd.DataFrame(list(self._store[pc_id]))

    def get_all_dataframe(self) -> pd.DataFrame:
        frames = []
        for pc_id, dq in self._store.items():
            if dq:
                df = pd.DataFrame(list(dq))
                df["pc_id"] = pc_id
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_recent_training_dataframe(self, per_pc_limit: int) -> pd.DataFrame:
        """PC별 최근 per_pc_limit 건만 모아 학습용 DataFrame 반환."""
        frames = []
        for pc_id, dq in self._store.items():
            if not dq:
                continue
            rows = list(dq)[-per_pc_limit:]
            df = pd.DataFrame(rows)
            df["pc_id"] = pc_id
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
            out = out.sort_values("timestamp").reset_index(drop=True)
        return out

    def is_ready_to_train(self) -> bool:
        return any(len(dq) >= MIN_TRAIN_POINTS for dq in self._store.values())

    def size(self) -> dict[str, int]:
        return {pc_id: len(dq) for pc_id, dq in self._store.items()}

    def clear(self) -> None:
        self._store.clear()

    def get_variance_profile(
        self, pc_id: str,
        features: list[str] | None = None,
        last_n: int = 120,
    ) -> dict[str, dict]:
        df = self.get_dataframe(pc_id)
        if df.empty or len(df) < 10:
            return {}

        if features is None:
            features = ["cpu_percent", "gpu_percent", "outbound_mb"]

        df_slice = df.tail(last_n)
        result = {}
        for feat in features:
            if feat not in df_slice.columns:
                continue
            vals = df_slice[feat].fillna(0.0).values.astype(float)
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            cv   = std / (mean + 0.001)
            result[feat] = {"mean": round(mean, 2), "std": round(std, 2), "cv": round(cv, 4)}
        return result

    def is_flat_usage(
        self, pc_id: str,
        mean_threshold: float = 30.0,
        cv_threshold: float = 0.08,
        last_n: int = 120,
    ) -> bool:
        """비정상적으로 평탄한 자원 사용 → 채굴 의심.

        mean > mean_threshold 이고 cv < cv_threshold 면 True.
        채굴은 수십 분~수 시간 동안 CPU/GPU 를 고정 점유하므로 정상 작업의
        burst-idle 패턴보다 CV 가 현저히 낮다.
        """
        profile = self.get_variance_profile(
            pc_id,
            features=["cpu_percent", "gpu_percent"],
            last_n=last_n,
        )
        if not profile:
            return False
        for stats in profile.values():
            if stats["mean"] > mean_threshold and stats["cv"] < cv_threshold:
                return True
        return False


history_manager = HistoryManager()
