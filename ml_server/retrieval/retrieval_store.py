"""In-memory retrieval store — slot 별 segment + embedding 누적.

R2: distance 함수에 cosine 모드 추가. RETRIEVAL_DISTANCE_MODE 로 전환.
  - cosine  (기본): 1 - cos(a,b) ∈ [0, 2]. 스케일 무관.
  - euclidean    : √Σ(a_i - b_i)^2 ∈ [0, ∞). 기존 동작 유지.

기존 search_similar 의 응답 형식은 동일. distance 값 의미만 바뀜.
"""
from __future__ import annotations
import math
import os
from collections import deque
from typing import Dict, List, Deque

_MAXLEN = 20000

segment_history_by_slot: Dict[str, Deque[dict]] = {
    "class": deque(maxlen=_MAXLEN),
    "free":  deque(maxlen=_MAXLEN),
}


def _ensure_slot(slot: str) -> Deque[dict]:
    if slot not in segment_history_by_slot:
        segment_history_by_slot[slot] = deque(maxlen=_MAXLEN)
    return segment_history_by_slot[slot]


def reset_store() -> None:
    for q in segment_history_by_slot.values():
        q.clear()


def add_segment(segment: dict, embedding: List[float],
                verdict: str, score: float) -> None:
    if not segment or not embedding:
        return
    slot = segment.get("slot") or "free"
    item = {
        "segment_id": segment.get("segment_id"),
        "pc_id":      segment.get("pc_id"),
        "slot":       slot,
        "embedding":  list(embedding),
        "verdict":    verdict,
        "score":      float(score) if score is not None else 0.0,
        "start_ts":   segment.get("start_ts"),
        "end_ts":     segment.get("end_ts"),
    }
    _ensure_slot(slot).append(item)


def _euclidean(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        d = (a[i] - b[i])
        s += d * d
    return math.sqrt(s)


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """1 - cos(a, b). 한쪽이 0벡터면 2.0 (최대 거리) 반환."""
    n = min(len(a), len(b))
    if n == 0:
        return 2.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 2.0
    cos = dot / math.sqrt(na * nb)
    # numerical clamp
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


def _distance_mode() -> str:
    m = os.environ.get("RETRIEVAL_DISTANCE_MODE", "cosine").strip().lower()
    if m not in ("cosine", "euclidean"):
        m = "cosine"
    return m


def _distance(a: List[float], b: List[float]) -> float:
    if _distance_mode() == "euclidean":
        return _euclidean(a, b)
    return _cosine_distance(a, b)


def search_similar(segment: dict, embedding: List[float],
                   top_k: int = 3) -> List[dict]:
    if not segment or not embedding:
        return []
    slot = segment.get("slot") or "free"
    self_id = segment.get("segment_id")
    self_end = segment.get("end_ts")
    pool = list(_ensure_slot(slot))
    scored = []
    for past in pool:
        if past.get("segment_id") == self_id and past.get("end_ts") == self_end:
            continue
        dist = _distance(embedding, past.get("embedding") or [])
        scored.append((dist, past))
    scored.sort(key=lambda x: x[0])
    results = []
    for dist, past in scored[:top_k]:
        results.append({
            "segment_id": past.get("segment_id"),
            "pc_id":      past.get("pc_id"),
            "distance":   round(dist, 4),
            "verdict":    past.get("verdict"),
            "score":      past.get("score"),
            "start_ts":   past.get("start_ts"),
            "end_ts":     past.get("end_ts"),
        })
    return results


def clear_pc(pc_id: str) -> bool:
    removed = False
    for slot, q in list(segment_history_by_slot.items()):
        kept = [item for item in q if item.get("pc_id") != pc_id]
        if len(kept) != len(q):
            removed = True
            new_q: Deque[dict] = deque(kept, maxlen=q.maxlen)
            segment_history_by_slot[slot] = new_q
    return removed
