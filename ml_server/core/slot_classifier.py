"""datetime → "class" / "free" 슬롯 분류 (inha_RADA 와 동일 2분류).

주말·공휴일·방학도 free. 수업 시간(평일 09–18)만 class.
"""
from datetime import datetime

from ..config import get_timetable_slot


def classify_slot(ts: datetime) -> str:
    """"class" 또는 "free" 만 반환."""
    return get_timetable_slot(ts)
