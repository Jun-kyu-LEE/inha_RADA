"""mtime 기반 YAML 캐시 — 파일이 변경된 경우에만 재읽기.

매 요청마다 디스크에서 YAML 을 읽는 I/O 낭비 제거.
설정 파일 핫리로드도 자연스럽게 지원됨.
"""
from pathlib import Path
from typing import Any

import yaml

_cache: dict[str, tuple[float, Any]] = {}


def load_yaml(path: Path) -> Any:
    """파일 수정 시각이 바뀐 경우에만 재파싱, 그 외엔 캐시 반환."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    cached = _cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _cache[key] = (mtime, data)
    return data
