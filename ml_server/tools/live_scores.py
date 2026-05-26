"""실시간 ML 점수만 출력 (client_core 수집 + /analyze).

사용 (repo 루트, ddr_project):

  set RADA_MODE=mlserver
  set RADA_ML_SERVER_URL=http://127.0.0.1:8000/analyze
  python -m ml_server.tools.live_scores

  python -m ml_server.tools.live_scores --once   # 1회만
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _ensure_repo_root() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def _format_line(metrics: dict, server_result: dict | None) -> str:
    ts = str(metrics.get("timestamp", ""))[:19]
    if not server_result:
        return f"[{ts}] ML 연결 실패"

    scores = server_result.get("scores") or {}
    parts = [
        f"[{ts}]",
        f"slot={server_result.get('timetable_slot', '-')}",
        f"verdict={server_result.get('verdict', '?')}",
        f"severity={server_result.get('overall_severity', '?')}",
        f"final={scores.get('final', 0)}",
        f"rule={scores.get('rule', 0)}",
        f"ml={scores.get('ml', 0)}",
    ]
    if scores.get("ensemble") is not None:
        parts.append(f"ensemble={scores.get('ensemble')}")
    if scores.get("if_score") is not None:
        parts.append(f"if={scores.get('if_score')}")

    for key in ("gpu_mining", "cpu_mining", "exfil", "stealth", "process"):
        val = scores.get(key)
        if val:
            parts.append(f"{key}={val}")

    alerts = server_result.get("alerts") or []
    if alerts:
        flags = ",".join(f"{a.get('type')}:{a.get('score', 0)}" for a in alerts)
        parts.append(f"alerts=[{flags}]")

    return " ".join(parts)


def main() -> int:
    _ensure_repo_root()

    parser = argparse.ArgumentParser(description="ML 점수만 실시간 출력")
    parser.add_argument("--once", action="store_true", help="1회만 실행")
    args = parser.parse_args()

    os.environ.setdefault("RADA_MODE", "mlserver")
    os.environ.setdefault("RADA_ML_SERVER_URL", "http://127.0.0.1:8000/analyze")

    from client_core.runtime.loop import ClientRuntime

    class ScoreOnlyRuntime(ClientRuntime):
        def _banner(self) -> None:
            cfg = self.config
            print(f"score-only | {cfg.target_url()} | {cfg.interval}s", flush=True)

        def _print(self, metrics, local_alerts, hw_alerts, boxplot, server_result) -> None:
            print(_format_line(metrics, server_result), flush=True)

    rt = ScoreOnlyRuntime()
    rt.collector.collect()
    time.sleep(rt.config.interval)

    if args.once:
        rt.step()
        return 0

    while True:
        t0 = time.time()
        rt.step()
        time.sleep(max(0.0, rt.config.interval - (time.time() - t0)))


if __name__ == "__main__":
    raise SystemExit(main())
