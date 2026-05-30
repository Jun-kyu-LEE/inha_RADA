"""AI Agent 실행 — Claude 우선, 실패 시 Mock fallback."""
from .. import config
from ..model.requests import MetricsRequest
from .claude_api_agent import call_claude_api, build_prompt
from .mock_agent import mock_agent_judgment

# 호환 alias — 기존 테스트가 monkeypatch.setattr(runner, "USE_REAL_CLAUDE", True)
# 형태로 분기를 강제하므로 모듈 변수로 노출. 실제 결정은 _should_use_claude().
USE_REAL_CLAUDE = False


def _should_use_claude() -> bool:
    if globals().get("USE_REAL_CLAUDE"):
        return True
    return config.use_real_claude()


def _normalize_hw_degradation(result: dict) -> None:
    """Claude/Mock 응답의 hw_degradation을 {NONE, SUSPECTED, CONFIRMED}로 정규화."""
    hw = result.get("hw_degradation")
    if isinstance(hw, bool):
        result["hw_degradation"] = "SUSPECTED" if hw else "NONE"
    elif hw not in ("NONE", "SUSPECTED", "CONFIRMED"):
        result["hw_degradation"] = "NONE"


def run_ai_agent(metrics: MetricsRequest, pattern_result: dict, global_hw: dict) -> dict:
    if _should_use_claude():
        try:
            result = call_claude_api(build_prompt(metrics, pattern_result, global_hw))
            result["is_mock"] = False
            _normalize_hw_degradation(result)
            return result
        except Exception as e:
            print(f"  [Claude API 호출 실패] {e} — Mock 판정으로 대체합니다.")
    result = mock_agent_judgment(metrics, pattern_result, global_hw)
    result["is_mock"] = True
    _normalize_hw_degradation(result)
    return result
