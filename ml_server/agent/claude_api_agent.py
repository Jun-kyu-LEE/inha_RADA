"""Anthropic Claude API agent."""
from typing import Dict, Any
from ..model.requests import MetricsRequest


def build_prompt(metrics: MetricsRequest, pattern_result: dict,
                 global_hw: dict) -> str:
    alerts_text = "\n".join(
        [f"- [{a['severity']}] {a['type']}: {a['detail']}" for a in pattern_result["alerts"]]
    )
    procs_text = "\n".join(
        [f"- {p['name']} (CPU {p['cpu_percent']}%, MEM {p['memory_percent']}%, {p['path']})"
         for p in metrics.top_processes[:5]]
    )
    gpu_text = ""
    if metrics.gpu:
        gpu_text = (f"GPU {metrics.gpu.load_percent}%, VRAM {metrics.gpu.memory_used_mb}MB, "
                    f"SM {metrics.gpu.sm_utilization}%, 텐서코어 {metrics.gpu.tensor_core_active}%, "
                    f"전력 {metrics.gpu.power_draw_w}W")

    global_hw_text = ""
    if global_hw.get("detected"):
        global_hw_text = f"\n[전체 PC 노후화 신호]\n{global_hw.get('detail','')}"

    retrieval_text = ""
    ev = pattern_result.get("retrieval_evidence")
    if isinstance(ev, dict) and ev.get("available"):
        top_k_lines = []
        for c in ev.get("top_k", [])[:3]:
            top_k_lines.append(
                f"- {c.get('segment_id')} dist={c.get('distance')} "
                f"verdict={c.get('verdict')} score={c.get('score')}"
            )
        topk_block = "\n".join(top_k_lines) if top_k_lines else "- 없음"
        retrieval_text = (
            f"\n[유사 과거 사례 top-k]\n{topk_block}\n"
            f"[Peer 비교] same_slot_peers={ev.get('same_slot_peer_count',0)} "
            f"peer_mismatch={ev.get('peer_mismatch',False)} "
            f"retrieval_score={ev.get('retrieval_score',0)}\n"
        )

    scores   = pattern_result.get("scores", {})
    verdict  = pattern_result.get("verdict", "NORMAL")
    signals  = pattern_result.get("signals", {})
    active_s = [k for k, v in signals.items() if v and k not in ("is_gaming","is_compiling")]

    return f"""당신은 학교 실습실 PC 보안 및 유지보수 분석 전문가입니다.
아래 데이터를 분석해 이상 여부를 판단하고 JSON으로만 응답하세요.

[현재 메트릭]
PC={metrics.pc_id}, 시각={metrics.timestamp}, 시간표={pattern_result['timetable_slot']}
CPU={metrics.cpu_percent}%, 메모리={metrics.memory_percent}%
{gpu_text}
외부연결={metrics.external_packet_count}건, Net ↑{metrics.outbound_mb}MB/5s ↓{metrics.inbound_mb}MB/5s
{global_hw_text}
{retrieval_text}
[규칙 기반 스코어링 결과]
verdict={verdict} (NORMAL|OBSERVE|SUSPICIOUS|HIGH_RISK), 최종점수={scores.get('final',0):.1f}
(GPU채굴:{scores.get('gpu_mining',0)} CPU채굴:{scores.get('cpu_mining',0)} 스텔스:{scores.get('stealth',0)} 유출:{scores.get('exfil',0)} 프로세스:{scores.get('process',0)})
컨텍스트배율={scores.get('context_multiplier',1.0)} (게임={signals.get('is_gaming',False)}, 컴파일={signals.get('is_compiling',False)})
활성신호={active_s}

[탐지 알람]
{alerts_text if alerts_text else "- 이상 없음"}

[프로세스]
{procs_text}

JSON 형식으로만 응답:
{{"judgment":"NORMAL|SUSPICIOUS|DANGEROUS","severity":"LOW|MEDIUM|HIGH",
  "reason":"판단근거(2~3문장)","action":"추천조치",
  "hw_degradation":"NONE|SUSPECTED|CONFIRMED"}}""".strip()


def call_claude_api(prompt: str) -> dict:
    import requests as req, json
    from ..config import (
        get_anthropic_api_key,
        get_claude_model,
        get_claude_timeout_sec,
        get_claude_max_tokens,
    )

    api_key = get_anthropic_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    response = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": get_claude_model(),
              "max_tokens": get_claude_max_tokens(),
              "messages": [{"role": "user", "content": prompt}]},
        timeout=get_claude_timeout_sec(),
    )
    return json.loads(response.json()["content"][0]["text"])


class ClaudeApiAgent:
    def judge(self, metrics: MetricsRequest, pattern_result: Dict[str, Any],
              global_hw: Dict[str, Any]) -> Dict[str, Any]:
        return call_claude_api(build_prompt(metrics, pattern_result, global_hw))
