"""ML·룰·하이브리드 위험 점수 계산.

ml_server2 의 scoring 정책 그대로 이식.

  · ml_risk_score(): 앙상블 원점수(낮을수록 이상) → 0~1 위험도
                     thresholds.yaml::ml_risk_mapping 에 base/anomaly_scale/normal_scale 설정.
  · combine_risk(): 모델이 준비되어 있으면 ml × scoring_weights.ml_ensemble
                    + rule × scoring_weights.rule_based 가중합.
                    cold-start(모델 미준비) 시 rule_score 단독으로 판정.
"""
from __future__ import annotations


def ml_risk_score(
    ensemble_score: float,
    threshold: float,
    mapping: dict | None = None,
) -> float:
    """앙상블 원점수 → 0~1 위험도.

    ensemble_score < threshold 이면 이상으로 본다 (IsolationForest 등은 음수일수록 이상).
    base 부터 시작해 (threshold - ensemble_score) 만큼 anomaly_scale 로 증폭한다.
    정상 쪽이면 (ensemble_score - threshold) 만큼 normal_scale 로 감쇠.
    """
    m = mapping or {}
    base = m.get("base", 0.5)
    if ensemble_score < threshold:
        return min(1.0, base + (threshold - ensemble_score) * m.get("anomaly_scale", 2.0))
    return max(0.0, m.get("normal_scale", 0.5) * (1.0 - (ensemble_score - threshold)))


def combine_risk(
    ml_risk: float,
    rule_score: float,
    cfg: dict,
    *,
    model_ready: bool,
) -> tuple[float, bool]:
    """최종 위험도와 이상 여부 산출.

    model_ready=False(cold-start) 시 rule_score 단독 판정.
    """
    if model_ready:
        w = cfg.get("scoring_weights", {"ml_ensemble": 0.6, "rule_based": 0.4})
        total = ml_risk * w["ml_ensemble"] + rule_score * w["rule_based"]
        return total, total >= cfg.get("final_risk_threshold", 0.6)
    cold = cfg.get("cold_start_rule_threshold", 0.6)
    return rule_score, rule_score >= cold
