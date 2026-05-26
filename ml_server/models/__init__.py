"""ml_server.models — 이상 탐지 detector 풀.

config/models.yaml 에 등록된 detector 들이 동적으로 import 되어 앙상블로 결합.
BaseAnomalyModel 인터페이스(fit/score/save/load)를 구현하면 새 모델 추가 가능.
"""
