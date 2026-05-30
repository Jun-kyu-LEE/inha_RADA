
=== Stage 1: Health checks ===
  [PASS] ML  /health 200  | code=200 body={'status': 'ok', 'timestamp': '2026-05-30T23:20:13.566043'}
  [PASS] ML  /status running  | code=200
    ML inference mode: rule_based
    WARN: saved_models 없음 — rule-based 단독 모드. ML+rule 하이브리드 검증을 원하면 POST /pretrain/csv 후 재실행.
  [PASS] Spring API 401 w/o key  | code=401
  [PASS] Grafana /api/health 200  | code=200

=== Stage 2: ML 서버 단독 /analyze 시나리오 4종 ===
  [PASS] normal: HTTP 200 + verdict 존재  | code=200
    normal verdict: NORMAL / sev=NORMAL
    normal alert_types: []
    normal category_signals: {"resource_abnormal": false, "network_abnormal": false, "system_abnormal": false, "sustained_minutes": 0, "triggered_patterns": [], "verdict_from_gating": "NORMAL"}
    normal scores.final: 0.0
  [PASS] normal → verdict=NORMAL or OBSERVE  | got=NORMAL
  [PASS] gpu_mining: HTTP 200 + verdict 존재  | code=200
    gpu_mining verdict: HIGH_RISK / sev=HIGH
    gpu_mining alert_types: ["GPU_MINING", "CONFIRMED_MINING", "GPU_CPU_IMBALANCE", "HIGH_GPU", "POOL_TRAFFIC", "OUTBOUND_DOMINANT", "MANY_EXTERNAL"]
    gpu_mining category_signals: {"resource_abnormal": true, "network_abnormal": true, "system_abnormal": false, "sustained_minutes": 1, "triggered_patterns": ["GPU_MINING", "CONFIRMED_MINING", "GPU_CPU_IMBALANCE", "HIGH_GPU", "POOL_TRAFFIC", "OUTBOUND_DOMINANT", "MANY_EXT
    gpu_mining scores.final: 100.0
  [PASS] gpu_mining → mining/network alert 존재  | alerts=['GPU_MINING', 'CONFIRMED_MINING', 'GPU_CPU_IMBALANCE', 'HIGH_GPU', 'POOL_TRAFFIC', 'OUTBOUND_DOMINANT', 'MANY_EXTERNAL'] verdict=HIGH_RISK
  [PASS] gpu_mining → category.resource OR network abnormal
  [PASS] cpu_only_mining: HTTP 200 + verdict 존재  | code=200
    cpu_only_mining verdict: HIGH_RISK / sev=HIGH
    cpu_only_mining alert_types: ["CONFIRMED_MINING", "CPU_GPU_IMBALANCE", "HIGH_CPU", "MANY_EXTERNAL", "STRATUM_PATTERN"]
    cpu_only_mining category_signals: {"resource_abnormal": true, "network_abnormal": true, "system_abnormal": false, "sustained_minutes": 1, "triggered_patterns": ["CONFIRMED_MINING", "CPU_GPU_IMBALANCE", "HIGH_CPU", "MANY_EXTERNAL", "STRATUM_PATTERN"], "verdict_from_gating": 
    cpu_only_mining scores.final: 100.0
  [PASS] cpu_only_mining → cpu/network alert 존재  | alerts=['CONFIRMED_MINING', 'CPU_GPU_IMBALANCE', 'HIGH_CPU', 'MANY_EXTERNAL', 'STRATUM_PATTERN'] verdict=HIGH_RISK
  [PASS] network_anomaly: HTTP 200 + verdict 존재  | code=200
    network_anomaly verdict: SUSPICIOUS / sev=MEDIUM
    network_anomaly alert_types: ["POOL_TRAFFIC", "OUTBOUND_DOMINANT", "MANY_EXTERNAL"]
    network_anomaly category_signals: {"resource_abnormal": false, "network_abnormal": true, "system_abnormal": false, "sustained_minutes": 0, "triggered_patterns": ["POOL_TRAFFIC", "OUTBOUND_DOMINANT", "MANY_EXTERNAL"], "verdict_from_gating": "SUSPICIOUS"}
    network_anomaly scores.final: 60.0
  [PASS] network_anomaly → network alert 존재 or OBSERVE 이상  | alerts=['POOL_TRAFFIC', 'OUTBOUND_DOMINANT', 'MANY_EXTERNAL'] verdict=SUSPICIOUS

=== Stage 2b: retrieval pool warm-up (peer segments) ===
    peer snapshots ingested: 36 / 36
  [PASS] peer warmup ≥ peer_count×12

=== Stage 3a: 테스트 PC 등록 (pc_info + 해시된 API key) ===
  [PASS] pc_info upsert PC-E2E-001  | rc=0 err=

=== Stage 3b: Spring /api/metrics → ML → DB 적재 시나리오 ===
  [PASS] spring_normal: 202 accepted  | code=202 body={'id': 2485, 'pcId': 'PC-E2E-001', 'status': 'accepted'}
  [PASS] spring_gpu_mining: 202 accepted  | code=202 body={'id': 2486, 'pcId': 'PC-E2E-001', 'status': 'accepted'}
  [PASS] spring_cpu_mining: 202 accepted  | code=202 body={'id': 2487, 'pcId': 'PC-E2E-001', 'status': 'accepted'}

=== Stage 4: DB 상태 검증 ===
    metrics_history rows: 3
    anomaly_history rows: 1
    ai_judgment_history rows: 1
  [PASS] metrics_history ≥ 3
  [PASS] anomaly_history ≥ 1 (HIGH/SUSPICIOUS 시나리오 1건+)
    anomaly_history latest: HIGH / HIGH_RISK | final=100.0 | cat={"system_abnormal": false, "network_abnormal": true, "resource_abnormal": true, "sustained_minutes": 1, "triggered_patterns": ["CONFIRMED_MINING", "CPU_GPU_IMBALANCE", "HIGH_CPU", "MANY_EXTERNAL", "STRAT
    scores JSONB 키 존재 (cat/final/retr): true,true,false
  [PASS] scores.category_signals 키 존재
  [PASS] scores.final 키 존재
    INFO: retrieval_evidence 키 없음 — peer warmup 부족 또는 segment store 미적재

=== Stage 5: Grafana 데이터소스/대시보드 점검 ===
  [PASS] GET /api/datasources 200  | code=200
  [PASS] Postgres datasource 존재  | types=['grafana-postgresql-datasource', 'prometheus']
    postgres datasource: name=RADA-Postgres uid=rada_pg
  [PASS] Postgres datasource health OK  | code=200 body={'message': 'Database Connection OK', 'status': 'OK'}
    provisioned dashboards: ["60주년 808 실습실 자원 모니터링", "RADA — PC Detail", "실습실 PC 자원 이상탐지 시스템 | LAB-01 관제"]
  [PASS] 최소 1개 대시보드 프로비저닝

=== Teardown: 테스트 PC/피어 PC 의 적재 행 정리 ===
    deleted rows: 10


결과: PASS=26  FAIL=0