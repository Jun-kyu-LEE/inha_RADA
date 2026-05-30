"""학습/추론/CSV 에서 공통으로 쓰는 raw 컬럼 정의.

ML 피처(7 raw + 6 derived)와 룰 판정은 아래 11 개 raw 만 사용한다.
derived 는 ``training_adapter._compute_derived()`` 에서만 계산 — CSV 의
미리 계산된 derived 컬럼은 읽지 않고 import 시 제거한다.
"""

# 학습·추론·룰 공통 raw (11)
RAW_METRIC_COLS: list[str] = [
    "cpu_percent",
    "memory_percent",
    "gpu_percent",
    "gpu_vram_mb",
    "gpu_total_mb",
    "gpu_power_w",
    "inbound_mb",
    "outbound_mb",
    "external_packet_count",
    "disk_read_mb",
    "disk_write_mb",
]

# CSV 에 있어도 무시·제거 (코드에서 재계산하거나 미사용)
PRECOMPUTED_DERIVED_COLS: list[str] = [
    "cpu_gpu_ratio",
    "gpu_minus_cpu",
    "traffic_direction",
    "gpu_vram_complex",
    "ext_traffic_complex",
    "cpu_mining_index",
    "temp_efficiency",
    "foreign_per_process",
    "new_proc_outbound",
]

# 사용하지 않는 확장 raw (ml_server2 잔여 — import 시 제거)
UNUSED_RAW_COLS: list[str] = [
    "connection_foreign_count",
    "cpu_temp",
    "gpu_temp",
    "network_error_rate",
    "udp_packet_count",
    "process_count",
    "privileged_process_count",
    "new_process_count",
    "tensor_core_active",
]

OPTIONAL_LABEL_COLS: list[str] = ["scenario"]

DROP_ON_IMPORT_COLS: list[str] = PRECOMPUTED_DERIVED_COLS + UNUSED_RAW_COLS
