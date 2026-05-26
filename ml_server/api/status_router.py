"""GET /status — 서버 상태."""
from fastapi import APIRouter

from ..storage import pc_history_store, model_store
from ..detector.global_degradation import detect_global_hw_degradation

router = APIRouter()


@router.get("/status")
def status():
    return {
        "status":           "running",
        "monitored_pcs":    list(pc_history_store.pc_history.keys()),
        "total_pcs":        len(pc_history_store.all_pc_latest),
        "pc_history_sizes": {pc_id: len(h)
                             for pc_id, h in pc_history_store.pc_history.items()},
        "trained_models":   model_store.list_trained(),
        "global_hw_latest": detect_global_hw_degradation(),
        "silent_fail_counters": {},
    }
