"""DELETE /history/{pc_id} — 히스토리 초기화."""
from fastapi import APIRouter

from ..storage import pc_history_store, score_history_store, model_store
from ..retrieval import clear_pc as retrieval_clear_pc

router = APIRouter()

# (pc_id, slot) 키 패턴 — config.CONTAMINATION 슬롯과 동일
_SLOTS = ("class", "free")


@router.delete("/history/{pc_id}")
def clear_history(pc_id: str):
    found = False

    if pc_id in pc_history_store.pc_history:
        pc_history_store.pc_history.pop(pc_id, None)
        found = True
    if pc_id in pc_history_store.pc_train_history:
        pc_history_store.pc_train_history.pop(pc_id, None)
        found = True
    if pc_id in pc_history_store.all_pc_latest:
        pc_history_store.all_pc_latest.pop(pc_id, None)
        found = True

    for slot in _SLOTS:
        if score_history_store.pc_score_history.pop((pc_id, slot), None) is not None:
            found = True
        if score_history_store.rule_score_history.pop((pc_id, slot), None) is not None:
            found = True

    if model_store.clear_pc(pc_id):
        found = True

    if retrieval_clear_pc(pc_id):
        found = True

    if found:
        return {"message": f"{pc_id} 히스토리 초기화 완료"}
    return {"message": f"{pc_id} 없음"}
