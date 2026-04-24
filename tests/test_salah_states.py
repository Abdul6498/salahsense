"""Tests for Salah state mapping from FSM states."""

from pathlib import Path

from salahsense.config.salah_states import SalahStateCatalog
from salahsense.state_machine import SalahState


def test_resolve_from_fsm_states() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahStateCatalog.from_json(str(repo_root / "config" / "salah_states.json"))

    assert catalog.resolve_from_fsm(SalahState.RUKU).english == "Bowing"
    assert catalog.resolve_from_fsm(SalahState.QAUMA).english == "Rising from Bowing"
    assert catalog.resolve_from_fsm(SalahState.SUJUD_1).english == "Prostration"
    assert catalog.resolve_from_fsm(SalahState.JALSA).english == "Sitting"
