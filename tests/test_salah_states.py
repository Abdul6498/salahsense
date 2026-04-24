"""Tests for runtime Salah state mapping from config."""

from pathlib import Path

from salahsense.config.salah_states import SalahStateCatalog
from salahsense.counting.rakat_counter import RakatStage
from salahsense.state_machine import MovementDirection, VerticalLevel


def test_resolve_qauma_after_ruku() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahStateCatalog.from_json(str(repo_root / "config" / "salah_states.json"))

    # Enter ruku first.
    state1 = catalog.resolve_runtime_state(
        level=VerticalLevel.MID,
        direction=MovementDirection.GOING_DOWN,
        stage=RakatStage.WAIT_FIRST_LOW,
    )
    # Then rise to high before first sujud -> Qauma.
    state2 = catalog.resolve_runtime_state(
        level=VerticalLevel.HIGH,
        direction=MovementDirection.GOING_UP,
        stage=RakatStage.WAIT_FIRST_LOW,
    )

    assert state1.english == "Bowing"
    assert state2.english == "Rising from Bowing"
