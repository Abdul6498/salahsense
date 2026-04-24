"""Load Salah state definitions and map FSM state names to readable labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from salahsense.state_machine import SalahState


@dataclass(frozen=True)
class SalahStateInfo:
    """Human-readable state metadata from config."""

    english: str
    arabic: str
    action: str


class SalahStateCatalog:
    """Lookup for Salah state labels loaded from `salah_states.json`."""

    def __init__(self, states: dict[str, SalahStateInfo]) -> None:
        self._states = states

    @classmethod
    def from_json(cls, path: str) -> "SalahStateCatalog":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Salah states file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_states = data["salah_guide"]["states"]
        states: dict[str, SalahStateInfo] = {}
        for item in raw_states:
            english = item["state_name_english"]
            states[english] = SalahStateInfo(
                english=english,
                arabic=item["state_name_arabic"],
                action=item["action"],
            )
        return cls(states=states)

    def resolve_from_fsm(self, state: SalahState) -> SalahStateInfo:
        """Map FSM state to Salah label from config."""
        if state == SalahState.QIYAM:
            return self._state_or_fallback("Standing")
        if state == SalahState.RUKU:
            return self._state_or_fallback("Bowing")
        if state == SalahState.QAUMA:
            return self._state_or_fallback("Rising from Bowing")
        if state in (SalahState.SUJUD_1, SalahState.SUJUD_2):
            return self._state_or_fallback("Prostration")
        if state == SalahState.JALSA:
            return self._state_or_fallback("Sitting")
        if state == SalahState.TASHAHHUD:
            return self._state_or_fallback("Sitting Testimony")
        if state == SalahState.QIYAM_NEXT:
            return SalahStateInfo(
                english="Standing (Next Rak'ah)",
                arabic="Qiyam (Next Rak'ah)",
                action="Standing for next unit after completed sujud cycle.",
            )

        return SalahStateInfo(
            english="Unknown/Transition",
            arabic="Unknown/Transition",
            action="Moving between prayer positions.",
        )

    def _state_or_fallback(self, english_name: str) -> SalahStateInfo:
        state = self._states.get(english_name)
        if state is not None:
            return state
        return SalahStateInfo(english=english_name, arabic=english_name, action="")
