"""Load Salah state definitions and map runtime posture to readable Salah states."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from salahsense.counting.rakat_counter import RakatStage
from salahsense.state_machine import MovementDirection, VerticalLevel


@dataclass(frozen=True)
class SalahStateInfo:
    """Human-readable state metadata from config."""

    english: str
    arabic: str
    action: str


class SalahStateCatalog:
    """Lookup and runtime mapping for Salah states."""

    def __init__(self, states: dict[str, SalahStateInfo]) -> None:
        self._states = states
        self._seen_ruku_in_current_cycle = False

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

    def resolve_runtime_state(
        self,
        *,
        level: VerticalLevel,
        direction: MovementDirection,
        stage: RakatStage,
    ) -> SalahStateInfo:
        """Map current level/stage to best matching Salah state label."""
        if level == VerticalLevel.MID and stage == RakatStage.WAIT_FIRST_LOW:
            self._seen_ruku_in_current_cycle = True
            return self._state_or_fallback("Bowing")

        # In many real videos, Ruku can appear as UNKNOWN if thresholds are tight.
        # Keep a Bowing context before first sujud so Qauma can be recognized next.
        if level == VerticalLevel.UNKNOWN and stage == RakatStage.WAIT_FIRST_LOW:
            if direction in (MovementDirection.GOING_DOWN, MovementDirection.STILL):
                self._seen_ruku_in_current_cycle = True
                return self._state_or_fallback("Bowing")

        if level == VerticalLevel.LOW:
            self._seen_ruku_in_current_cycle = False
            return self._state_or_fallback("Prostration")

        if level == VerticalLevel.HIGH:
            # Qauma is a short stand after Ruku before moving to first sujud.
            if stage == RakatStage.WAIT_FIRST_LOW and self._seen_ruku_in_current_cycle:
                if direction in (MovementDirection.GOING_UP, MovementDirection.STILL):
                    return self._state_or_fallback("Rising from Bowing")
            return self._state_or_fallback("Standing")

        if stage in (RakatStage.WAIT_SECOND_LOW, RakatStage.WAIT_EXIT_SECOND_LOW):
            return self._state_or_fallback("Sitting")

        return SalahStateInfo(
            english="Transition",
            arabic="Transition",
            action="Moving between prayer positions.",
        )

    def _state_or_fallback(self, english_name: str) -> SalahStateInfo:
        state = self._states.get(english_name)
        if state is not None:
            return state
        return SalahStateInfo(english=english_name, arabic=english_name, action="")
