"""Track runtime progress against normalized Salah sequence."""

from __future__ import annotations

from dataclasses import dataclass

from salahsense.config.salah_sequences import SalahSequenceProfile
from salahsense.state_machine import SalahState


@dataclass(frozen=True)
class SequenceProgress:
    """Current sequence progress snapshot."""

    current_index: int
    total_states: int
    next_expected_state: SalahState | None
    matched_states: int


class SalahSequenceTracker:
    """Advance sequence index on detected state changes."""

    def __init__(self, profile: SalahSequenceProfile) -> None:
        self.profile = profile
        self._index = 0

    def on_state_change(self, detected_state: SalahState) -> SequenceProgress:
        if self._index < len(self.profile.state_sequence):
            expected = self.profile.state_sequence[self._index]
            if _states_match(expected, detected_state):
                self._index += 1

        return self.progress()

    def progress(self) -> SequenceProgress:
        next_state = None
        if self._index < len(self.profile.state_sequence):
            next_state = self.profile.state_sequence[self._index]

        return SequenceProgress(
            current_index=self._index,
            total_states=len(self.profile.state_sequence),
            next_expected_state=next_state,
            matched_states=self._index,
        )


def _states_match(expected: SalahState, detected: SalahState) -> bool:
    """Match runtime FSM states against sequence states with practical equivalence rules."""
    if expected == detected:
        return True

    # Sequence uses QIYAM while runtime may emit QIYAM_NEXT between rakats.
    if expected == SalahState.QIYAM and detected == SalahState.QIYAM_NEXT:
        return True

    return False
