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
    missing_states: tuple[SalahState, ...] = ()
    completed_rakats: int = 0
    current_rakat: int = 1
    current_rakat_missing_states: tuple[SalahState, ...] = ()
    missing_state_entries: tuple["MissingStateEntry", ...] = ()


@dataclass(frozen=True)
class MissingStateEntry:
    """A missing state tagged with its rakah number."""

    rakat_number: int
    state: SalahState


class SalahSequenceTracker:
    """Advance sequence index on detected state changes."""

    def __init__(self, profile: SalahSequenceProfile) -> None:
        self.profile = profile
        self._index = 0
        self._per_rakat_sequence = _split_sequence_into_rakats(profile.state_sequence)

        self._completed_rakats = 0
        self._current_rakat = 1
        self._mistake_found = False

        self._active_rakat_index = 0
        self._active_rakat_match_index = 0
        self._active_rakat_missing: list[SalahState] = []
        self._last_detected_state: SalahState | None = None

    def on_state_change(self, detected_state: SalahState) -> SequenceProgress:
        normalized_state = _normalize_runtime_state(detected_state)
        missing_entries: list[MissingStateEntry] = []
        total_states = len(self.profile.state_sequence)
        if self._index < total_states:
            expected = self.profile.state_sequence[self._index]
            if _states_match(expected, normalized_state):
                self._index += 1
            else:
                # If we detect a later valid state, treat skipped states as missing.
                forward_match_index: int | None = None
                for lookahead_index in range(self._index + 1, total_states):
                    lookahead_expected = self.profile.state_sequence[lookahead_index]
                    if _states_match(lookahead_expected, normalized_state):
                        forward_match_index = lookahead_index
                        break

                if forward_match_index is not None:
                    missing_states = tuple(
                        self.profile.state_sequence[self._index : forward_match_index]
                    )
                    missing_entries.extend(
                        _to_missing_entries(
                            missing_states,
                            rakat_number=self._active_rakat_index + 1,
                        )
                    )
                    self._index = forward_match_index + 1

        rakat_boundary_missing_entries = self._handle_rakat_boundary_if_needed(normalized_state)
        local_missing_entries = self._consume_within_active_rakat(normalized_state)
        merged_entries = _unique_ordered_entries(
            [*missing_entries, *rakat_boundary_missing_entries, *local_missing_entries]
        )

        self._last_detected_state = normalized_state
        return self.progress(missing_state_entries=tuple(merged_entries))

    def progress(
        self,
        missing_state_entries: tuple[MissingStateEntry, ...] = (),
    ) -> SequenceProgress:
        next_state = None
        if self._index < len(self.profile.state_sequence):
            next_state = self.profile.state_sequence[self._index]

        missing_states = tuple(entry.state for entry in missing_state_entries)
        return SequenceProgress(
            current_index=self._index,
            total_states=len(self.profile.state_sequence),
            next_expected_state=next_state,
            matched_states=self._index,
            missing_states=missing_states,
            completed_rakats=self._completed_rakats,
            current_rakat=self._current_rakat,
            current_rakat_missing_states=tuple(self._active_rakat_missing),
            missing_state_entries=missing_state_entries,
        )

    def _handle_rakat_boundary_if_needed(
        self,
        detected_state: SalahState,
    ) -> list[MissingStateEntry]:
        # Standing after sajda-like phase means we moved to the next rakah.
        is_qiyam_start = detected_state == SalahState.QIYAM
        stood_after_sajda = self._last_detected_state in {
            SalahState.SUJUD_1,
            SalahState.JALSA,
            SalahState.SUJUD_2,
            SalahState.TASHAHHUD,
        }
        if is_qiyam_start and stood_after_sajda:
            return self._finalize_active_rakat_and_advance()

        # Final rakah can finish on TASHAHHUD (without standing again).
        if (
            detected_state == SalahState.TASHAHHUD
            and self._last_detected_state == SalahState.SUJUD_2
            and self._active_rakat_index == len(self._per_rakat_sequence) - 1
        ):
            return self._finalize_active_rakat_and_advance()

        return []

    def _consume_within_active_rakat(
        self,
        detected_state: SalahState,
    ) -> list[MissingStateEntry]:
        if not self._per_rakat_sequence:
            return []

        active_sequence = self._per_rakat_sequence[min(self._active_rakat_index, len(self._per_rakat_sequence) - 1)]
        if self._active_rakat_match_index >= len(active_sequence):
            return []

        expected = active_sequence[self._active_rakat_match_index]
        if _states_match(expected, detected_state):
            self._active_rakat_match_index += 1
            return []

        forward_match_index: int | None = None
        for idx in range(self._active_rakat_match_index + 1, len(active_sequence)):
            if _states_match(active_sequence[idx], detected_state):
                forward_match_index = idx
                break

        if forward_match_index is None:
            return []

        missing = active_sequence[self._active_rakat_match_index : forward_match_index]
        for state in missing:
            if state not in self._active_rakat_missing:
                self._active_rakat_missing.append(state)
        self._active_rakat_match_index = forward_match_index + 1
        return _to_missing_entries(
            missing,
            rakat_number=self._active_rakat_index + 1,
        )

    def _finalize_active_rakat_and_advance(self) -> list[MissingStateEntry]:
        if not self._per_rakat_sequence:
            return []

        active_rakat_number = self._active_rakat_index + 1
        active_sequence = self._per_rakat_sequence[min(self._active_rakat_index, len(self._per_rakat_sequence) - 1)]
        remaining = active_sequence[self._active_rakat_match_index :]
        merged_missing = _unique_ordered([*self._active_rakat_missing, *remaining])

        if merged_missing:
            self._mistake_found = True
        elif not self._mistake_found:
            # Completed rakats remain "strict": freeze once the first mistake happens.
            self._completed_rakats = self._active_rakat_index + 1

        if self._active_rakat_index < len(self._per_rakat_sequence) - 1:
            self._active_rakat_index += 1
            self._current_rakat = self._active_rakat_index + 1
        else:
            # Stay on final rakah number once the configured prayer is fully traversed.
            self._current_rakat = len(self._per_rakat_sequence)

        self._active_rakat_match_index = 0
        self._active_rakat_missing = []
        return _to_missing_entries(
            merged_missing,
            rakat_number=active_rakat_number,
        )


def _states_match(expected: SalahState, detected: SalahState) -> bool:
    """Match runtime FSM states against sequence states with practical equivalence rules."""
    if expected == detected:
        return True

    return False


def _normalize_runtime_state(state: SalahState) -> SalahState:
    if state == SalahState.QIYAM_NEXT:
        return SalahState.QIYAM
    return state


def _split_sequence_into_rakats(sequence: list[SalahState]) -> list[list[SalahState]]:
    if not sequence:
        return []

    rakats: list[list[SalahState]] = []
    current: list[SalahState] = []

    for state in sequence:
        if state == SalahState.QIYAM and current:
            rakats.append(current)
            current = [state]
        else:
            current.append(state)

    if current:
        rakats.append(current)
    return rakats


def _unique_ordered(states: list[SalahState]) -> list[SalahState]:
    output: list[SalahState] = []
    for state in states:
        if state not in output:
            output.append(state)
    return output


def _to_missing_entries(
    states: tuple[SalahState, ...] | list[SalahState],
    *,
    rakat_number: int,
) -> list[MissingStateEntry]:
    return [MissingStateEntry(rakat_number=rakat_number, state=state) for state in states]


def _unique_ordered_entries(entries: list[MissingStateEntry]) -> list[MissingStateEntry]:
    output: list[MissingStateEntry] = []
    for entry in entries:
        if not any(
            existing.rakat_number == entry.rakat_number and existing.state == entry.state
            for existing in output
        ):
            output.append(entry)
    return output
