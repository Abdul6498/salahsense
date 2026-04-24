"""Tests for salah sequence catalog and runtime tracker."""

from pathlib import Path

from salahsense.config.salah_sequences import SalahSequenceCatalog
from salahsense.counting import SalahSequenceTracker
from salahsense.state_machine import SalahState


def test_sequence_catalog_loads_profiles() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))

    profile = catalog.get_profile("2_rakat_prayer")
    assert profile.expected_rakats == 2
    assert profile.profile_name
    assert len(profile.state_sequence) > 0


def test_sequence_tracker_advances_on_matching_states() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")

    tracker = SalahSequenceTracker(profile)
    first_expected = tracker.progress().next_expected_state
    assert first_expected is not None

    # Feed first 3 expected states and verify index grows.
    for i in range(3):
        tracker.on_state_change(profile.state_sequence[i])

    progress = tracker.progress()
    assert progress.current_index == 3
    assert progress.total_states == len(profile.state_sequence)


def test_sequence_tracker_accepts_qiyam_next_for_expected_qiyam() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Advance through first rakah states.
    for state in profile.state_sequence[:6]:
        tracker.on_state_change(state)

    # Next expected is QIYAM of rakah 2; runtime emits QIYAM_NEXT.
    before = tracker.progress().current_index
    tracker.on_state_change(SalahState.QIYAM_NEXT)
    after = tracker.progress().current_index

    assert after == before + 1

