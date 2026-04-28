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


def test_sequence_tracker_reports_missing_state_on_forward_jump() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Reach JALSA of first rakah (SUJUD_2 is expected next).
    for state in profile.state_sequence[:5]:
        tracker.on_state_change(state)

    # User skips SUJUD_2 and stands up. Runtime often emits QIYAM_NEXT here.
    progress = tracker.on_state_change(SalahState.QIYAM_NEXT)

    assert progress.missing_states == (SalahState.SUJUD_2,)
    # Should move to after first state of rakah 2.
    assert progress.current_index == 7


def test_completed_rakat_freezes_after_first_mistake() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Mistake in first rakah: jump from SUJUD_1 directly to QIYAM_NEXT.
    for state in [
        SalahState.QIYAM,
        SalahState.RUKU,
        SalahState.QAUMA,
        SalahState.SUJUD_1,
        SalahState.QIYAM_NEXT,
    ]:
        progress = tracker.on_state_change(state)

    assert progress.current_rakat == 2
    assert progress.completed_rakats == 0

    # Even if second rakah is done cleanly, completed stays frozen at 0.
    for state in [
        SalahState.RUKU,
        SalahState.QAUMA,
        SalahState.SUJUD_1,
        SalahState.JALSA,
        SalahState.SUJUD_2,
        SalahState.TASHAHHUD,
    ]:
        progress = tracker.on_state_change(state)

    assert progress.completed_rakats == 0


def test_current_rakat_increments_when_standing_after_sajda() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    for state in [
        SalahState.QIYAM,
        SalahState.RUKU,
        SalahState.QAUMA,
        SalahState.SUJUD_1,
    ]:
        tracker.on_state_change(state)

    progress = tracker.on_state_change(SalahState.QIYAM_NEXT)
    assert progress.current_rakat == 2


def test_sequence_tracker_marks_ruku_and_qauma_missing_on_direct_sujud() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Enter rakat 2 with first rakat complete.
    for state in profile.state_sequence[:6]:
        tracker.on_state_change(state)
    tracker.on_state_change(SalahState.QIYAM_NEXT)

    # Direct sujud should report both missing RUKU and QAUMA.
    progress = tracker.on_state_change(SalahState.SUJUD_1)
    assert progress.missing_states == (SalahState.RUKU, SalahState.QAUMA)
    assert [
        (entry.rakat_number, entry.state)
        for entry in progress.missing_state_entries
    ] == [
        (2, SalahState.RUKU),
        (2, SalahState.QAUMA),
    ]


def test_sequence_tracker_freezes_after_done() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Complete full 2-rakat sequence.
    for state in profile.state_sequence:
        tracker.on_state_change(state)
    done_progress = tracker.progress()
    assert done_progress.next_expected_state is None
    completed_before = done_progress.completed_rakats

    # Extra post-prayer movement should not mutate counters.
    after = tracker.on_state_change(SalahState.QIYAM_NEXT)
    assert after.completed_rakats == completed_before


def test_completed_rakat_changes_on_sujud2_not_on_later_stand() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SalahSequenceCatalog.from_json(str(repo_root / "config" / "salah_sequences.json"))
    profile = catalog.get_profile("2_rakat_prayer")
    tracker = SalahSequenceTracker(profile)

    # Drive first rakah up to JALSA: completion must still be 0.
    for state in [
        SalahState.QIYAM,
        SalahState.RUKU,
        SalahState.QAUMA,
        SalahState.SUJUD_1,
        SalahState.JALSA,
    ]:
        progress = tracker.on_state_change(state)
    assert progress.completed_rakats == 0

    # SUJUD_2 marks completion.
    progress = tracker.on_state_change(SalahState.SUJUD_2)
    assert progress.completed_rakats == 1

    # Standing after that should not increment completion again.
    progress = tracker.on_state_change(SalahState.QIYAM_NEXT)
    assert progress.completed_rakats == 1
