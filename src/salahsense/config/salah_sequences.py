"""Load and normalize Salah sequence definitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from salahsense.state_machine import SalahState


@dataclass(frozen=True)
class SalahSequenceProfile:
    """Normalized sequence profile for one prayer type."""

    profile_key: str
    profile_name: str
    expected_rakats: int
    state_sequence: list[SalahState]


class SalahSequenceCatalog:
    """Catalog for `config/salah_sequences.json`."""

    _STATE_MAP = {
        "takbirat al-ihram": SalahState.QIYAM,
        "qiyam": SalahState.QIYAM,
        "ruku": SalahState.RUKU,
        "i'tidal": SalahState.QAUMA,
        "i’tidal": SalahState.QAUMA,
        "sajda 1": SalahState.SUJUD_1,
        "jalsa": SalahState.JALSA,
        "sajda 2": SalahState.SUJUD_2,
        "tashahhud": SalahState.TASHAHHUD,
        "tashahhud & salawat": SalahState.TASHAHHUD,
        # Taslim is verbal/head-turning and not yet classified by current pose FSM.
        "taslim": None,
    }

    def __init__(self, profiles: dict[str, SalahSequenceProfile]) -> None:
        self._profiles = profiles

    @classmethod
    def from_json(cls, path: str) -> "SalahSequenceCatalog":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Salah sequences file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_profiles = data["salah_types"]
        profiles: dict[str, SalahSequenceProfile] = {}

        for profile_key, payload in raw_profiles.items():
            expected_rakats = _extract_expected_rakats(profile_key)
            raw_sequence = payload.get("sequence", [])
            normalized = _normalize_state_sequence(raw_sequence, cls._STATE_MAP)
            profiles[profile_key] = SalahSequenceProfile(
                profile_key=profile_key,
                profile_name=payload.get("name", profile_key),
                expected_rakats=expected_rakats,
                state_sequence=normalized,
            )

        return cls(profiles=profiles)

    def get_profile(self, profile_key: str) -> SalahSequenceProfile:
        if profile_key not in self._profiles:
            known = ", ".join(sorted(self._profiles.keys()))
            raise KeyError(f"Unknown salah profile '{profile_key}'. Known: {known}")
        return self._profiles[profile_key]

    def profile_keys(self) -> list[str]:
        return sorted(self._profiles.keys())


def _extract_expected_rakats(profile_key: str) -> int:
    # Examples: 2_rakat_prayer, 3_rakat_prayer, 4_rakat_prayer
    prefix = profile_key.split("_", 1)[0]
    return int(prefix)


def _normalize_state_sequence(raw_sequence: list, state_map: dict[str, SalahState | None]) -> list[SalahState]:
    normalized: list[SalahState] = []

    for rakah_item in raw_sequence:
        for _, states in rakah_item.items():
            for state_obj in states:
                raw_name = str(state_obj.get("state", "")).strip().lower()
                mapped = state_map.get(raw_name)
                if mapped is not None:
                    # Collapse adjacent duplicates so runtime progress can advance
                    # on state changes (e.g., Takbir + Qiyam both map to QIYAM).
                    if not normalized or normalized[-1] != mapped:
                        normalized.append(mapped)

    return normalized
