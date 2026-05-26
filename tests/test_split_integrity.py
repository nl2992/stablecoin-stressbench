"""Split integrity tests for event windows defined in configs/event_windows.yaml.

Verifies:
  1. No two event windows with *different* splits overlap in time.
  2. Every event window with an explicit 'split' key is assigned to exactly
     one of {train, validation, test}.
  3. Each window has a valid ISO-8601 start < end.
  4. Test and validation windows are non-overlapping with each other and with
     all train windows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pytest
import yaml


_VALID_SPLITS = {"train", "validation", "test"}


def _load_events() -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "event_windows.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)["events"]


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


@pytest.fixture(scope="module")
def events() -> dict:
    return _load_events()


def test_event_windows_load(events: dict) -> None:
    """Sanity check: config loads and has at least one event."""
    assert isinstance(events, dict)
    assert len(events) >= 1


def test_all_splits_are_valid(events: dict) -> None:
    """Every explicit split label must be in {train, validation, test}."""
    for name, ev in events.items():
        split = ev.get("split")
        if split is not None:
            assert split in _VALID_SPLITS, (
                f"Event '{name}' has unknown split '{split}'. "
                f"Expected one of {_VALID_SPLITS}."
            )


def test_start_before_end(events: dict) -> None:
    """Each event window must have start < end."""
    for name, ev in events.items():
        if "start" not in ev or "end" not in ev:
            continue
        start = _parse_ts(ev["start"])
        end = _parse_ts(ev["end"])
        assert start < end, (
            f"Event '{name}': start={ev['start']} is not before end={ev['end']}"
        )


def test_no_cross_split_overlap(events: dict) -> None:
    """No two windows assigned to *different* splits may overlap in time.

    Train windows may overlap each other (consecutive calm periods are fine),
    but a train window must not overlap any validation or test window.
    """
    windowed = [
        (name, _parse_ts(ev["start"]), _parse_ts(ev["end"]), ev.get("split", "train"))
        for name, ev in events.items()
        if "start" in ev and "end" in ev
    ]

    for (n1, s1, e1, sp1), (n2, s2, e2, sp2) in combinations(windowed, 2):
        if sp1 == sp2:
            continue  # same split — overlaps are acceptable
        # Two intervals [s1, e1] and [s2, e2] overlap iff s1 < e2 and s2 < e1
        overlap = s1 < e2 and s2 < e1
        assert not overlap, (
            f"Cross-split overlap detected:\n"
            f"  '{n1}' ({sp1}): {s1} → {e1}\n"
            f"  '{n2}' ({sp2}): {s2} → {e2}\n"
            "Events from different splits must not overlap."
        )


def test_test_and_validation_non_overlapping(events: dict) -> None:
    """Test windows and validation windows must be strictly disjoint."""
    test_windows = [
        (_parse_ts(ev["start"]), _parse_ts(ev["end"]), name)
        for name, ev in events.items()
        if ev.get("split") == "test" and "start" in ev
    ]
    val_windows = [
        (_parse_ts(ev["start"]), _parse_ts(ev["end"]), name)
        for name, ev in events.items()
        if ev.get("split") == "validation" and "start" in ev
    ]

    for (ts, te, tn) in test_windows:
        for (vs, ve, vn) in val_windows:
            overlap = ts < ve and vs < te
            assert not overlap, (
                f"Test event '{tn}' ({ts} → {te}) overlaps "
                f"validation event '{vn}' ({vs} → {ve})."
            )


def test_known_splits_present(events: dict) -> None:
    """The config must define at least one train, one validation, and one test window."""
    splits_found = {ev.get("split") for ev in events.values()}
    for required in ("train", "validation", "test"):
        assert required in splits_found, (
            f"No event window with split='{required}' found in event_windows.yaml. "
            "All three splits are required for the benchmark."
        )
