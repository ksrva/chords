"""Tests for key detection and transposition.

The interesting failures here are musical, not numerical: a transposer that
changes mode, wraps an octave the wrong way, or silently prefers +7 over -5
produces output that parses fine and is wrong to play.
"""

from __future__ import annotations

import numpy as np
import pytest

from chordrec.transpose import (
    Key,
    best_shift,
    detect_key,
    parse_chord_symbol,
    parse_key,
    suggest,
    transpose_label,
    transpose_progression,
)
from chordrec.vocab import NO_CHORD_INDEX, parse_label, state_label


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C", "C:maj"), ("Am", "A:min"), ("F#m", "F#:min"), ("Bb", "A#:maj"),
        ("Cmaj7", "C:maj"), ("Am7", "A:min"), ("G7", "G:maj"),
        ("C/G", "C:maj"), ("Dm/F", "D:min"), ("c", "C:maj"),
        ("Bdim", "B:min"), ("A:min", "A:min"), ("N", "N"), ("-", "N"),
    ],
)
def test_lead_sheet_spellings(text, expected):
    assert state_label(parse_chord_symbol(text)) == expected


def test_unknown_root_is_rejected_not_guessed():
    with pytest.raises(ValueError):
        parse_chord_symbol("H")


@pytest.mark.parametrize("text,pc", [("D", 2), ("d", 2), ("Eb", 3), ("F#", 6), ("Dm", 2)])
def test_parse_key_ignores_mode(text, pc):
    assert parse_key(text) == pc


def test_transposition_preserves_mode():
    """The failure that sounds worst and reads fine."""
    for label in ("C:maj", "A:min", "F#:min", "B:maj"):
        for shift in range(-12, 13):
            moved = transpose_label(label, shift)
            assert moved.split(":")[1] == label.split(":")[1]


def test_transposition_is_invertible():
    labels = ["C:maj", "A:min", "F:maj", "G:maj"]
    for shift in range(-11, 12):
        there = transpose_progression(labels, shift)
        assert transpose_progression(there, -shift) == labels


def test_octave_wraps_to_the_same_chord():
    assert transpose_label("C:maj", 12) == "C:maj"
    assert transpose_label("A:min", -12) == "A:min"


def test_no_chord_is_never_transposed():
    assert transpose_label("N", 5) == "N"
    assert parse_label(transpose_label("N", 5)) == NO_CHORD_INDEX


def test_shift_takes_the_short_way_round():
    """C to G is -5, not +7. Both land on G; only one is what you play."""
    assert best_shift(0, [7]) == -5
    assert best_shift(7, [0]) == 5


def test_shift_stays_within_half_an_octave():
    for from_tonic in range(12):
        for target in range(12):
            assert -6 <= best_shift(from_tonic, [target]) <= 5


def test_shift_picks_the_nearest_comfortable_key():
    assert best_shift(0, [2, 7]) == 2          # D is +2, G is -5
    assert best_shift(0, [5, 11]) == -1        # B is -1, F is +5


def test_tritone_tie_resolves_downward():
    """Equidistant either way. Down is the safe guess for a voice."""
    assert best_shift(0, [6]) == -6


def test_shift_of_zero_when_already_comfortable():
    assert best_shift(2, [2, 7]) == 0


def test_no_comfortable_keys_is_an_error():
    with pytest.raises(ValueError):
        best_shift(0, [])


def test_detects_a_plain_major_progression():
    key, margin = detect_key(["C:maj", "A:min", "F:maj", "G:maj"])
    assert key == Key(0, "maj")
    assert margin > 0.1


def test_detects_a_minor_progression():
    key, _ = detect_key(["A:min", "D:min", "E:maj", "A:min"])
    assert key == Key(9, "min")


def test_detects_a_transposed_progression_consistently():
    """Whatever the key detector believes, it must believe the same thing
    about the same progression moved. A detector that is key-dependent is
    detecting something other than key."""
    labels = ["C:maj", "A:min", "F:maj", "G:maj"]
    base, _ = detect_key(labels)
    for shift in range(-6, 6):
        moved, _ = detect_key(transpose_progression(labels, shift))
        assert moved.mode == base.mode
        assert moved.tonic == (base.tonic + shift) % 12


def test_duration_weighting_changes_the_verdict():
    """A long tonic outweighs passing chords. Without weighting, a chord that
    sounds for half a bar counts as much as one that holds for four."""
    labels = ["C:maj", "G:maj", "G:maj", "G:maj"]
    unweighted, _ = detect_key(labels)
    weighted, _ = detect_key(labels, durations=[30.0, 1.0, 1.0, 1.0])
    assert unweighted == Key(7, "maj")
    assert weighted == Key(0, "maj")


def test_durations_must_match_labels():
    with pytest.raises(ValueError):
        detect_key(["C:maj", "G:maj"], durations=[1.0])


def test_no_chord_contributes_no_evidence():
    with_n = detect_key(["C:maj", "N", "F:maj", "G:maj"])[0]
    without = detect_key(["C:maj", "F:maj", "G:maj"])[0]
    assert with_n == without


def test_empty_progression_reports_no_confidence():
    _, margin = detect_key([])
    assert margin == 0.0


def test_suggest_end_to_end():
    result = suggest(["C:maj", "A:min", "F:maj", "G:maj"], comfortable=[2])
    assert result["from_key"] == Key(0, "maj")
    assert result["to_key"] == Key(2, "maj")
    assert result["semitones"] == 2
    assert result["labels"] == ["D:maj", "B:min", "G:maj", "A:maj"]


def test_suggest_leaves_a_comfortable_song_alone():
    result = suggest(["D:maj", "G:maj", "A:maj"], comfortable=[2, 7])
    assert result["semitones"] == 0
    assert result["labels"] == ["D:maj", "G:maj", "A:maj"]


def test_suggest_accepts_durations_from_recognize():
    intervals = np.array([[0.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    durations = (intervals[:, 1] - intervals[:, 0]).tolist()
    result = suggest(["C:maj", "G:maj", "G:maj"], comfortable=[2], durations=durations)
    assert result["from_key"] == Key(0, "maj")


# --- display: spelling and summaries -----------------------------------------
# vocab spells everything with sharps so the detector's output is canonical.
# These cover the boundary where that becomes something a musician reads.

from chordrec.transpose import chord_totals  # noqa: E402


@pytest.mark.parametrize(
    "key,pc,expected",
    [
        (Key(8, "maj"), 8, "Ab"),      # Ab major, not G#
        (Key(8, "maj"), 10, "Bb"),
        (Key(7, "maj"), 6, "F#"),      # G major is a sharp key
        (Key(0, "maj"), 0, "C"),
        (Key(5, "min"), 8, "Ab"),      # F minor takes flats
        (Key(4, "min"), 6, "F#"),      # E minor takes sharps
    ],
)
def test_spelling_follows_the_key_signature(key, pc, expected):
    assert key.spell(pc) == expected


@pytest.mark.parametrize(
    "key,label,expected",
    [
        (Key(8, "maj"), "G#:maj", "Ab"),
        (Key(8, "maj"), "F:min", "Fm"),
        (Key(7, "maj"), "E:min", "Em"),
        (Key(0, "maj"), "N", "-"),
    ],
)
def test_chords_render_as_a_musician_writes_them(key, label, expected):
    assert key.chord(parse_label(label)) == expected


def test_display_transposition_preserves_mode():
    """Regression: the summary shifted the state index directly, so Cm down a
    semitone rendered as B rather than Bm -- states run [0-11 maj, 12-23 min]
    and plain addition walks off the end of major into minor."""
    source, target = Key(8, "maj"), Key(7, "maj")
    moved = transpose_label("C:min", -1)
    assert target.chord(parse_label(moved)) == "Bm"


def test_chord_totals_sorts_by_duration():
    labels = ["C:maj", "G:maj", "C:maj"]
    totals = chord_totals(labels, [1.0, 5.0, 1.0])
    assert totals[0] == ("G:maj", 5.0)
    assert totals[1] == ("C:maj", 2.0)


def test_chord_totals_counts_occurrences_without_durations():
    totals = dict(chord_totals(["C:maj", "C:maj", "G:maj"]))
    assert totals["C:maj"] == 2.0 and totals["G:maj"] == 1.0
