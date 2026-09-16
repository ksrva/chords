import pytest

from chordrec.vocab import (
    N_STATES, NO_CHORD_INDEX, NO_CHORD_LABEL,
    all_labels, chord_pitch_classes, parse_label, reduce_label, state_label,
)


def test_state_ordering_is_pinned():
    """This ordering is shared with the transition matrix and the Rust core.

    If this test fails, exported parameter files from before the change are
    silently wrong -- the detector will run and report confident nonsense.
    """
    assert N_STATES == 25
    assert state_label(0) == "C:maj"
    assert state_label(11) == "B:maj"
    assert state_label(12) == "C:min"
    assert state_label(23) == "B:min"
    assert state_label(24) == NO_CHORD_LABEL


def test_labels_round_trip():
    for i in range(N_STATES):
        assert parse_label(state_label(i)) == i


def test_labels_are_unique():
    assert len(set(all_labels())) == N_STATES


@pytest.mark.parametrize("label,expected", [
    ("C", "C:maj"), ("Db:maj", "C#:maj"), ("Bb:min", "A#:min"),
    ("G:7", "G:maj"), ("C:min7", "C:min"), ("F:maj7", "F:maj"),
    ("A:dim", "A:min"), ("C:maj/3", "C:maj"), ("N", "N"), ("X", "N"),
])
def test_reduction_to_majmin(label, expected):
    assert state_label(reduce_label(label)) == expected


def test_parse_refuses_what_it_cannot_represent():
    """Out-of-vocabulary labels must raise, not guess.

    Reduction is available and explicit; a silent fallback here would hide
    evaluation bugs behind plausible-looking chords.
    """
    with pytest.raises(ValueError):
        parse_label("G:7")
    with pytest.raises(ValueError):
        parse_label("H:maj")


def test_chord_pitch_classes():
    assert chord_pitch_classes(0) == {0, 4, 7}       # C E G
    assert chord_pitch_classes(12) == {0, 3, 7}      # C Eb G
    assert chord_pitch_classes(9) == {9, 1, 4}       # A C# E
    assert chord_pitch_classes(NO_CHORD_INDEX) == set()


def test_mir_eval_accepts_our_labels():
    """Our spelling has to survive the scorer, or every number is wrong."""
    mir_eval = pytest.importorskip("mir_eval")
    for label in all_labels():
        mir_eval.chord.encode(label)
