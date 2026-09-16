"""Tests for the smoothing stage.

The claims worth testing here are the ones that would fail silently: that the
causal window really is causal, that silence does not poison the average, and
that smoothing actually removes a spike rather than merely blurring it.
"""

from __future__ import annotations

import numpy as np
import pytest

from chordrec.recognize import (
    confidence,
    merge_short_segments,
    recognize,
    smooth_scores,
)
from chordrec.signals import progression, triad
from chordrec.vocab import N_STATES, NO_CHORD_INDEX

SR = 22050


def _scores(rows: list[list[float]]) -> np.ndarray:
    """Build an (n, 25) score matrix from a few leading columns."""
    S = np.zeros((len(rows), N_STATES))
    for i, row in enumerate(rows):
        S[i, : len(row)] = row
    return S


def test_window_of_one_is_identity():
    S = _scores([[0.9, 0.1], [0.2, 0.8]])
    assert np.allclose(smooth_scores(S, window=1), S)


def test_causal_window_uses_only_past_frames():
    """The point of causal=True. If a future frame can reach back and change
    an earlier one, the smoother cannot run live and the test must say so."""
    S = _scores([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    before = smooth_scores(S, window=3, causal=True)

    S_changed = S.copy()
    S_changed[2, 0] = -5.0                      # perturb only the last frame
    after = smooth_scores(S_changed, window=3, causal=True)

    assert np.allclose(before[:2], after[:2])   # earlier frames untouched
    assert not np.allclose(before[2], after[2])


def test_centred_window_does_look_ahead():
    """The complement of the test above: centred smoothing is non-causal, and
    that is precisely why it is not the default."""
    S = _scores([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    before = smooth_scores(S, window=3, causal=False)
    S_changed = S.copy()
    S_changed[2, 0] = -5.0
    after = smooth_scores(S_changed, window=3, causal=False)
    assert not np.allclose(before[1], after[1])


def test_single_frame_spike_is_outvoted():
    """One frame of a wrong chord between two right ones should not survive."""
    S = _scores([[0.9, 0.1], [0.9, 0.1], [0.1, 0.95], [0.9, 0.1], [0.9, 0.1]])
    assert S[2].argmax() == 1                       # unsmoothed, the spike wins
    smoothed = smooth_scores(S, window=5, causal=False)
    assert smoothed[2].argmax() == 0                # smoothed, it does not


def test_silent_frames_do_not_poison_the_average():
    """Silent frames are -inf everywhere but N. A naive mean would propagate
    that -inf across the whole window and hand every neighbour to no-chord."""
    S = _scores([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    S[1, :] = -np.inf
    S[1, NO_CHORD_INDEX] = 0.0

    out = smooth_scores(S, window=3, causal=True)
    assert np.isfinite(out[2, 0])
    assert out[2].argmax() == 0


def test_state_never_observed_stays_negative_infinity():
    S = np.full((3, N_STATES), -np.inf)
    S[:, 0] = 1.0
    out = smooth_scores(S, window=3)
    assert np.isinf(out[:, 1]).all()
    assert np.isfinite(out[:, 0]).all()


def test_smoothing_rejects_wrong_shape():
    with pytest.raises(ValueError):
        smooth_scores(np.zeros((4, 12)))


def test_short_segment_is_absorbed_by_longer_neighbour():
    intervals = np.array([[0.0, 1.0], [1.0, 1.05], [1.05, 2.0]])
    labels = ["C:maj", "F:maj", "C:maj"]
    out_iv, out_lab = merge_short_segments(intervals, labels, min_duration=0.15)
    assert out_lab == ["C:maj"]
    assert out_iv[0, 0] == 0.0 and out_iv[-1, 1] == 2.0


def test_merge_preserves_total_duration():
    intervals = np.array([[0.0, 0.5], [0.5, 0.55], [0.55, 1.2], [1.2, 1.22]])
    labels = ["C:maj", "F:maj", "G:maj", "A:min"]
    out_iv, _ = merge_short_segments(intervals, labels, min_duration=0.15)
    assert out_iv[0, 0] == pytest.approx(0.0)
    assert out_iv[-1, 1] == pytest.approx(1.22)


def test_merge_leaves_long_segments_alone():
    intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
    labels = ["C:maj", "F:maj"]
    out_iv, out_lab = merge_short_segments(intervals, labels, min_duration=0.15)
    assert out_lab == labels
    assert np.allclose(out_iv, intervals)


def test_merge_handles_empty():
    out_iv, out_lab = merge_short_segments(np.zeros((0, 2)), [])
    assert len(out_lab) == 0


@pytest.mark.parametrize(
    "chords",
    [
        [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")],
        [("D", "min"), ("G", "maj"), ("C", "maj")],
    ],
)
def test_recognizes_a_synthetic_progression(chords):
    y, _ = progression(chords, seconds_each=2.0, sr=SR)
    _, labels = recognize(y, SR)
    assert labels == [f"{root}:{quality}" for root, quality in chords]


def test_recognize_returns_one_segment_for_one_sustained_chord():
    y = triad("C", "maj", seconds=2.0, sr=SR)
    intervals, labels = recognize(y, SR)
    assert labels == ["C:maj"]
    assert len(intervals) == 1


def test_recognize_handles_audio_shorter_than_a_window():
    intervals, labels = recognize(np.zeros(100), SR)
    assert len(labels) == 0
    assert intervals.shape == (0, 2)


def test_smoothing_does_not_lose_accuracy_on_clean_audio():
    """Smoothing trades latency for stability. On already-stable audio it must
    not cost correctness, or the trade is not worth making anywhere."""
    chords = [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")]
    y, _ = progression(chords, seconds_each=2.0, sr=SR)
    expected = [f"{r}:{q}" for r, q in chords]
    assert recognize(y, SR, smoothing=1)[1] == expected
    assert recognize(y, SR, smoothing=5)[1] == expected


def test_uncompressed_chroma_gives_a_wider_margin():
    """The gamma finding, pinned down. Compression as currently ordered is
    applied to raw power and collapses the gap to the runner-up; if that ever
    stops being true this test should fail and the default revisited."""
    y = triad("C", "maj", seconds=2.0, sr=SR)
    assert confidence(y, SR, gamma=None) > 5 * confidence(y, SR, gamma=100.0)
