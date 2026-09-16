import numpy as np
import pytest

from chordrec.evaluate import (
    frame_intervals, read_lab, score, states_to_intervals,
    transitions_per_minute, write_lab,
)
from chordrec.vocab import parse_label

SR, N_FFT, HOP = 22050, 4096, 2048


def test_frame_intervals_are_centred_on_windows():
    """Frames describe the middle of their window, not its end.

    Using the end would shift every prediction late by half a window -- 93 ms
    at n_fft=4096 -- and cost WCSR that would get blamed on the algorithm.
    """
    iv = frame_intervals(3, SR, HOP, N_FFT)
    centres = iv.mean(axis=1)
    assert centres[0] == pytest.approx(N_FFT / 2 / SR, abs=1e-9)
    assert centres[1] == pytest.approx((HOP + N_FFT / 2) / SR, abs=1e-9)


def test_frame_intervals_tile_exactly():
    """Bit-identical boundaries, not merely close ones.

    mir_eval compares interval endpoints exactly. Endpoints computed
    independently drift by a few times 1e-15, which makes a minority of pairs
    overlap and makes mir_eval reject the entire annotation. np.allclose passes
    happily on that, which is precisely why this asserts equality.
    """
    iv = frame_intervals(500, SR, HOP, N_FFT)
    assert np.all(iv[1:, 0] == iv[:-1, 1])
    assert np.all(iv[:, 1] > iv[:, 0])


def test_intervals_survive_mir_eval_exact_check():
    """The regression that matters: scoring must not raise on real interval counts."""
    mir_eval = pytest.importorskip("mir_eval")
    iv = frame_intervals(500, SR, HOP, N_FFT)
    states = np.arange(500) % 5
    est_iv, est_lb = states_to_intervals(states, iv)
    ref_iv = np.array([[0.0, float(iv[-1, 1])]])
    assert score(ref_iv, ["C:maj"], est_iv, est_lb)["wcsr"] >= 0.0


def test_states_collapse_into_runs():
    states = np.array([0, 0, 0, 9, 9, 24, 24, 0])
    iv = frame_intervals(len(states), SR, HOP, N_FFT)
    out_iv, labels = states_to_intervals(states, iv)
    assert labels == ["C:maj", "A:maj", "N", "C:maj"]
    assert len(out_iv) == 4
    assert out_iv[0, 0] == iv[0, 0]
    assert out_iv[-1, 1] == iv[-1, 1]


def test_empty_input():
    iv, labels = states_to_intervals(np.array([], dtype=int), np.zeros((0, 2)))
    assert len(iv) == 0 and labels == []


def test_perfect_prediction_scores_one():
    iv = np.array([[0.0, 2.0], [2.0, 4.0]])
    labels = ["C:maj", "A:min"]
    assert score(iv, labels, iv, labels)["wcsr"] == pytest.approx(1.0)


def test_completely_wrong_prediction_scores_zero():
    iv = np.array([[0.0, 2.0], [2.0, 4.0]])
    assert score(iv, ["C:maj", "A:min"], iv, ["F#:min", "D#:maj"])["wcsr"] == pytest.approx(0.0)


def test_wcsr_is_duration_weighted_not_chord_counted():
    """Getting the long chord right must beat getting three short ones right.

    This is the whole point of the W in WCSR, and it is the property that
    stops a flickering detector looking good.
    """
    ref_iv = np.array([[0.0, 9.0], [9.0, 10.0], [10.0, 11.0], [11.0, 12.0]])
    ref = ["C:maj", "D:min", "E:min", "F:maj"]
    long_right = score(ref_iv, ref, ref_iv, ["C:maj", "A:min", "A:min", "A:min"])["wcsr"]
    short_right = score(ref_iv, ref, ref_iv, ["A:min", "D:min", "E:min", "F:maj"])["wcsr"]
    assert long_right == pytest.approx(0.75)
    assert short_right == pytest.approx(0.25)
    assert long_right > short_right


def test_lab_round_trip(tmp_path):
    iv = np.array([[0.0, 1.5], [1.5, 3.0]])
    labels = ["C:maj", "A:min"]
    p = tmp_path / "x.lab"
    write_lab(p, iv, labels)
    back_iv, back_labels = read_lab(p)
    assert np.allclose(back_iv, iv)
    assert back_labels == labels


def test_lab_reduces_extended_chords_on_read(tmp_path):
    p = tmp_path / "x.lab"
    p.write_text("0.0\t1.0\tG:7\n1.0\t2.0\tC:min7\n2.0\t3.0\tN\n")
    _, labels = read_lab(p)
    assert labels == ["G:maj", "C:min", "N"]


def test_transitions_per_minute():
    iv = np.array([[0.0, 30.0], [30.0, 60.0]])
    assert transitions_per_minute(iv) == pytest.approx(2.0)
