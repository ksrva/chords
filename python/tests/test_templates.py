import numpy as np
import pytest

from chordrec.chroma import chroma
from chordrec.signals import triad
from chordrec.templates import (
    binary_templates, emission_scores, log_emission_probabilities, predict_states,
)
from chordrec.vocab import NO_CHORD_INDEX, N_STATES, chord_pitch_classes, state_label

SR, N_FFT, HOP = 22050, 4096, 2048


def test_templates_shape_and_normalization():
    """Chord rows are unit length; the no-chord row is zeros, not uniform.

    A uniform no-chord row loses to nothing and beats everything -- measured at
    0.900 against C:maj's 0.706 on a synthesised triad -- because cosine
    against twelve nonzeros is not comparable to cosine against three. N is a
    threshold instead, supplied in emission_scores.
    """
    T = binary_templates()
    assert T.shape == (N_STATES, 12)
    assert np.allclose(np.linalg.norm(T[:NO_CHORD_INDEX], axis=1), 1.0)
    assert np.all(T[NO_CHORD_INDEX] == 0.0)


def test_templates_mark_the_right_notes():
    T = binary_templates()
    for state in range(N_STATES - 1):
        nonzero = set(np.flatnonzero(T[state]).tolist())
        assert nonzero == chord_pitch_classes(state), state_label(state)
    assert np.all(T[NO_CHORD_INDEX] == 0.0)       # no chord has no notes


def test_perfect_chroma_scores_its_own_template_highest():
    """A template fed to itself must win. If this fails nothing else matters."""
    T = binary_templates()
    scores = emission_scores(T[:-1])          # skip the uniform no-chord row
    assert np.all(scores.argmax(axis=1) == np.arange(N_STATES - 1))
    assert np.allclose(np.max(scores, axis=1), 1.0)


@pytest.mark.parametrize("root,quality", [
    ("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj"), ("D", "min"), ("E", "min"),
])
def test_real_triads_are_identified(root, quality):
    """End to end on synthesised audio: the right chord wins outright."""
    X = chroma(triad(root, quality, 2.0, SR), SR, n_fft=N_FFT, hop=HOP)
    states = predict_states(X)
    expected = f"{root}:{quality}"
    got = [state_label(s) for s in states]
    assert all(g == expected for g in got), f"expected {expected}, got {set(got)}"


def test_silence_is_no_chord():
    """Zero chroma must not fabricate a chord, at any threshold."""
    X = chroma(np.zeros(SR), SR, n_fft=N_FFT, hop=HOP)
    for threshold in (0.0, 0.1, 0.9):
        scores = emission_scores(X, no_chord_threshold=threshold)
        assert np.all(scores.argmax(axis=1) == NO_CHORD_INDEX)
        assert np.all(np.isneginf(scores[:, :NO_CHORD_INDEX]))


def test_threshold_controls_willingness_to_name_a_chord():
    """Raising the threshold makes the detector abstain rather than guess."""
    X = chroma(triad("C", "maj", 2.0, SR), SR, n_fft=N_FFT, hop=HOP)
    assert np.all(predict_states(X, no_chord_threshold=0.0) != NO_CHORD_INDEX)
    assert np.all(predict_states(X, no_chord_threshold=0.99) == NO_CHORD_INDEX)


def test_relative_minor_is_the_nearest_confusion():
    """C:maj and A:min share two of three notes, so they should be adjacent
    in score -- and the correct one still has to win.

    Worth pinning: relative-major/minor confusion is the classic template
    failure, and a change that makes it worse should be visible here.
    """
    X = chroma(triad("C", "maj", 2.0, SR), SR, n_fft=N_FFT, hop=HOP)
    s = emission_scores(X).mean(axis=0)
    ranking = np.argsort(s)[::-1]
    assert state_label(ranking[0]) == "C:maj"
    assert state_label(ranking[1]) in ("A:min", "E:min")


def test_log_probabilities_are_normalized():
    X = chroma(triad("C", "maj", 2.0, SR), SR, n_fft=N_FFT, hop=HOP)
    logp = log_emission_probabilities(emission_scores(X))
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0)
    assert np.all(logp <= 0.0)


def test_temperature_controls_sharpness():
    """Low temperature approaches argmax; high approaches uniform."""
    X = chroma(triad("C", "maj", 2.0, SR), SR, n_fft=N_FFT, hop=HOP)
    s = emission_scores(X)
    sharp = np.exp(log_emission_probabilities(s, temperature=0.01)).max(axis=1).mean()
    flat = np.exp(log_emission_probabilities(s, temperature=100.0)).max(axis=1).mean()
    assert sharp > 0.9
    assert flat == pytest.approx(1.0 / N_STATES, abs=0.01)
