"""Template emissions: how well does a chroma frame match each chord? (M2)

A chord template is the idealized chroma of that chord -- ones on the notes it
contains, zeros elsewhere -- and the score is the cosine similarity between a
frame and each template. Cosine is the right similarity here because it is
invariant to loudness: a quiet C major and a loud one point the same direction
in chroma space, and only the direction carries harmonic meaning.

Templates are derived, not learned. That makes them the honest baseline for S4,
where Gaussian emissions fitted per chord have to prove they beat them.

No-chord is the awkward state, and the obvious approach does not work. Giving
it a uniform template and letting it compete on cosine similarity looks
reasonable and is measurably wrong: a 12-nonzero template against a 3-nonzero
one is not a fair comparison, because any spread-out frame scores close to 1.0
on uniform. Measured on a synthesised C major triad, uniform beat the correct
template 0.900 to 0.706 -- the no-chord state would have won every frame of
every song.

So N is a *threshold*, not a template: the minimum similarity at which we are
willing to name a chord at all. Its row in the template matrix is zeros, which
is honest (no chord has no notes) and makes the dot product vanish, and the
threshold is substituted in afterwards. Default 0.0 means "only call it N when
there is no energy whatsoever"; raising it makes the detector more willing to
admit it does not know, and that trade belongs in the sweep.
"""

from __future__ import annotations

import numpy as np

from .vocab import (
    N_PITCH_CLASSES,
    N_STATES,
    NO_CHORD_INDEX,
    chord_pitch_classes,
)

DEFAULT_NO_CHORD_THRESHOLD = 0.0


def binary_templates() -> np.ndarray:
    """(25, 12) templates, rows indexed by state.

    Rows 0-23 are unit-length triads, so cosine similarity is a plain dot
    product. Row 24 (no-chord) is zeros: it contributes no similarity and its
    score is supplied by the threshold in `emission_scores`.
    """
    T = np.zeros((N_STATES, N_PITCH_CLASSES), dtype=np.float64)
    for state in range(N_STATES):
        for pc in chord_pitch_classes(state):        # empty for no-chord
            T[state, pc] = 1.0
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    return np.divide(T, norms, out=np.zeros_like(T), where=norms > 0)


def emission_scores(
    chroma_frames: np.ndarray,
    templates: np.ndarray | None = None,
    no_chord_threshold: float = DEFAULT_NO_CHORD_THRESHOLD,
) -> np.ndarray:
    """Cosine similarity of every frame against every chord, plus the N score.

    Returns (n_frames, 25) in [-1, 1]; with non-negative chroma, in [0, 1].
    Column 24 is `no_chord_threshold` rather than a similarity.

    Silent frames -- which `chroma` deliberately leaves as zeros rather than
    normalizing noise up into a confident chord -- are given to N outright,
    since a threshold of 0.0 would otherwise tie with every chord and the
    argmax would pick whichever came first.
    """
    X = np.asarray(chroma_frames, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_PITCH_CLASSES:
        raise ValueError(f"expected (n_frames, 12), got {X.shape}")
    if templates is None:
        templates = binary_templates()

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)

    scores = unit @ templates.T
    scores[:, NO_CHORD_INDEX] = no_chord_threshold

    silent = (norms.ravel() <= 0.0)
    if silent.any():
        scores[silent, :] = -np.inf
        scores[silent, NO_CHORD_INDEX] = no_chord_threshold
    return scores


def log_emission_probabilities(
    scores: np.ndarray,
    temperature: float = 1.0,
    floor: float = 1e-12,
) -> np.ndarray:
    """Turn similarities into log probabilities for the HMM (M4).

    Cosine similarity is not a probability, and pretending otherwise is the
    usual sleight of hand in template-based systems. A softmax with a
    temperature is an explicit, invertible choice: temperature -> 0 approaches
    a hard argmax, large temperature approaches uniform. The temperature
    controls how much the observations get to outvote the transition model, so
    it is a real parameter of the system and goes in the sweep.

    Computed in log space throughout, per M4.
    """
    S = np.asarray(scores, dtype=np.float64) / max(temperature, floor)
    S = S - S.max(axis=1, keepdims=True)          # stabilize before exp
    return S - np.log(np.sum(np.exp(S), axis=1, keepdims=True))


def predict_states(
    chroma_frames: np.ndarray,
    no_chord_threshold: float = DEFAULT_NO_CHORD_THRESHOLD,
) -> np.ndarray:
    """Frame-wise argmax. The baseline the HMM has to beat.

    No smoothing, no memory, no transition model -- each frame decides alone.
    Expect it to flicker badly between neighbouring chords; that flicker is
    precisely the thing the HMM exists to fix, and measuring it here is what
    makes the improvement a number rather than an assertion.
    """
    return emission_scores(
        chroma_frames, no_chord_threshold=no_chord_threshold
    ).argmax(axis=1)
