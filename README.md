# chords *(working name)*

Real-time chord detection in the browser, built from first principles: a custom
spectral front end, an HMM with parameters learned from annotated music, and
online inference. Prototyped and evaluated in Python; shipped as a Rust core
compiled to WebAssembly behind a React UI.

Python never runs on a user's audio. It is the reference implementation, the
parameter-learning pipeline, and the evaluation harness.

## Layout

    python/     reference implementation, parameter learning, evaluation
    rust/       real-time core (native + wasm)          -- milestone 5
    web/        React + TypeScript front end            -- milestone 6
    fixtures/   shared ground truth for Rust-vs-Python  -- labels only, no audio
    params/     learned transition/emission data, exported from python/

## Status

Milestone 2 in progress: chroma front end written and validated.

## Running

    python3 -m venv .venv && .venv/bin/pip install -e 'python[dev,eval,reference]'
    .venv/bin/python -m pytest python/tests

## Decisions so far

**Causal framing by default.** `center=False`: frame *i* starts at sample
*i*·hop and reads no future samples. librosa's default centres frames on a
half-window of padding, which a live engine cannot do. The offline reference
and the online engine therefore see identical windows; `center=True` exists
only to line up with librosa in the comparison test.

**Frames are rows,** shape `(n_frames, 12)`. One row is one engine step.

**Hard semitone assignment.** Each FFT bin in the band is credited entirely to
its nearest semitone's pitch class. Simplest defensible mapping, so it is the
baseline that soft/Gaussian weighting has to beat rather than replace by
assumption.

**Silent frames stay zero.** Normalization leaves near-silent rows at zero
instead of scaling noise up into a confident wrong chord; the no-chord state
consumes that.

**No external dataset on the critical path.** Transition probabilities come
from freely published label corpora, which need no audio. The evaluation set is
locally recorded piano with labels *derived* (from MIDI, or a chart plus a
click) rather than hand-annotated, so ground truth is exact and nothing is
blocked on acquiring copyrighted recordings.

## Measured, not assumed

Agreement with librosa on a I–vi–IV–V progression: **cosine mean 0.974, min
0.971, argmax agreement 96.6%**. Not 1.0 by design — librosa weights octaves
with a Gaussian centred near C5, ours uses a flat band.

**The fifth usually beats the root.** A C major triad reads G > E > C. A note's
third partial is a twelfth above it, so the chord's fifth collects energy from
the root as well as itself. This is what octave-folded spectra look like, and
it is the reason the detector scores whole templates instead of picking a root.

**The bass limit, in numbers.** Bins per semitone at sr=22050; above 1.0 means
the distinction is not present in the transform at all:

| n_fft | C2 (65 Hz) | G2 (98 Hz) | C3 (131 Hz) | window |
|-------|-----------|-----------|------------|--------|
| 4096  | **1.38**  | 0.92      | 0.69       | 186 ms |
| 8192  | 0.69      | 0.46      | 0.35       | 372 ms |

So `n_fft=4096` cannot resolve semitones below roughly G2, and fixing it costs
a 372 ms window against a "few hundred ms" p95 latency budget. Time-frequency
uncertainty means dropping the sample rate buys nothing — 2.7 Hz bins cost
372 ms of audio either way. That trade is the substance of the window sweep.
