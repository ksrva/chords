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

Milestones 2–3 in progress. Chroma front end validated; template emissions and
the evaluation harness in place, with a frame-wise baseline to beat. No HMM yet.

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

## Baseline

`python -m chordrec.baseline` runs chroma → templates → frame-wise argmax and
scores it. On **synthetic audio only**, so it measures plumbing, not accuracy on
music — the number that counts comes from recorded piano at milestone 4.

| condition | WCSR | margin |
|-----------|------|--------|
| clean | 0.983 | 0.028 |
| white noise, 0 dB SNR | 0.983 | 0.019 |
| percussion (loud) | 0.983 | 0.026 |
| harmonic interferer, 0.71× RMS | 0.611 | 0.013 |
| harmonic interferer, 1.06× RMS | 0.364 | 0.012 |

**White noise and percussion barely register.** Both spread energy across all
twelve pitch classes, and per-frame normalization divides it straight back out.
A robustness claim built on added noise would be worthless — which is worth
knowing before designing the ablation, since "add noise, report degradation" is
the obvious experiment and it measures nothing here.

**Harmonic interference is what breaks it**, because it lands on one pitch
class rather than all of them. Interferer levels are calibrated by sweep:
below ~0.35× the music's RMS nothing happens, above ~1.5× the interferer *is*
the signal and a score of 0.000 is not a robustness result.

**Margins are thin everywhere.** The gap between the best chord and the
runner-up is 0.028 even on clean audio. WCSR holds up while confidence does
not, and that gap is what the transition model has to exploit.

## Measured, not assumed

Agreement with librosa on a I–vi–IV–V progression: **cosine mean 0.974, min
0.971, argmax agreement 96.6%**. Not 1.0 by design — librosa weights octaves
with a Gaussian centred near C5, ours uses a flat band.

**The fifth usually beats the root.** A C major triad reads G > E > C. A note's
third partial is a twelfth above it, so the chord's fifth collects energy from
the root as well as itself. This is what octave-folded spectra look like, and
it is the reason the detector scores whole templates instead of picking a root.

**A uniform no-chord template cannot work.** Giving N a flat template and
letting it compete on cosine similarity is the obvious design and is measurably
wrong: uniform beat the correct chord 0.900 to 0.706 on a synthesised C major,
because cosine against twelve nonzeros is not comparable to cosine against
three. N is a threshold instead.

**Log compression costs discriminability.** The C:maj margin over uniform runs
+0.432 uncompressed and −0.194 at γ=100, because compression lifts the noise
floor and flattens the vector. Measured on synthetic audio with no dynamic
range to tame, so γ stays in the sweep rather than changing on this evidence.

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
