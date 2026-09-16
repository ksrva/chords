"""Reading annotated corpora, so accuracy can be a measurement instead of a claim.

Every number in this project so far comes from synthesised audio or from
listening to a result and deciding it sounded about right. Neither is
evaluation. This module reads a corpus that ships audio *and* ground truth, so
`evaluate_corpus` can produce a weighted chord symbol recall over real
recordings played by real people.

GuitarSet
---------
Solo acoustic guitar, 360 excerpts of about thirty seconds, audio and
annotations together under CC BY 4.0 -- which is the point. Isophonics is the
corpus everyone reports against, and it distributes annotations only: the
Beatles recordings are copyrighted, so using it means sourcing 180 tracks and
matching them to the exact remasters the annotations were timed against.
GuitarSet costs a 700 MB download and nothing else.

The trade is comparability. A GuitarSet number cannot be set beside a published
MIREX result, and solo guitar is an easier signal than a produced mix -- no
bass guitar in the octave the front end cannot resolve, no drums, no vocal.
Read the number as "how well does this do on one clean polyphonic instrument",
which is the question the piano use case actually asks.

Two ground truths
-----------------
Each file carries two chord annotations and they disagree, usefully:

  instructed  the lead sheet the player was handed -- "D#:maj"
  performed   what they actually played, transcribed and verified -- "D#:sus2(7)/1"

A player told "Eb" who voices an Ebsus2 is not playing Eb major, and a detector
that declines to call it Eb major is not wrong. Scoring against `instructed`
therefore measures agreement with the chart; scoring against `performed`
measures agreement with the audio. They are different questions and this reads
either, because reporting one without saying which would make the number
uninterpretable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vocab import reduce_label, state_label

# The performed annotation names its provenance; the instructed one is bare.
PERFORMED_MARKER = "transcription"


def _chord_annotations(document: dict) -> list[dict]:
    return [a for a in document.get("annotations", []) if a.get("namespace") == "chord"]


def read_jams_chords(
    path: str | Path,
    source: str = "instructed",
) -> tuple[np.ndarray, list[str]]:
    """Chord intervals and reduced labels from a JAMS file.

    `source` is "instructed" or "performed"; see the module docstring. Labels
    come back inside the 25-state vocabulary, via `reduce_label`, so anything
    richer than a triad has already been collapsed and the collapse is visible
    rather than implied.

    Annotations are picked by their metadata rather than by position. JAMS does
    not promise an order, and an index that happens to work on one file is the
    kind of thing that silently scores the wrong ground truth on another.
    """
    document = json.loads(Path(path).read_text())
    annotations = _chord_annotations(document)
    if not annotations:
        raise ValueError(f"no chord annotation in {path}")

    def is_performed(annotation: dict) -> bool:
        provenance = annotation.get("annotation_metadata", {}).get("data_source", "")
        return PERFORMED_MARKER in provenance.lower()

    if source == "performed":
        chosen = next((a for a in annotations if is_performed(a)), None)
    elif source == "instructed":
        chosen = next((a for a in annotations if not is_performed(a)), None)
    else:
        raise ValueError(f"source must be 'instructed' or 'performed', got {source!r}")

    if chosen is None:
        raise ValueError(f"no {source} chord annotation in {path}")

    starts, ends, labels = [], [], []
    for observation in chosen["data"]:
        start = float(observation["time"])
        duration = float(observation["duration"])
        if duration <= 0:
            continue
        starts.append(start)
        ends.append(start + duration)
        labels.append(state_label(reduce_label(observation["value"])))

    if not starts:
        return np.zeros((0, 2)), []
    return np.stack([starts, ends], axis=1), labels


def guitarset_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """(audio, annotation) pairs for a GuitarSet download.

    Expects the two zips unpacked side by side:

        root/annotation/00_BN1-129-Eb_comp.jams
        root/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav

    Excerpts whose audio is missing are skipped rather than raising, so a
    partial download still produces a number over whatever is present -- and
    `evaluate_corpus` reports how many it actually scored, so a half-finished
    download cannot be mistaken for a full run.
    """
    root = Path(root)
    annotations = sorted((root / "annotation").glob("*.jams"))
    audio_dir = root / "audio_mono-mic"

    pairs: list[tuple[Path, Path]] = []
    for annotation in annotations:
        audio = audio_dir / f"{annotation.stem}_mic.wav"
        if audio.exists():
            pairs.append((audio, annotation))
    return pairs
