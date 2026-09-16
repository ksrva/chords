"""The chord vocabulary, and the state ordering everything else depends on.

Twenty-five states: twelve major triads, twelve minor triads, and "no chord".
Extended chords are explicitly out of scope for version one (PRD non-goals).

The ordering defined here is load-bearing and must not be rearranged casually.
It indexes the rows of the transition matrix, the rows of the emission
templates, and the state numbering the Rust core uses when it reads the
exported parameter files. A silent reordering would not crash anything -- it
would just make the detector confidently wrong, which is worse.

    index  0..11   major, root pitch class 0..11   (C:maj .. B:maj)
    index 12..23   minor, root pitch class 0..11   (C:min .. B:min)
    index    24    N, no chord

Labels use Harte notation ("C:maj", "A:min", "N"), which is what the
annotation corpora publish and what mir_eval parses.
"""

from __future__ import annotations

N_PITCH_CLASSES = 12
N_STATES = 25
NO_CHORD_INDEX = 24
NO_CHORD_LABEL = "N"

# Sharps throughout. The corpora mix sharps and flats; we normalize on input
# and emit one spelling so that string comparison is meaningful. Spelling for
# *display* is a UI concern and belongs nowhere near the detector.
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_NAME_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

MAJOR_INTERVALS = (0, 4, 7)
MINOR_INTERVALS = (0, 3, 7)


def state_label(index: int) -> str:
    """Harte label for a state index."""
    if index == NO_CHORD_INDEX:
        return NO_CHORD_LABEL
    if not 0 <= index < N_STATES:
        raise ValueError(f"state index out of range: {index}")
    quality = "maj" if index < N_PITCH_CLASSES else "min"
    return f"{PITCH_CLASS_NAMES[index % N_PITCH_CLASSES]}:{quality}"


def all_labels() -> list[str]:
    return [state_label(i) for i in range(N_STATES)]


def parse_label(label: str) -> int:
    """Harte label to state index.

    Deliberately narrow: this understands the vocabulary the detector has, and
    raises on anything else rather than guessing. Reducing a corpus's richer
    labels ("G:7", "C:maj/3") down to this vocabulary is a separate, explicit
    step -- see `reduce_label` -- because silently mapping an unfamiliar chord
    to something plausible is how evaluation bugs get hidden.
    """
    label = label.strip()
    if label in ("N", "X", ""):
        return NO_CHORD_INDEX

    root, _, quality = label.partition(":")
    if root not in _NAME_TO_PC:
        raise ValueError(f"unknown root in {label!r}")
    pc = _NAME_TO_PC[root]

    quality = quality or "maj"          # bare "C" means C major in Harte
    if quality == "maj":
        return pc
    if quality == "min":
        return N_PITCH_CLASSES + pc
    raise ValueError(f"quality {quality!r} is outside the major/minor vocabulary")


def reduce_label(label: str) -> int:
    """Map any corpus chord label into the 25-state vocabulary.

    The rule is the standard major/minor reduction: a chord counts as minor if
    it contains a minor third above its root, major otherwise, and anything
    without a root (N, X) becomes no-chord. So G:7 reduces to G:maj and
    C:min7 to C:min -- the reduction that WCSR-majmin assumes.

    Kept separate from `parse_label` so that reduction is always a visible,
    deliberate act rather than a side effect of reading a file.
    """
    label = label.strip()
    if label in ("N", "X", ""):
        return NO_CHORD_INDEX

    # Strip any bass inversion ("C:maj/3"); inversions are out of scope and
    # chroma is octave-invariant anyway, so the bass note changes nothing here.
    label = label.split("/")[0]
    root, _, quality = label.partition(":")
    if root not in _NAME_TO_PC:
        raise ValueError(f"unknown root in {label!r}")
    pc = _NAME_TO_PC[root]

    q = quality.lower()
    is_minor = q.startswith("min") or q.startswith("dim") or q.startswith("hdim")
    return (N_PITCH_CLASSES + pc) if is_minor else pc


def chord_pitch_classes(index: int) -> set[int]:
    """The pitch classes sounding in a state. Empty for no-chord."""
    if index == NO_CHORD_INDEX:
        return set()
    root = index % N_PITCH_CLASSES
    intervals = MAJOR_INTERVALS if index < N_PITCH_CLASSES else MINOR_INTERVALS
    return {(root + i) % N_PITCH_CLASSES for i in intervals}
