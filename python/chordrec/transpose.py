"""Transpose a progression into a key the singer is comfortable in.

    python -m chordrec.transpose --keys D,G C Am F G
    python -m chordrec.transpose --keys D,G --file piano.wav

This is the second half of the product and it shares no machinery with the
first. Detection asks "what chords are these?"; this asks "what key are they
in, and what should it be instead?" -- a question about labels, not audio. It
runs on the detector's output, on a typed progression, or on a lead sheet, and
needs none of the chroma pipeline to work.

Why key and not melody register
-------------------------------
The obvious framing is "put the melody inside the singer's range", and it is
not available to us. `chroma` folds every octave into twelve pitch classes by
design -- that is what makes template matching tractable -- so the detector
cannot report that a melody peaked at G5. It cannot report octaves at all.

So the rule here is the one a working musician actually uses: you know the keys
your voice sits well in, the song is in some other key, move it. That trades
precision for only needing information we have. The melody-span rule is
strictly better when the melody is known, and is a later addition rather than
a replacement -- see `best_shift`, which is where it would slot in.

Comfortable keys are tonics, not modes
--------------------------------------
`--keys D,G` means "D and G are comfortable tonics", and a minor song moved to
D becomes D minor, not D major. Transposition never changes mode, and asking a
singer to name twelve major and twelve minor keys separately is a worse
interface than it is an improvement in accuracy. The simplification is real
though: D major and D minor do not sit identically in a voice, and if that
starts mattering the flag should grow rather than the code get cleverer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vocab import (
    N_PITCH_CLASSES,
    NO_CHORD_INDEX,
    NO_CHORD_LABEL,
    PITCH_CLASS_NAMES,
    _NAME_TO_PC,
    parse_label,
    reduce_label,
    state_label,
)

# How much each diatonic degree counts as evidence for a key.
#
# Derived from function, not fitted: the tonic is the strongest evidence, the
# dominant and subdominant next because they define the key's frame, and the
# remaining diatonic chords count but weakly since they are shared with
# neighbouring keys. A vi chord is equally at home in its relative major, which
# is exactly why it cannot be allowed to decide between them.
#
# These are a defensible starting point and nothing more. The honest way to set
# them is to fit them on an annotated corpus, which is milestone 4 work; until
# then they are a baseline for that to beat.
MAJOR_PROFILE = {
    (0, "maj"): 3.0,     # I
    (7, "maj"): 2.0,     # V
    (5, "maj"): 2.0,     # IV
    (9, "min"): 1.0,     # vi
    (2, "min"): 1.0,     # ii
    (4, "min"): 1.0,     # iii
}

MINOR_PROFILE = {
    (0, "min"): 3.0,     # i
    (7, "min"): 1.5,     # v
    (7, "maj"): 1.5,     # V   (harmonic minor; common enough to weigh equally)
    (5, "min"): 2.0,     # iv
    (8, "maj"): 1.0,     # VI
    (3, "maj"): 1.0,     # III
    (10, "maj"): 1.0,    # VII
}


@dataclass(frozen=True)
class Key:
    """A key as a tonic pitch class plus a mode."""

    tonic: int
    mode: str            # "maj" or "min"

    def __str__(self) -> str:
        return f"{PITCH_CLASS_NAMES[self.tonic]} {'major' if self.mode == 'maj' else 'minor'}"


def parse_key(text: str) -> int:
    """A comfortable-key flag value to a tonic pitch class.

    Accepts "D", "d", "Eb", "F#". Mode is deliberately ignored if present, per
    the module docstring -- "Dm" and "D" both mean the tonic D.
    """
    text = text.strip().rstrip("m").rstrip(":")
    name = text[:2] if len(text) > 1 and text[1] in "#b" else text[:1]
    name = name[0].upper() + name[1:]
    if name not in _NAME_TO_PC:
        raise ValueError(f"not a note name: {text!r}")
    return _NAME_TO_PC[name]


def parse_chord_symbol(text: str) -> int:
    """A chord as a human writes it, to a state index. "Am", "F#m7", "Bb".

    `vocab.parse_label` and `reduce_label` both speak Harte ("A:min"), which is
    right for corpora and wrong for a person at a terminal. This translates the
    lead-sheet spelling everyone actually uses into Harte and hands off, so the
    reduction rules stay in one place instead of being reimplemented here with
    slightly different edge cases.

    Anything richer than major/minor reduces, per `reduce_label`: Am7 is a
    minor chord, Cmaj7 a major one, C/G a C. That is the 25-state vocabulary
    doing what it says, not an accident.
    """
    text = text.strip()
    if not text or text in ("N", "NC", "-", "X"):
        return NO_CHORD_INDEX
    if ":" in text:                                   # already Harte
        return reduce_label(text)

    root = text[:2] if len(text) > 1 and text[1] in "#b" else text[:1]
    root = root[0].upper() + root[1:]
    if root not in _NAME_TO_PC:
        raise ValueError(f"unknown root in {text!r}")

    suffix = text[len(root):].split("/")[0]           # drop any bass note
    lowered = suffix.lower()
    if lowered.startswith(("m", "-")) and not lowered.startswith("maj"):
        quality = "min"
    elif lowered.startswith(("dim", "o", "°")):
        quality = "dim"
    else:
        quality = "maj"
    return reduce_label(f"{root}:{quality}")


def detect_key(
    labels: list[str],
    durations: list[float] | np.ndarray | None = None,
) -> tuple[Key, float]:
    """Best-fitting key for a progression, and how clearly it won.

    Scores all 24 keys against the weighted profiles above and returns the
    winner plus a margin in [0, 1] -- the winning score's lead over the
    runner-up, as a fraction of the winner. A low margin is the signal that
    matters: relative major and minor share six of seven chords, so C major and
    A minor routinely finish within a few percent of each other, and a caller
    that transposes on a 2% margin is guessing rather than knowing.

    `durations` weights each chord by how long it sounds. Without it every
    chord counts once, which over-weights passing chords -- a four-bar tonic
    and a half-bar chromatic approach are not equal evidence. Pass the segment
    lengths from `recognize` whenever you have them.
    """
    if durations is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(durations, dtype=np.float64)
        if len(weights) != len(labels):
            raise ValueError(
                f"{len(labels)} labels but {len(weights)} durations"
            )

    scores: dict[Key, float] = {}
    for tonic in range(N_PITCH_CLASSES):
        for mode, profile in (("maj", MAJOR_PROFILE), ("min", MINOR_PROFILE)):
            total = 0.0
            for label, weight in zip(labels, weights):
                state = parse_label(label)
                if state == NO_CHORD_INDEX:
                    continue
                degree = (state % N_PITCH_CLASSES - tonic) % N_PITCH_CLASSES
                quality = "maj" if state < N_PITCH_CLASSES else "min"
                total += weight * profile.get((degree, quality), 0.0)
            scores[Key(tonic, mode)] = total

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (best, best_score), (_, second) = ranked[0], ranked[1]
    margin = 0.0 if best_score <= 0 else (best_score - second) / best_score
    return best, margin


def transpose_label(label: str, semitones: int) -> str:
    """Shift one Harte label. No-chord passes through untouched."""
    state = parse_label(label)
    if state == NO_CHORD_INDEX:
        return NO_CHORD_LABEL
    offset = state // N_PITCH_CLASSES * N_PITCH_CLASSES     # 0 for maj, 12 for min
    return state_label(offset + (state + semitones) % N_PITCH_CLASSES)


def transpose_progression(labels: list[str], semitones: int) -> list[str]:
    return [transpose_label(label, semitones) for label in labels]


def best_shift(from_tonic: int, comfortable: list[int]) -> int:
    """Semitones from the song's tonic to the nearest comfortable one.

    Expressed in [-6, +5]: every transposition has an equivalent an octave
    away, and the small one is always the one you want on an instrument. Ties
    (a tritone away, reachable equally in both directions) resolve downward,
    because a shift that is too low is uncomfortable and one that is too high
    is unsingable.

    This is the seam where the melody-span rule would go. That version would
    take the melody's range and the singer's, and choose among the candidate
    shifts by headroom rather than by distance -- same signature, more
    information, and the reason this returns a shift rather than a key.
    """
    if not comfortable:
        raise ValueError("no comfortable keys given")

    def normalize(semitones: int) -> int:
        return (semitones + 6) % N_PITCH_CLASSES - 6

    candidates = [normalize(tonic - from_tonic) for tonic in comfortable]
    return min(candidates, key=lambda s: (abs(s), s))


def suggest(
    labels: list[str],
    comfortable: list[int],
    durations: list[float] | np.ndarray | None = None,
) -> dict:
    """Detect the key, pick a shift, and transpose. The whole feature."""
    key, margin = detect_key(labels, durations)
    shift = best_shift(key.tonic, comfortable)
    target = Key((key.tonic + shift) % N_PITCH_CLASSES, key.mode)
    return {
        "from_key": key,
        "to_key": target,
        "margin": margin,
        "semitones": shift,
        "labels": transpose_progression(labels, shift),
    }


def _print_result(original: list[str], result: dict) -> None:
    arrow = "no change" if result["semitones"] == 0 else (
        f"{result['semitones']:+d} semitones"
    )
    print(f"  key        {result['from_key']}  ->  {result['to_key']}   ({arrow})")
    print(f"  confidence {result['margin']:.0%}")
    if result["margin"] < 0.10:
        print("  ^ thin. relative major and minor share six of seven chords;")
        print("    check this is the key you think it is before trusting the shift.")
    print()
    width = max((len(x) for x in original), default=1)
    print("  " + "  ".join(f"{x:<{width}}" for x in original))
    print("  " + "  ".join(f"{x:<{width}}" for x in result["labels"]))


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("chords", nargs="*", help="progression, e.g. C Am F G")
    p.add_argument("--keys", required=True,
                   help="comfortable tonics, comma separated (e.g. D,G)")
    p.add_argument("--file", help="recognize chords from an audio file first")
    args = p.parse_args(argv)

    comfortable = [parse_key(k) for k in args.keys.split(",") if k.strip()]

    if args.file:
        from .recognize import SR, load_audio, recognize
        intervals, labels = recognize(load_audio(args.file, SR), SR)
        durations = [float(e - s) for s, e in intervals]
        print(f"\n{args.file}  ->  {len(labels)} segments")
    elif args.chords:
        labels = [state_label(parse_chord_symbol(c)) for c in args.chords]
        durations = None
        print()
    else:
        p.error("give a progression or --file")

    print()
    _print_result(labels, suggest(labels, comfortable, durations))
    print()


if __name__ == "__main__":
    main()
