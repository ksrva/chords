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


# Enharmonic spelling, for display only.
#
# `vocab` spells everything with sharps, deliberately: one spelling makes string
# comparison meaningful and keeps the detector's output canonical. But "G#
# major" is not a key any musician writes, and a chart in G# is unreadable where
# the same chart in Ab is obvious. vocab's own docstring says spelling for
# display is a UI concern and belongs nowhere near the detector -- so it lives
# here, at the boundary, and never propagates back inward.
#
# Which accidental a key uses is fixed by its key signature: the flat keys take
# flats, the sharp keys take sharps, and C/A minor take neither.
FLAT_TONICS_MAJOR = {5, 10, 3, 8, 1, 6}        # F Bb Eb Ab Db Gb
FLAT_TONICS_MINOR = {2, 7, 0, 5, 10, 3}        # Dm Gm Cm Fm Bbm Ebm

FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass(frozen=True)
class Key:
    """A key as a tonic pitch class plus a mode."""

    tonic: int
    mode: str            # "maj" or "min"

    @property
    def uses_flats(self) -> bool:
        pool = FLAT_TONICS_MAJOR if self.mode == "maj" else FLAT_TONICS_MINOR
        return self.tonic in pool

    def spell(self, pitch_class: int) -> str:
        """Name a pitch class the way this key would write it."""
        names = FLAT_NAMES if self.uses_flats else PITCH_CLASS_NAMES
        return names[pitch_class % N_PITCH_CLASSES]

    def chord(self, state: int) -> str:
        """A state index as a musician writes it in this key: "Ab", "Fm"."""
        if state == NO_CHORD_INDEX:
            return "-"
        suffix = "" if state < N_PITCH_CLASSES else "m"
        return self.spell(state % N_PITCH_CLASSES) + suffix

    @property
    def relative(self) -> "Key":
        """The relative major or minor -- the key this one is confused with.

        Sharing six of seven chords, it is almost always the runner-up when
        detection is uncertain, so it is the one worth naming. "Could be F
        minor" tells someone what to go and check; explaining why the two are
        hard to tell apart does not.
        """
        if self.mode == "maj":
            return Key((self.tonic + 9) % N_PITCH_CLASSES, "min")
        return Key((self.tonic + 3) % N_PITCH_CLASSES, "maj")

    def __str__(self) -> str:
        quality = "major" if self.mode == "maj" else "minor"
        return f"{self.spell(self.tonic)} {quality}"


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


def chord_totals(
    labels: list[str],
    durations: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Chords by how long they sound, longest first.

    The useful summary of a song. A per-segment timeline is the honest raw
    output and is unreadable past about twenty segments -- a four-minute track
    produces several hundred, most of them fragments. What a player actually
    needs is which chords the song is made of, which is this.
    """
    weights = durations if durations is not None else [1.0] * len(labels)
    totals: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        totals[label] = totals.get(label, 0.0) + float(weight)
    return sorted(totals.items(), key=lambda kv: -kv[1])


def _print_result(
    original: list[str],
    result: dict,
    durations: list[float] | None = None,
    show_timeline: bool = False,
    top: int = 8,
) -> None:
    from_key, to_key = result["from_key"], result["to_key"]
    shift = result["semitones"]

    print(f"  key     {from_key}"
          + ("" if shift == 0 else f"   ->   {to_key}"))
    if shift == 0:
        print("  shift   none -- already in a key you sing")
    else:
        direction = "up" if shift > 0 else "down"
        plural = "" if abs(shift) == 1 else "s"
        print(f"  shift   {direction} {abs(shift)} semitone{plural}")

    totals = chord_totals(original, durations)
    span = sum(weight for _, weight in totals) or 1.0

    print()
    print("  chords to play")
    covered = 0.0
    for label, weight in totals[:top]:
        # Shift the *label*, not the state index: states are laid out
        # [0-11 major, 12-23 minor], so plain addition walks off the end of
        # major and into minor. Cm - 1 is Bm, but state 12 - 1 is state 11,
        # which is B major.
        state = parse_label(label)
        was = from_key.chord(state)
        now = to_key.chord(parse_label(transpose_label(label, shift)))
        share = weight / span
        covered += share
        bar = "#" * max(int(share * 40), 1)
        print(f"    {was:<4} -> {now:<4} {share:5.0%}  {bar}")
    if len(totals) > top:
        print(f"    ... and {len(totals) - top} more, "
              f"{1 - covered:.0%} of the time between them")
    print()
    print(f"  the {min(top, len(totals))} above cover {covered:.0%} of the song")

    if show_timeline:
        print()
        print("  timeline")
        width = 0
        line: list[str] = []
        for label in result["labels"]:
            cell = to_key.chord(parse_label(label))
            line.append(f"{cell:<5}")
            width += 1
            if width == 12:
                print("    " + "".join(line))
                line, width = [], 0
        if line:
            print("    " + "".join(line))

    if result["margin"] < 0.10:
        print()
        print(f"  not certain -- could be {from_key.relative}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("chords", nargs="*", help="progression, e.g. C Am F G")
    p.add_argument("--keys", required=True,
                   help="comfortable tonics, comma separated (e.g. D,G)")
    p.add_argument("--file", help="recognize chords from an audio file first")
    p.add_argument("--timeline", action="store_true",
                   help="print every chord in order, not just the summary")
    p.add_argument("--top", type=int, default=8,
                   help="how many chords to summarize (default 8)")
    args = p.parse_args(argv)

    comfortable = [parse_key(k) for k in args.keys.split(",") if k.strip()]

    if args.file:
        import os

        from .recognize import SR, load_audio, recognize

        audio = load_audio(args.file, SR)
        # A song's chords last seconds, not one 93 ms frame. Smoothing hard is
        # what makes a four-minute track readable instead of several hundred
        # fragments; the latency it costs is irrelevant when reading a file.
        intervals, labels = recognize(audio, SR, smoothing=21)
        durations = [float(e - s) for s, e in intervals]
        minutes, seconds = divmod(int(len(audio) / SR), 60)
        print()
        print(f"  {os.path.basename(args.file)}")
        print(f"  {minutes}:{seconds:02d}, {len(labels)} chord segments")
        print()
    elif args.chords:
        labels = [state_label(parse_chord_symbol(c)) for c in args.chords]
        durations = None
        print()
    else:
        p.error("give a progression or --file")

    show_timeline = args.timeline or (not args.file and len(labels) <= 32)
    _print_result(
        labels,
        suggest(labels, comfortable, durations),
        durations=durations,
        show_timeline=show_timeline,
        top=args.top,
    )
    print()


if __name__ == "__main__":
    main()
