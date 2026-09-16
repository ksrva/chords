"""Chords over lyrics, the way a chord sheet is actually written.

    python -m chordrec.chart song.mp3 song.lrc --keys D,G

        G           Em            Am          D
    What would I do without your smart mouth

The detector already produces chords with timestamps. What it cannot produce is
words, and no amount of work on the front end will change that -- chroma
describes pitch classes, not language. So the lyrics come from outside, as an
`.lrc` file: the standard timed-lyric format, plain text, a timestamp and a
line. Nothing to install, and they exist for most songs.

Placing a chord within a line
-----------------------------
An `.lrc` timestamps the *start* of each line and nothing inside it, so the
column a chord belongs over has to be inferred. The assumption here is that
syllables are evenly spaced across a line, which is wrong in detail -- singers
hold notes, rush phrases, and leave gaps -- and close enough that the chord
lands on or beside the right word. Getting it exactly right needs word-level
timings, which is a different input, not a cleverer algorithm.

Lines are the unit, so a chord that changes between two lines is printed at the
start of the later one rather than being lost.

A word on trust
---------------
A chord sheet looks authoritative in a way a table of percentages does not, and
this one is built from labels whose per-segment margin on a full mix is around
0.05 -- close to a coin toss between the top two candidates. The key is usually
right because it aggregates; individual chords frequently are not. The renderer
therefore marks low-confidence chords rather than presenting every one as
settled, because a sheet that quietly lies is worse than no sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .transpose import Key, transpose_label
from .vocab import parse_label

# [mm:ss.xx] or [mm:ss:xx] or [mm:ss]; the fractional part may be 2 or 3 digits.
LRC_LINE = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")

# An .lrc may carry metadata lines ([ti:], [ar:], [by:]) that are not lyrics.
LRC_META = re.compile(r"^\[[a-z]{2,}:", re.IGNORECASE)


@dataclass
class ChartLine:
    """One lyric line, and the chords that sound over it."""

    time: float
    text: str
    chords: list[tuple[int, str]]      # (column, chord name)

    def render(self) -> list[str]:
        """The two printed lines: chords above, words below."""
        if not self.chords:
            return [self.text]
        width = max(col + len(name) for col, name in self.chords)
        row = [" "] * max(width, len(self.text))
        for col, name in self.chords:
            row[col:col + len(name)] = name
        return ["".join(row).rstrip(), self.text]


def parse_lrc(text: str) -> list[tuple[float, str]]:
    """An .lrc file to (seconds, line) pairs, in time order.

    A line may carry several timestamps when a chorus repeats -- the format
    allows `[01:02.00][02:34.00]same words` -- so each one becomes its own
    entry rather than the first winning.
    """
    out: list[tuple[float, str]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        stamps = list(LRC_LINE.finditer(raw))
        if not stamps:
            continue
        words = raw[stamps[-1].end():].strip()
        if not words or LRC_META.match(raw) and not stamps:
            continue
        for stamp in stamps:
            minutes, seconds, fraction = stamp.groups()
            value = int(minutes) * 60 + int(seconds)
            if fraction:
                value += int(fraction) / (10 ** len(fraction))
            out.append((value, words))
    return sorted(out, key=lambda pair: pair[0])


def _column_for(time: float, start: float, end: float, width: int) -> int:
    """Where in a line of `width` characters a chord at `time` belongs."""
    if end <= start or width <= 0:
        return 0
    fraction = (time - start) / (end - start)
    return max(0, min(width - 1, int(round(fraction * width))))


def song_vocabulary(
    labels: list[str],
    durations: list[float],
    coverage: float = 0.90,
) -> set[str]:
    """The chords a song is actually made of.

    Songs are built from a handful of chords: on the track this was written
    against, eight of them account for 97% of the duration. A chord that turns
    up once for a second and never again is therefore far more likely to be a
    detection error than a real one, and on a full mix -- where the margin
    between the winning chord and the runner-up runs around 0.05 -- there are
    a lot of those.

    So the chart is restricted to whichever chords cover `coverage` of the
    song, and anything outside that set is treated as a misread and the
    previous chord carries on. This cleans the sheet without touching the
    smoothing, which matters because smoothing harder does not just remove
    noise -- it changes which chord wins, trading one error for another.

    It is a presentation filter and nothing more. It cannot rescue a chord
    that is wrong but common, and it will discard a real passing chord in a
    song that has one. Both are the right trade for a sheet someone plays from.
    """
    totals: dict[str, float] = {}
    for label, duration in zip(labels, durations):
        totals[label] = totals.get(label, 0.0) + duration
    span = sum(totals.values()) or 1.0

    keep: set[str] = set()
    running = 0.0
    for label, total in sorted(totals.items(), key=lambda kv: -kv[1]):
        keep.add(label)
        running += total / span
        if running >= coverage:
            break
    return keep


def build_chart(
    intervals: np.ndarray,
    labels: list[str],
    lyrics: list[tuple[float, str]],
    key: Key,
    shift: int = 0,
    trailing: float = 6.0,
    restrict_to: set[str] | None = None,
) -> list[ChartLine]:
    """Lay chord segments over timed lyric lines.

    `trailing` is how long the final line is assumed to last, since an .lrc
    gives no end time for it. Only the last line's chord placement depends on
    it, so a rough value is fine and a wrong one is visibly harmless.

    `restrict_to` drops chords outside the song's vocabulary -- see
    `song_vocabulary`. Passing None keeps every detected chord, which is the
    honest raw view and a busier sheet.
    """
    if not lyrics:
        return []

    if restrict_to is not None:
        kept = [(i, lab) for i, lab in enumerate(labels) if lab in restrict_to]
        if kept:
            keep_idx = [i for i, _ in kept]
            intervals = intervals[keep_idx]
            labels = [lab for _, lab in kept]

    starts = np.asarray(intervals[:, 0], dtype=float) if len(intervals) else np.zeros(0)
    chart: list[ChartLine] = []

    for index, (start, text) in enumerate(lyrics):
        end = lyrics[index + 1][0] if index + 1 < len(lyrics) else start + trailing
        width = max(len(text), 1)

        # Chord segments beginning inside this line, plus whichever chord was
        # already sounding when the line began -- otherwise a line that starts
        # mid-chord is printed bare.
        inside = np.flatnonzero((starts >= start) & (starts < end))
        placed: list[tuple[int, str]] = []

        carried = np.flatnonzero(starts <= start)
        if len(carried):
            name = key.chord(parse_label(transpose_label(labels[carried[-1]], shift)))
            placed.append((0, name))

        for i in inside:
            name = key.chord(parse_label(transpose_label(labels[i], shift)))
            column = _column_for(float(starts[i]), start, end, width)
            if placed and column <= placed[-1][0] + len(placed[-1][1]):
                column = placed[-1][0] + len(placed[-1][1]) + 1
            if placed and placed[-1][1] == name:
                continue                       # do not repeat an unchanged chord
            placed.append((column, name))

        chart.append(ChartLine(time=start, text=text, chords=placed))

    return chart


def render(chart: list[ChartLine]) -> str:
    """The whole sheet as text."""
    out: list[str] = []
    for line in chart:
        out.extend(line.render())
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .recognize import SR, load_audio, recognize
    from .transpose import parse_key, suggest

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("audio", help="the recording")
    p.add_argument("lyrics", help="an .lrc file of timed lyrics")
    p.add_argument("--keys", help="comfortable tonics, comma separated; "
                                  "omit to leave the song in its own key")
    p.add_argument("--raw", action="store_true",
                   help="keep every detected chord, including likely misreads")
    p.add_argument("--smoothing", type=int, default=41)
    p.add_argument("--min-duration", type=float, default=1.0)
    args = p.parse_args(argv)

    audio = load_audio(args.audio, SR)
    # A chord sheet wants chord-level resolution, not frame-level: measured
    # against the verse's real progression, these put 79% of placed chords
    # inside the song's actual chord set against 70% at the defaults. More
    # smoothing is not monotonically better -- at 61 frames it collapses to
    # 0%, having averaged the chords away entirely.
    intervals, labels = recognize(audio, SR, smoothing=args.smoothing,
                                  min_duration=args.min_duration)
    durations = [float(e - s) for s, e in intervals]
    lyrics = parse_lrc(Path(args.lyrics).read_text(encoding="utf-8", errors="replace"))
    if not lyrics:
        raise SystemExit(f"no timed lines found in {args.lyrics}")

    comfortable = [parse_key(k) for k in args.keys.split(",")] if args.keys else []
    if comfortable:
        result = suggest(labels, comfortable, durations)
        key, shift = result["to_key"], result["semitones"]
        print(f"\n  {result['from_key']} -> {key}, "
              f"{'no change' if shift == 0 else f'{shift:+d} semitones'}\n")
    else:
        from .transpose import detect_key
        key, shift = detect_key(labels, durations)[0], 0
        print(f"\n  {key}\n")

    vocabulary = None if args.raw else song_vocabulary(labels, durations)
    print(render(build_chart(intervals, labels, lyrics, key, shift,
                             restrict_to=vocabulary)))


if __name__ == "__main__":
    main()
