"""Score the detector over an annotated corpus. The number the README needs.

    python -m chordrec.evaluate_corpus ~/chord/guitarset
    python -m chordrec.evaluate_corpus ~/chord/guitarset --source performed

Everything measured up to now has been synthetic audio, where the detector is
graded on signals built by the same assumptions it was written under, or a
couple of songs checked by ear. This scores it against recordings of people
playing, with ground truth someone else wrote down.

Weighting across excerpts
-------------------------
Weighted chord symbol recall is a fraction of *duration*, which makes averaging
per-excerpt scores wrong: a thirty-second excerpt and a three-minute one would
count equally, and the mean of the two fractions is not the fraction over the
whole corpus. So correct duration and total duration are accumulated separately
and divided once at the end. On GuitarSet, where every excerpt is about the
same length, the two agree closely -- and they will not on a corpus of whole
songs, which is when quietly averaging would start lying.

Both are reported anyway. If they diverge it means excerpt length correlates
with accuracy, which is worth seeing rather than smoothing over.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .datasets import guitarset_pairs, read_jams_chords
from .evaluate import score, transitions_per_minute
from .recognize import SR, confidence, load_audio, recognize


def evaluate_pair(
    audio_path: Path,
    annotation_path: Path,
    source: str = "instructed",
    **recognize_kwargs,
) -> dict:
    """Run the detector on one excerpt and score it against its annotation."""
    reference_intervals, reference_labels = read_jams_chords(annotation_path, source)
    if not reference_labels:
        raise ValueError(f"empty annotation: {annotation_path}")

    audio = load_audio(str(audio_path), SR)
    estimated_intervals, estimated_labels = recognize(audio, SR, **recognize_kwargs)

    results = score(
        reference_intervals, reference_labels,
        estimated_intervals, estimated_labels,
    )
    results.update(
        name=annotation_path.stem,
        duration=float(reference_intervals[-1, 1] - reference_intervals[0, 0]),
        margin=confidence(audio, SR, **recognize_kwargs),
        est_per_minute=transitions_per_minute(estimated_intervals),
        ref_per_minute=transitions_per_minute(reference_intervals),
    )
    return results


def aggregate(rows: list[dict]) -> dict:
    """Corpus totals: duration-weighted, with the unweighted mean alongside."""
    if not rows:
        return {}

    total = sum(row["duration"] for row in rows) or 1.0
    weighted = {
        metric: sum(row[metric] * row["duration"] for row in rows) / total
        for metric in ("wcsr", "root", "thirds", "seg")
    }
    return {
        **weighted,
        "unweighted_wcsr": float(np.mean([row["wcsr"] for row in rows])),
        "margin": float(np.mean([row["margin"] for row in rows])),
        "est_per_minute": float(np.mean([row["est_per_minute"] for row in rows])),
        "ref_per_minute": float(np.mean([row["ref_per_minute"] for row in rows])),
        "excerpts": len(rows),
        "minutes": total / 60.0,
    }


def _report(totals: dict, source: str, worst: list[dict], best: list[dict]) -> None:
    print()
    print(f"  GuitarSet, {source} annotations")
    print(f"  {totals['excerpts']} excerpts, {totals['minutes']:.0f} minutes of audio")
    print("  " + "-" * 46)
    print(f"  {'WCSR (majmin)':<24} {totals['wcsr']:.3f}")
    print(f"  {'root':<24} {totals['root']:.3f}")
    print(f"  {'thirds':<24} {totals['thirds']:.3f}")
    print(f"  {'segmentation':<24} {totals['seg']:.3f}")
    print("  " + "-" * 46)
    print(f"  {'unweighted mean WCSR':<24} {totals['unweighted_wcsr']:.3f}")
    print(f"  {'mean margin':<24} {totals['margin']:.3f}")
    print(f"  {'changes/min est vs ref':<24} "
          f"{totals['est_per_minute']:.0f} vs {totals['ref_per_minute']:.0f}")

    print()
    print("  best")
    for row in best:
        print(f"    {row['wcsr']:.3f}  {row['name']}")
    print("  worst")
    for row in worst:
        print(f"    {row['wcsr']:.3f}  {row['name']}")
    print()


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", help="a GuitarSet download (annotation/ and audio_mono-mic/)")
    p.add_argument("--source", default="instructed",
                   choices=("instructed", "performed"),
                   help="which ground truth: the lead sheet, or what was played")
    p.add_argument("--smoothing", type=int, default=5)
    p.add_argument("--limit", type=int, help="score only the first N excerpts")
    p.add_argument("--quiet", action="store_true", help="no per-excerpt progress")
    args = p.parse_args(argv)

    pairs = guitarset_pairs(args.root)
    if not pairs:
        raise SystemExit(
            f"no (audio, annotation) pairs under {args.root} -- expected "
            "annotation/*.jams and audio_mono-mic/*_mic.wav"
        )
    if args.limit:
        pairs = pairs[:args.limit]

    rows: list[dict] = []
    started = time.time()
    for index, (audio, annotation) in enumerate(pairs, start=1):
        try:
            rows.append(evaluate_pair(audio, annotation, args.source,
                                      smoothing=args.smoothing))
        except Exception as exc:                       # keep going, report at the end
            print(f"  skipped {annotation.stem}: {type(exc).__name__}: {exc}")
            continue
        if not args.quiet and index % 20 == 0:
            print(f"  {index}/{len(pairs)}  running WCSR {aggregate(rows)['wcsr']:.3f}")

    if not rows:
        raise SystemExit("nothing scored")

    by_score = sorted(rows, key=lambda row: row["wcsr"])
    _report(aggregate(rows), args.source, by_score[:5], by_score[-5:][::-1])
    print(f"  scored {len(rows)}/{len(pairs)} in {time.time() - started:.0f}s")
    print()


if __name__ == "__main__":
    main()
