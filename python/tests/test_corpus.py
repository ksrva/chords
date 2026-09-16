"""Tests for corpus reading and scoring.

These run without GuitarSet: a JAMS file is JSON, and audio matching a known
annotation can be synthesised. That matters because the harness has to be
trustworthy before its output means anything -- a corpus score is only evidence
if the thing producing it is not quietly mis-reading the ground truth.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from chordrec.datasets import guitarset_pairs, read_jams_chords
from chordrec.evaluate_corpus import aggregate, evaluate_pair
from chordrec.signals import progression

SR = 22050
CHORDS = [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")]
SECONDS = 2.0


def jams_document(instructed: list[tuple[float, float, str]],
                  performed: list[tuple[float, float, str]]) -> dict:
    """A minimal JAMS with the two chord annotations GuitarSet carries."""
    def annotation(observations, provenance):
        return {
            "namespace": "chord",
            "annotation_metadata": {"data_source": provenance},
            "data": [
                {"time": t, "duration": d, "value": v, "confidence": None}
                for t, d, v in observations
            ],
        }
    return {
        "file_metadata": {"duration": 8.0},
        "annotations": [
            # Deliberately performed-first: the reader must not rely on order.
            annotation(performed, "Semi-automatic chord transcription with manual verification"),
            annotation(instructed, ""),
        ],
    }


@pytest.fixture
def jams_path(tmp_path):
    instructed = [(i * SECONDS, SECONDS, f"{root}:{quality}")
                  for i, (root, quality) in enumerate(CHORDS)]
    performed = [(i * SECONDS, SECONDS, v) for i, v in enumerate(
        ["C:maj7", "A:min7/b3", "F:maj6(*5)", "G:sus2(7)/1"])]
    path = tmp_path / "excerpt.jams"
    path.write_text(json.dumps(jams_document(instructed, performed)))
    return path


def test_reads_the_instructed_annotation(jams_path):
    intervals, labels = read_jams_chords(jams_path, "instructed")
    assert labels == ["C:maj", "A:min", "F:maj", "G:maj"]
    assert intervals[0].tolist() == [0.0, 2.0]
    assert intervals[-1].tolist() == [6.0, 8.0]


def test_reads_the_performed_annotation(jams_path):
    """Richer labels, reduced into the 25-state vocabulary on the way in."""
    _, labels = read_jams_chords(jams_path, "performed")
    assert labels == ["C:maj", "A:min", "F:maj", "G:maj"]


def test_annotations_are_chosen_by_provenance_not_position(jams_path):
    """The fixture puts performed first. An index-based reader would score the
    wrong ground truth here and report a number that looks fine."""
    document = json.loads(jams_path.read_text())
    assert "transcription" in document["annotations"][0]["annotation_metadata"]["data_source"]
    instructed_first = read_jams_chords(jams_path, "instructed")[1]
    assert instructed_first == ["C:maj", "A:min", "F:maj", "G:maj"]


def test_unknown_source_is_rejected(jams_path):
    with pytest.raises(ValueError, match="instructed"):
        read_jams_chords(jams_path, "whatever")


def test_missing_chord_namespace_is_an_error(tmp_path):
    path = tmp_path / "empty.jams"
    path.write_text(json.dumps({"annotations": [{"namespace": "beat", "data": []}]}))
    with pytest.raises(ValueError, match="no chord annotation"):
        read_jams_chords(path)


def test_zero_duration_observations_are_dropped(tmp_path):
    path = tmp_path / "zero.jams"
    path.write_text(json.dumps(jams_document(
        [(0.0, 2.0, "C:maj"), (2.0, 0.0, "G:maj")], [(0.0, 2.0, "C:maj")])))
    _, labels = read_jams_chords(path, "instructed")
    assert labels == ["C:maj"]


def test_pairs_skip_excerpts_whose_audio_is_missing(tmp_path):
    (tmp_path / "annotation").mkdir()
    (tmp_path / "audio_mono-mic").mkdir()
    for stem in ("a", "b"):
        (tmp_path / "annotation" / f"{stem}.jams").write_text("{}")
    sf.write(tmp_path / "audio_mono-mic" / "a_mic.wav", np.zeros(100), SR)

    pairs = guitarset_pairs(tmp_path)
    assert [annotation.stem for _, annotation in pairs] == ["a"]


def test_scores_audio_that_matches_its_annotation(tmp_path, jams_path):
    """End to end on a signal whose ground truth is known exactly. If the
    harness scores this badly, the harness is wrong, not the detector."""
    audio, _ = progression(CHORDS, seconds_each=SECONDS, sr=SR)
    audio_path = tmp_path / "excerpt_mic.wav"
    sf.write(audio_path, audio, SR)

    result = evaluate_pair(audio_path, jams_path, "instructed")
    assert result["wcsr"] > 0.9
    assert result["duration"] == pytest.approx(8.0)
    assert result["name"] == "excerpt"


def test_scores_mismatched_audio_badly(tmp_path, jams_path):
    """The complement: a harness that cannot fail cannot be evidence."""
    wrong, _ = progression([("F#", "maj")] * 4, seconds_each=SECONDS, sr=SR)
    audio_path = tmp_path / "wrong_mic.wav"
    sf.write(audio_path, wrong, SR)
    assert evaluate_pair(audio_path, jams_path, "instructed")["wcsr"] < 0.2


def test_aggregate_weights_by_duration_not_by_excerpt():
    """A long excerpt must count for more. The mean of the fractions is not
    the fraction over the whole corpus, and on a corpus of whole songs the
    difference is the entire result."""
    rows = [
        {"wcsr": 1.0, "root": 1.0, "thirds": 1.0, "seg": 1.0, "duration": 300.0,
         "margin": 0.1, "est_per_minute": 20, "ref_per_minute": 20},
        {"wcsr": 0.0, "root": 0.0, "thirds": 0.0, "seg": 0.0, "duration": 100.0,
         "margin": 0.1, "est_per_minute": 20, "ref_per_minute": 20},
    ]
    totals = aggregate(rows)
    assert totals["wcsr"] == pytest.approx(0.75)          # 300 / 400
    assert totals["unweighted_wcsr"] == pytest.approx(0.5)
    assert totals["excerpts"] == 2
    assert totals["minutes"] == pytest.approx(400 / 60)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}
