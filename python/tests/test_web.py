"""Tests for the web layer, without a browser.

`analyze` and `parse_multipart` are both plain functions, so the only part that
needs a server is the socket. These cover the seam where a request becomes a
call -- which is exactly where the lyrics-in-the-audio-field bug lived.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from chordrec.signals import progression
from chordrec.web import LYRIC_SUFFIXES, analyze, parse_multipart
from chordrec.transpose import parse_key

BOUNDARY = "----chordrecTestBoundary"


def multipart(parts: list[tuple[str, str | None, bytes]]) -> tuple[bytes, str]:
    """Build a form body the way a browser would."""
    chunks: list[bytes] = []
    for name, filename, payload in parts:
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        chunks.append(
            f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n\r\n".encode()
            + payload
            + b"\r\n"
        )
    chunks.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={BOUNDARY}"


@pytest.fixture
def song(tmp_path):
    """A four-chord wav on disk."""
    y, _ = progression(
        [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")],
        seconds_each=2.0, sr=22050,
    )
    path = tmp_path / "song.wav"
    sf.write(path, y, 22050)
    return str(path)


def test_fields_map_to_their_names():
    """The bug: if these two ever swap, the decoder is handed a text file and
    reports 'Format not recognised', which explains nothing."""
    body, content_type = multipart([
        ("audio", "song.mp3", b"AUDIOBYTES"),
        ("lyrics", "song.lrc", b"[00:01.00]hi"),
    ])
    fields = parse_multipart(body, content_type)
    assert fields["audio"] == ("song.mp3", b"AUDIOBYTES")
    assert fields["lyrics"] == ("song.lrc", b"[00:01.00]hi")


def test_audio_only_form_has_no_lyrics_field():
    body, content_type = multipart([("audio", "song.wav", b"XX")])
    assert set(parse_multipart(body, content_type)) == {"audio"}


def test_binary_payload_survives_intact():
    """Audio is bytes, not text; a decode step anywhere would corrupt it."""
    payload = bytes(range(256))
    body, content_type = multipart([("audio", "song.wav", payload)])
    assert parse_multipart(body, content_type)["audio"][1] == payload


def test_a_field_without_a_filename_still_parses():
    body, content_type = multipart([("keys", None, b"D,G")])
    assert parse_multipart(body, content_type)["keys"] == (None, b"D,G")


def test_non_multipart_body_yields_nothing():
    assert parse_multipart(b"raw bytes", "application/octet-stream") == {}


def test_lyric_suffixes_cover_what_people_actually_upload():
    for suffix in (".lrc", ".txt", ".srt", ".vtt"):
        assert suffix in LYRIC_SUFFIXES
    for suffix in (".wav", ".mp3", ".flac"):
        assert suffix not in LYRIC_SUFFIXES


def test_analyze_returns_a_summary(song):
    result = analyze(song, [parse_key("D")], "song.wav")
    assert str(result["from_key"]) == "C major"
    assert result["to_key"] == "D major"
    assert result["semitones"] == 2
    assert result["duration"] == "0:08"
    assert result["sheet"] == []


def test_analyze_chord_rows_are_transposed_and_ordered(song):
    result = analyze(song, [parse_key("D")], "song.wav")
    assert [row["was"] for row in result["chords"]][:2] == ["C", "Am"]
    assert [row["now"] for row in result["chords"]][:2] == ["D", "Bm"]
    shares = [row["share"] for row in result["chords"]]
    assert shares == sorted(shares, reverse=True)


def test_analyze_builds_a_sheet_when_given_lyrics(song):
    lyrics = "[00:00.50]first line here\n[00:04.00]second line here"
    result = analyze(song, [parse_key("D")], "song.wav", lyrics=lyrics)
    assert len(result["sheet"]) == 2
    assert result["sheet"][0]["text"] == "first line here"
    # Columns, not a pre-rendered string -- the page measures and positions.
    for column, name in result["sheet"][0]["chords"]:
        assert isinstance(column, int)
        assert isinstance(name, str)


def test_lyrics_without_timestamps_produce_no_sheet(song):
    result = analyze(song, [parse_key("D")], "song.wav", lyrics="just some words")
    assert result["sheet"] == []


def test_analyze_names_the_relative_key_for_the_warning(song):
    result = analyze(song, [parse_key("D")], "song.wav")
    assert result["relative"] == "A minor"


def test_silence_has_no_chords_to_report(tmp_path):
    path = tmp_path / "quiet.wav"
    sf.write(path, np.zeros(22050 * 2), 22050)
    result = analyze(str(path), [parse_key("D")], "quiet.wav")
    assert result["segments"] >= 1


def test_audio_shorter_than_a_window_is_rejected_clearly(tmp_path):
    path = tmp_path / "tiny.wav"
    sf.write(path, np.zeros(100), 22050)
    with pytest.raises(ValueError, match="no chords found"):
        analyze(str(path), [parse_key("D")], "tiny.wav")
