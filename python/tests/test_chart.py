"""Tests for the chord sheet: lyric parsing, placement, and the noise filter."""

from __future__ import annotations

import numpy as np
import pytest

from chordrec.chart import (
    ChartLine,
    build_chart,
    parse_lrc,
    render,
    song_vocabulary,
)
from chordrec.transpose import Key


def test_parses_timestamps():
    lines = parse_lrc("[00:11.50]hello\n[01:02.25]world")
    assert lines == [(11.5, "hello"), (62.25, "world")]


def test_parses_two_and_three_digit_fractions():
    assert parse_lrc("[00:01.5]a")[0][0] == pytest.approx(1.5)
    assert parse_lrc("[00:01.50]a")[0][0] == pytest.approx(1.50)
    assert parse_lrc("[00:01.500]a")[0][0] == pytest.approx(1.500)


def test_accepts_colon_as_the_fraction_separator():
    assert parse_lrc("[00:01:50]a")[0][0] == pytest.approx(1.50)


def test_missing_fraction_is_whole_seconds():
    assert parse_lrc("[02:03]a")[0][0] == pytest.approx(123.0)


def test_repeated_line_gets_an_entry_per_timestamp():
    """A chorus is written once with several stamps; each is a real occurrence."""
    lines = parse_lrc("[00:10.00][01:20.00]chorus")
    assert lines == [(10.0, "chorus"), (80.0, "chorus")]


def test_metadata_and_blank_lines_are_skipped():
    lines = parse_lrc("[ti:Title]\n[ar:Artist]\n\n[00:05.00]real words")
    assert lines == [(5.0, "real words")]


def test_lines_come_back_in_time_order():
    lines = parse_lrc("[00:20.00]second\n[00:10.00]first")
    assert [text for _, text in lines] == ["first", "second"]


def test_unstamped_text_is_ignored():
    assert parse_lrc("just a plain line\n[00:01.00]stamped") == [(1.0, "stamped")]


def test_render_puts_chords_above_words():
    line = ChartLine(time=0.0, text="hello world", chords=[(0, "G"), (6, "Em")])
    assert line.render() == ["G     Em", "hello world"]


def test_render_of_a_line_with_no_chords_is_just_the_words():
    assert ChartLine(0.0, "silence", []).render() == ["silence"]


def test_chords_never_overlap_each_other():
    """Two changes close in time must not print on top of one another."""
    intervals = np.array([[0.0, 0.1], [0.05, 0.2], [0.1, 1.0]])
    labels = ["C:maj", "G:maj", "A:min"]
    chart = build_chart(intervals, labels, [(0.0, "a short line here")], Key(0, "maj"))
    columns = [c for c, _ in chart[0].chords]
    names = [n for _, n in chart[0].chords]
    for i in range(1, len(columns)):
        assert columns[i] >= columns[i - 1] + len(names[i - 1]) + 1


def test_a_line_starting_mid_chord_is_not_bare():
    """The chord sounding when the line begins carries over, or the singer
    sees nothing to play at the moment they start singing."""
    intervals = np.array([[0.0, 30.0]])
    chart = build_chart(intervals, ["C:maj"], [(10.0, "words here")], Key(0, "maj"))
    assert chart[0].chords == [(0, "C")]


def test_an_unchanged_chord_is_not_repeated():
    intervals = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    labels = ["C:maj", "C:maj", "C:maj"]
    chart = build_chart(intervals, labels, [(0.0, "a line of some length")], Key(0, "maj"))
    assert [n for _, n in chart[0].chords] == ["C"]


def test_chart_applies_the_transposition():
    intervals = np.array([[0.0, 4.0]])
    chart = build_chart(intervals, ["C:maj"], [(0.0, "words")], Key(2, "maj"), shift=2)
    assert chart[0].chords == [(0, "D")]


def test_chart_spells_for_the_target_key():
    intervals = np.array([[0.0, 4.0]])
    chart = build_chart(intervals, ["G#:maj"], [(0.0, "words")], Key(8, "maj"))
    assert chart[0].chords == [(0, "Ab")]


def test_no_lyrics_is_an_empty_chart():
    assert build_chart(np.zeros((0, 2)), [], [], Key(0, "maj")) == []


def test_vocabulary_keeps_the_common_chords():
    labels = ["C:maj", "G:maj", "F:maj", "B:min"]
    durations = [50.0, 30.0, 15.0, 1.0]
    keep = song_vocabulary(labels, durations, coverage=0.90)
    assert "C:maj" in keep and "G:maj" in keep
    assert "B:min" not in keep          # 1% of the song, almost certainly a misread


def test_vocabulary_of_a_single_chord_song():
    assert song_vocabulary(["C:maj"], [10.0]) == {"C:maj"}


def test_restriction_drops_the_rare_chord_from_the_sheet():
    intervals = np.array([[0.0, 4.0], [4.0, 4.3], [4.3, 9.0]])
    labels = ["C:maj", "B:min", "C:maj"]
    durations = [4.0, 0.3, 4.7]
    keep = song_vocabulary(labels, durations, coverage=0.90)
    chart = build_chart(intervals, labels, [(0.0, "a line here")], Key(0, "maj"),
                        restrict_to=keep)
    assert "Bm" not in [n for _, n in chart[0].chords]


def test_render_joins_lines_with_blank_separators():
    chart = [ChartLine(0.0, "one", [(0, "C")]), ChartLine(1.0, "two", [])]
    assert render(chart).splitlines() == ["C", "one", "", "two"]
