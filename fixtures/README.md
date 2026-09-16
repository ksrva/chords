# fixtures

Shared ground truth for the Rust-vs-Python comparison tests (M8).

Audio is **not** committed here. Two kinds of fixture instead:

- **Synthetic** — generated on demand by `chordrec.signals`. Deterministic, so
  both implementations can produce the same input from the same code without
  shipping a waveform.
- **Recorded** — local piano recordings, kept outside the repo, with only their
  `.lab` label files committed. Labels are derived (from MIDI, or from a chart
  plus a click track) rather than hand-annotated.
