"""chordrec: reference implementation of the chord detector.

Python is the research and evaluation side of this project. It prototypes the
algorithm, learns the parameters, and produces the fixtures that the Rust core
is tested against. It is never run on a user's audio -- the shipped path is
Rust compiled to wasm, reading the parameters this side exports.
"""

from .chroma import chroma, stft_magnitude, pitch_class_matrix, frame_times

__all__ = ["chroma", "stft_magnitude", "pitch_class_matrix", "frame_times"]
