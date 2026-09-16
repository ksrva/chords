"""A local web UI for trying the detector on your own music.

    python -m chordrec.web
    open http://localhost:8000

Built on `http.server` from the standard library. A web framework would be one
import and several dependencies, and this needs to do exactly two things --
serve one page and accept one file -- so the dependency is not worth taking.
Nothing here is on the algorithm path; it calls `recognize` and `transpose` and
formats what comes back.

This is a local testing tool, not a deployment. It binds to localhost, runs
single-threaded per request, and holds the uploaded audio only long enough to
decode it. Do not put it on a network.

Uploads arrive as a multipart form, parsed with `email.parser`. Python 3.13
removed `cgi`, but the MIME machinery `cgi.FieldStorage` wrapped is still in
the standard library and still does the work. A raw request body was enough
while there was one file; the optional second one -- an .lrc of timed lyrics --
is what makes a form worth the parsing.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .chart import build_chart, parse_lrc, song_vocabulary
from .page import PAGE
from .recognize import SR, load_audio, recognize
from .transpose import chord_totals, parse_key, suggest, transpose_label
from .vocab import parse_label

# Uploads are read into memory before hitting disk, so the ceiling is real
# rather than advisory. 60 MB is a long lossless track.
MAX_UPLOAD_BYTES = 60 * 1024 * 1024

# Songs, not four-chord demos: chords last seconds, and the latency the long
# window costs is irrelevant when reading a file. Matches the transpose CLI.
SONG_SMOOTHING = 21

# A sheet someone plays from wants one chord per phrase, not one per fraction
# of a bar. Measured in chart.py: these put 79% of placed chords inside the
# song's real chord set, against 70% at the settings above.
CHART_SMOOTHING = 41
CHART_MIN_DURATION = 1.0

# Extensions that mean "this is the lyrics file, in the wrong box".
LYRIC_SUFFIXES = {".lrc", ".txt", ".srt", ".vtt"}


def parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """Fields from a multipart form, as {name: (filename, bytes)}.

    `cgi.FieldStorage` did this until Python 3.13 removed it, but it was only
    ever a wrapper over the email package's MIME parser -- which is still in
    the standard library and still correct. Synthesising the headers is enough
    to make a request body parse as the MIME document it already is.
    """
    prologue = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=email_policy).parsebytes(prologue + body)
    fields: dict[str, tuple[str | None, bytes]] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = (part.get_filename(), part.get_payload(decode=True) or b"")
    return fields


def analyze(path: str, comfortable: list[int], name: str, top: int = 8,
            lyrics: str | None = None) -> dict:
    """Run the whole pipeline on one file and shape it for the page."""
    audio = load_audio(path, SR)
    smoothing = CHART_SMOOTHING if lyrics else SONG_SMOOTHING
    min_duration = CHART_MIN_DURATION if lyrics else 0.15
    intervals, labels = recognize(audio, SR, smoothing=smoothing,
                                  min_duration=min_duration)
    if not labels:
        raise ValueError("no chords found -- is the file shorter than 200 ms?")

    durations = [float(e - s) for s, e in intervals]
    result = suggest(labels, comfortable, durations)
    from_key, to_key, shift = result["from_key"], result["to_key"], result["semitones"]

    totals = chord_totals(labels, durations)
    span = sum(weight for _, weight in totals) or 1.0
    chords, covered = [], 0.0
    for label, weight in totals[:top]:
        moved = transpose_label(label, shift)
        share = weight / span
        covered += share
        chords.append({
            "was": from_key.chord(parse_label(label)),
            "now": to_key.chord(parse_label(moved)),
            "share": share,
        })

    minutes, seconds = divmod(int(len(audio) / SR), 60)
    if shift == 0:
        shift_text = "No change needed."
    else:
        direction = "up" if shift > 0 else "down"
        plural = "" if abs(shift) == 1 else "s"
        shift_text = f"Transpose {direction} {abs(shift)} semitone{plural}."

    sheet = []
    if lyrics:
        timed = parse_lrc(lyrics)
        if timed:
            vocabulary = song_vocabulary(labels, durations)
            for line in build_chart(intervals, labels, timed, to_key, shift,
                                    restrict_to=vocabulary):
                # Columns, not a rendered string: the page sets lyrics in a
                # proportional serif and measures where each chord belongs.
                sheet.append({
                    "text": line.text,
                    "chords": [[col, name] for col, name in line.chords],
                })

    return {
        "sheet": sheet,
        "name": name,
        "duration": f"{minutes}:{seconds:02d}",
        "segments": len(labels),
        "from_key": str(from_key),
        "to_key": str(to_key),
        "relative": str(from_key.relative),
        "semitones": shift,
        "shift_text": shift_text,
        "margin": result["margin"],
        "covered": covered,
        "chords": chords,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "chordrec"

    def log_message(self, fmt, *args):     # quieter than the default
        print(f"  {self.command} {self.path.split('?')[0]} -> {args[1]}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self._send(404, b"not found", "text/plain")
            return

        query = parse_qs(parsed.query)
        name = (query.get("name") or ["audio"])[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "no audio uploaded"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": f"file is over {MAX_UPLOAD_BYTES // 1048576} MB"})
            return

        try:
            comfortable = [parse_key(k) for k in (query.get("keys") or [""])[0].split(",") if k]
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if not comfortable:
            self._json(400, {"error": "pick at least one key"})
            return

        body = self.rfile.read(length)
        lyrics: str | None = None
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/"):
            fields = parse_multipart(body, content_type)
            if "audio" not in fields:
                self._json(400, {"error": "no audio uploaded"})
                return
            filename, body = fields["audio"]
            name = filename or name
            if "lyrics" in fields and fields["lyrics"][1].strip():
                lyrics = fields["lyrics"][1].decode("utf-8", errors="replace")

        suffix = os.path.splitext(name)[1] or ".wav"
        if suffix.lower() in LYRIC_SUFFIXES:
            # Two adjacent file pickers, and the decoder's own complaint about
            # this is "Format not recognised", which says nothing about what to
            # do. Name the mistake instead.
            self._json(400, {"error": f"{name} looks like a lyrics file. "
                                      "Put it in the lyrics field and choose a "
                                      "recording for the audio field."})
            return

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(body)
            tmp.close()
            self._json(200, analyze(tmp.name, comfortable, name, lyrics=lyrics))
        except Exception as exc:
            traceback.print_exc()
            if suffix.lower() in (".m4a", ".aac", ".mp4"):
                message = (f"{suffix} is not supported. Convert it first: "
                           f"afconvert -f WAVE -d LEI16 in{suffix} out.wav")
            elif "Format not recognised" in str(exc):
                message = f"Could not decode {name}. Is it really audio?"
            else:
                message = f"{type(exc).__name__}: {exc}"
            self._json(400, {"error": message})
        finally:
            os.unlink(tmp.name)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="localhost by default; this is a test tool, not a server")
    args = p.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  chords -- http://{args.host}:{args.port}\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
        server.server_close()


if __name__ == "__main__":
    main()
