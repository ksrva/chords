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


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chords</title>
<style>
  :root { --ink:#1a1a1a; --dim:#6b6b6b; --line:#d8d4cc; --bg:#f7f5f0; --accent:#8a5a2b; }
  * { box-sizing: border-box; }
  body {
    margin:0; padding:48px 16px; background:var(--bg); color:var(--ink);
    font:16px/1.55 Georgia, "Times New Roman", serif;
  }
  main { max-width:660px; margin:0 auto; }
  h1 { font-size:28px; font-weight:normal; margin:0 0 4px; letter-spacing:.01em; }
  .sub { color:var(--dim); font-style:italic; margin:0 0 32px; }
  fieldset { border:1px solid var(--line); padding:20px; margin:0 0 20px; background:#fff; }
  legend { padding:0 8px; color:var(--dim); font-size:13px; text-transform:uppercase;
           letter-spacing:.08em; font-family:system-ui, sans-serif; }
  label { display:block; margin-bottom:6px; font-size:14px; color:var(--dim);
          font-family:system-ui, sans-serif; }
  input[type=file], input[type=text] {
    width:100%; padding:9px; border:1px solid var(--line); background:#fff;
    font:14px system-ui, sans-serif; color:var(--ink);
  }
  .keys { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
  .keys button {
    border:1px solid var(--line); background:#fff; padding:6px 11px; cursor:pointer;
    font:13px system-ui, sans-serif; color:var(--ink); min-width:42px;
  }
  .keys button[aria-pressed=true] { background:var(--ink); color:#fff; border-color:var(--ink); }
  .go { margin-top:18px; width:100%; padding:12px; border:none; cursor:pointer;
        background:var(--ink); color:#fff; font:15px Georgia, serif; letter-spacing:.02em; }
  .go:disabled { background:var(--dim); cursor:progress; }
  .note { font-size:13px; color:var(--dim); margin-top:10px; font-family:system-ui, sans-serif; }
  #out { margin-top:28px; }
  .card { border:1px solid var(--line); background:#fff; padding:22px; }
  .file { font-size:13px; color:var(--dim); font-family:system-ui, sans-serif;
          margin:0 0 14px; word-break:break-all; }
  .key { font-size:26px; margin:0 0 2px; }
  .key .to { color:var(--accent); }
  .shift { color:var(--dim); font-style:italic; margin:0 0 20px; }
  table { width:100%; border-collapse:collapse; font-size:15px; }
  th { text-align:left; font:12px system-ui, sans-serif; text-transform:uppercase;
       letter-spacing:.07em; color:var(--dim); font-weight:normal;
       border-bottom:1px solid var(--line); padding:0 0 6px; }
  td { padding:7px 0; border-bottom:1px solid #efece6; }
  td.was { width:70px; color:var(--dim); }
  td.now { width:70px; font-weight:bold; color:var(--accent); }
  th.pct, td.pct { width:58px; text-align:right; color:var(--dim); font-size:13px;
           font-family:system-ui, sans-serif; }
  .bar { height:9px; background:var(--accent); opacity:.75; }
  .cover { margin:16px 0 0; font-size:13px; color:var(--dim);
           font-family:system-ui, sans-serif; }
  .maybe { margin:6px 0 0; font-size:13px; color:var(--dim);
           font-family:system-ui, sans-serif; }
  .sheet { margin-top:26px; padding-top:20px; border-top:1px solid var(--line); }
  .sheet h2 { font:12px system-ui, sans-serif; text-transform:uppercase;
              letter-spacing:.07em; color:var(--dim); font-weight:normal; margin:0 0 14px; }
  .sheet pre { font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
               margin:0; white-space:pre; overflow-x:auto; }
  .sheet .c { color:var(--accent); font-weight:bold; }
  .sheet .w { color:var(--ink); }
  .err { border-left:3px solid #a33; background:#fdf5f5; padding:14px;
         font:14px system-ui, sans-serif; white-space:pre-wrap; }
</style>
</head>
<body>
<main>
  <h1>chords</h1>
  <p class="sub">Find a song&rsquo;s key, and move it to one you can sing.</p>

  <fieldset>
    <legend>audio</legend>
    <label for="f">wav, mp3, flac, aiff or ogg &mdash; not m4a</label>
    <input type="file" id="f" accept=".wav,.mp3,.flac,.aiff,.aif,.ogg,.caf,audio/*">
  </fieldset>

  <fieldset>
    <legend>lyrics <span style="text-transform:none;letter-spacing:0">(optional)</span></legend>
    <label for="l">An <code>.lrc</code> file &mdash; timed lyrics. Add one and you
      get a chord sheet instead of a summary.</label>
    <input type="file" id="l" accept=".lrc,.txt,text/plain">
  </fieldset>

  <fieldset>
    <legend>keys you sing well</legend>
    <label>Pick the tonics your voice sits in. Minor songs move to the minor of
      the same letter.</label>
    <div class="keys" id="keys"></div>
    <button class="go" id="go" disabled>Analyze</button>
    <p class="note" id="note">Choose a file and at least one key.</p>
  </fieldset>

  <div id="out"></div>
</main>

<script>
const NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const picked = new Set(["D","G"]);
const keysEl = document.getElementById("keys");
const fileEl = document.getElementById("f");
const lrcEl  = document.getElementById("l");
const goEl   = document.getElementById("go");
const noteEl = document.getElementById("note");
const outEl  = document.getElementById("out");

NOTES.forEach(n => {
  const b = document.createElement("button");
  b.textContent = n;
  b.setAttribute("aria-pressed", picked.has(n));
  b.onclick = () => {
    picked.has(n) ? picked.delete(n) : picked.add(n);
    b.setAttribute("aria-pressed", picked.has(n));
    refresh();
  };
  keysEl.appendChild(b);
});

function refresh() {
  const ready = fileEl.files.length > 0 && picked.size > 0;
  goEl.disabled = !ready;
  noteEl.textContent = ready
    ? "Ready. A four-minute track takes about a second."
    : "Choose a file and at least one key.";
}
fileEl.onchange = refresh;
refresh();

goEl.onclick = async () => {
  const file = fileEl.files[0];
  goEl.disabled = true;
  goEl.textContent = "Listening\\u2026";
  outEl.innerHTML = "";
  try {
    const form = new FormData();
    form.append("audio", file);
    if (lrcEl.files.length) form.append("lyrics", lrcEl.files[0]);
    const qs = new URLSearchParams({ keys: [...picked].join(","), name: file.name });
    const res = await fetch("/analyze?" + qs, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "analysis failed");
    render(data);
  } catch (e) {
    outEl.innerHTML = '<div class="err">' + escape(e.message) + "</div>";
  } finally {
    goEl.disabled = false;
    goEl.textContent = "Analyze";
    refresh();
  }
};

const escape = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));

function sheet(d) {
  if (!d.sheet || !d.sheet.length) return "";
  const body = d.sheet.map(l =>
    (l.chords ? '<span class="c">' + escape(l.chords) + "</span>\n" : "") +
    '<span class="w">' + escape(l.text) + "</span>"
  ).join("\n\n");
  return '<div class="sheet"><h2>chord sheet</h2><pre>' + body + "</pre></div>";
}

function render(d) {
  const rows = d.chords.map(c => `
    <tr>
      <td class="was">${escape(c.was)}</td>
      <td class="now">${escape(c.now)}</td>
      <td class="pct">${Math.round(c.share * 100)}%</td>
      <td><div class="bar" style="width:${Math.max(c.share * 100, 1.5)}%"></div></td>
    </tr>`).join("");

  const heading = d.semitones === 0
    ? `${escape(d.from_key)} <span class="to">&mdash; already yours</span>`
    : `${escape(d.from_key)} &rarr; <span class="to">${escape(d.to_key)}</span>`;

  const warn = d.margin < 0.10
    ? `<p class="maybe">Not certain &mdash; could be ${escape(d.relative)}.</p>`
    : "";

  outEl.innerHTML = `
    <div class="card">
      <p class="file">${escape(d.name)} &middot; ${escape(d.duration)} &middot;
         ${d.segments} chord segments</p>
      <p class="key">${heading}</p>
      <p class="shift">${escape(d.shift_text)}</p>
      <table>
        <thead><tr><th>played</th><th>you play</th><th class="pct">share</th><th></th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="cover">These ${d.chords.length} cover
         ${Math.round(d.covered * 100)}% of the song.</p>
      ${warn}
      ${sheet(d)}
    </div>`;
}
</script>
</body>
</html>
"""


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
                rendered = line.render()
                sheet.append({
                    "chords": rendered[0] if len(rendered) == 2 else "",
                    "text": rendered[-1],
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
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(body)
            tmp.close()
            self._json(200, analyze(tmp.name, comfortable, name, lyrics=lyrics))
        except Exception as exc:
            traceback.print_exc()
            hint = ""
            if suffix.lower() in (".m4a", ".aac", ".mp4"):
                hint = " -- m4a is not supported; convert with afconvert first"
            self._json(400, {"error": f"{type(exc).__name__}: {exc}{hint}"})
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
