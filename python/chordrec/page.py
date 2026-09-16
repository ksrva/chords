"""The page `chordrec.web` serves. Markup only -- no analysis lives here.

Kept out of `web.py` so the server module reads as a server. The chord sheet is
positioned in the browser rather than with spaces: lyrics are set in a serif at
proportional widths, so a column index means nothing on screen, and the only
way to put a chord over the right syllable is to measure the text before it.
`measureText` on a canvas does that in one call per chord.

PAGE is a raw string. It holds JavaScript, and in a normal string Python
eats the escapes first -- a JS "\\n" becomes a real newline and splits the
literal, a regex "\\." raises a SyntaxWarning. Raw keeps backslashes for the
language that is going to run them.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chords</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink:#000; --paper:#fff; --soft:#666; --hair:#dcdcdc;
    /* Anthropic sets its text in Tiempos, which is licensed and cannot be
       embedded here. Source Serif 4 is the nearest freely-licensed face:
       same Times-derived skeleton, similar contrast, sturdier at text
       sizes. The stack falls back to a local serif if the CDN is down. */
    --serif:"Source Serif 4","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body {
    margin:0; background:var(--paper); color:var(--ink);
    font:17px/1.5 var(--serif); -webkit-font-smoothing:antialiased;
  }
  button, input { font-family:var(--serif); }

  header {
    display:flex; align-items:baseline; gap:20px;
    padding:22px 40px; border-bottom:1px solid var(--ink);
  }
  header h1 { font-size:26px; font-weight:600; margin:0; letter-spacing:-.01em; }
  header p { margin:0; color:var(--soft); font-size:15px; font-style:italic; }

  .shell { display:grid; grid-template-columns:340px minmax(0,1fr); min-height:calc(100% - 68px); }
  aside { border-right:1px solid var(--ink); padding:30px 30px 48px; }
  main  { padding:36px 48px 64px; min-width:0; }

  .field { margin-bottom:30px; }
  .field > h2 {
    font-size:11px; font-weight:400; letter-spacing:.16em; text-transform:uppercase;
    color:var(--soft); margin:0 0 10px; padding-bottom:7px; border-bottom:1px solid var(--hair);
  }
  .hint { display:block; font-size:13px; color:var(--soft); margin:0 0 10px; line-height:1.45; }
  .hint code { font-family:ui-monospace,Menlo,monospace; font-size:12px; }

  input[type=file] { width:100%; font-size:13px; color:var(--soft); }
  input[type=file]::file-selector-button {
    font:13px var(--serif); padding:6px 12px; margin-right:10px; cursor:pointer;
    background:var(--paper); color:var(--ink); border:1px solid var(--ink); border-radius:0;
  }
  input[type=file]::file-selector-button:hover { background:var(--ink); color:var(--paper); }

  .keys { display:grid; grid-template-columns:repeat(6,1fr); gap:-1px 0; }
  .keys button {
    font-size:14px; padding:9px 0; cursor:pointer; background:var(--paper);
    color:var(--ink); border:1px solid var(--hair); margin:0 -1px -1px 0;
  }
  .keys button:hover { border-color:var(--ink); position:relative; z-index:1; }
  .keys button[aria-pressed=true] {
    background:var(--ink); color:var(--paper); border-color:var(--ink);
    position:relative; z-index:2;
  }

  .go {
    margin-top:26px; width:100%; padding:13px; cursor:pointer; font-size:15px;
    background:var(--ink); color:var(--paper); border:1px solid var(--ink);
    letter-spacing:.04em;
  }
  .go:hover:not(:disabled) { background:var(--paper); color:var(--ink); }
  .go:disabled { background:var(--paper); color:var(--hair); border-color:var(--hair); cursor:default; }
  .note { font-size:12.5px; color:var(--soft); margin:9px 0 0; font-style:italic; }

  .empty { color:var(--soft); font-style:italic; font-size:17px; margin:0; }

  .headline { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; margin:0 0 6px; }
  .headline .from { font-size:44px; letter-spacing:-.02em; font-weight:400; }
  .headline .arr  { font-size:30px; color:var(--soft); line-height:1; }
  .headline .to   { font-size:44px; letter-spacing:-.02em; font-weight:600;
                    border-bottom:2px solid var(--ink); }
  .shift { margin:0 0 4px; font-size:17px; }
  .meta  { margin:0 0 34px; font-size:13px; color:var(--soft); }
  .meta .sep { padding:0 7px; }

  .cols { display:grid; grid-template-columns:minmax(300px,380px) minmax(0,1fr); gap:0 56px; align-items:start; }
  @media (max-width:1100px) { .cols { grid-template-columns:1fr; gap:44px 0; } }

  h3.rule {
    font-size:11px; font-weight:400; letter-spacing:.16em; text-transform:uppercase;
    color:var(--soft); margin:0 0 2px; padding-bottom:8px; border-bottom:1px solid var(--ink);
  }

  table { width:100%; border-collapse:collapse; font-size:17px; }
  td { padding:9px 0; border-bottom:1px solid var(--hair); vertical-align:middle; }
  td.was { width:64px; color:var(--soft); }
  td.now { width:64px; font-weight:600; }
  td.pct { width:52px; text-align:right; font-size:13px; color:var(--soft); }
  td.viz { padding-left:14px; }
  .bar { height:2px; background:var(--ink); min-width:5px; }

  .sheet { margin-top:2px; }
  .line { position:relative; padding-top:24px; margin-bottom:16px; }
  .line .lyric { white-space:pre; font-size:18px; }
  .line .chord {
    position:absolute; top:0; font-size:14px; white-space:pre;
    border-bottom:1px solid var(--ink); padding-bottom:1px;
  }
  .err { border-left:2px solid var(--ink); padding:12px 16px; font-size:14px; white-space:pre-wrap; }
</style>
</head>
<body>

<header>
  <h1>chords</h1>
</header>

<div class="shell">
  <aside>
    <div class="field">
      <h2>audio</h2>
      <label class="hint" for="f">wav, mp3, flac, aiff, ogg</label>
      <input type="file" id="f" accept=".wav,.mp3,.flac,.aiff,.aif,.ogg,.caf,audio/*">
    </div>

    <div class="field">
      <h2>lyrics &middot; optional</h2>
      <label class="hint" for="l"><code>.lrc</code> timed lyrics</label>
      <input type="file" id="l" accept=".lrc,.txt,text/plain">
    </div>

    <div class="field">
      <h2>your range</h2>
      <div class="keys" id="keys"></div>
    </div>

    <button class="go" id="go" disabled>Analyze</button>
    <p class="note" id="note">Choose a file and at least one key.</p>
  </aside>

  <main id="out">
    <p class="empty">Nothing yet.</p>
  </main>
</div>

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

const TEXTY = /\.(lrc|txt|srt)$/i;

function refresh() {
  const chosen = fileEl.files[0];
  if (chosen && TEXTY.test(chosen.name)) {
    goEl.disabled = true;
    noteEl.textContent = "That looks like lyrics. Put it in the lyrics field instead.";
    return;
  }
  const ready = fileEl.files.length > 0 && picked.size > 0;
  goEl.disabled = !ready;
  noteEl.textContent = ready
    ? "A four-minute track takes about a second."
    : "Choose a file and at least one key.";
}
fileEl.onchange = refresh;
refresh();

goEl.onclick = async () => {
  const file = fileEl.files[0];
  goEl.disabled = true;
  goEl.textContent = "Listening";
  outEl.innerHTML = '<p class="empty">Listening&hellip;</p>';
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
    outEl.innerHTML = '<div class="err">' + esc(e.message) + "</div>";
  } finally {
    goEl.textContent = "Analyze";
    refresh();
  }
};

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));

function render(d) {
  const rows = d.chords.map(c => `
    <tr>
      <td class="was">${esc(c.was)}</td>
      <td class="now">${esc(c.now)}</td>
      <td class="pct">${Math.round(c.share * 100)}%</td>
      <td class="viz"><div class="bar" style="width:${Math.max(c.share * 100, 1)}%"></div></td>
    </tr>`).join("");

  const head = d.semitones === 0
    ? `<span class="from">${esc(d.from_key)}</span>`
    : `<span class="from">${esc(d.from_key)}</span>
       <span class="arr">&rarr;</span>
       <span class="to">${esc(d.to_key)}</span>`;

  const sheet = d.sheet && d.sheet.length ? `
    <div>
      <h3 class="rule">chord sheet</h3>
      <div class="sheet" id="sheet">${d.sheet.map(lineHTML).join("")}</div>
    </div>` : "";

  outEl.innerHTML = `
    <div class="headline">${head}</div>
    <p class="shift">${esc(d.shift_text)}</p>
    <p class="meta">${esc(d.name)}<span class="sep">&middot;</span>${esc(d.duration)}
       <span class="sep">&middot;</span>${d.segments} chord segments</p>
    <div class="cols">
      <div>
        <h3 class="rule">chords</h3>
        <table><tbody>${rows}</tbody></table>
      </div>
      ${sheet}
    </div>`;

  if (d.sheet && d.sheet.length) placeChords();
}

function lineHTML(line) {
  const chords = line.chords.map(([col, name]) =>
    `<span class="chord" data-col="${col}">${esc(name)}</span>`).join("");
  return `<div class="line">${chords}<span class="lyric">${esc(line.text)}</span></div>`;
}

// Lyrics are set in a proportional serif, so a column index is not a position.
// Measure the text preceding each chord and place it at that pixel offset --
// the same thing a printed chord sheet does with a fixed-width font, minus the
// fixed-width font.
function placeChords() {
  const ctx = document.createElement("canvas").getContext("2d");
  document.querySelectorAll("#sheet .line").forEach(line => {
    const lyric = line.querySelector(".lyric");
    const text = lyric.textContent;
    ctx.font = getComputedStyle(lyric).font;
    let guard = 0;
    line.querySelectorAll(".chord").forEach(chord => {
      const col = Math.min(+chord.dataset.col, text.length);
      let x = ctx.measureText(text.slice(0, col)).width;
      if (x < guard) x = guard;                 // never let two chords collide
      chord.style.left = x + "px";
      guard = x + chord.getBoundingClientRect().width + 10;
    });
  });
}
</script>
</body>
</html>
"""
