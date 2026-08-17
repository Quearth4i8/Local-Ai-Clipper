# 🔥 Local AI Clip Finder

Find the **gold moments** in long videos — podcasts, interviews, streams, lectures —
and get precise timestamps, category scores and suggested hooks, so you can cut the
Shorts yourself.

**100% local. 100% free. No API keys. No cloud.** After installation it works with
your network cable unplugged.

It is *not* an auto-editor. It is the intelligence layer: **"where are the 12 best
45-second moments in this 2-hour video, and why?"**

---

## What you get

```
#1 — 94/100                                     [controversial]
00:47:18 → 00:48:02   (44s)

Hook         ████████████████████  24/25
Payoff       ███████████████████   23/25
Emotion      ███████████████       14/15
Curiosity    ███████████████       14/15
Standalone   ████████████████████  10/10
Editability  ██████████████████     9/10

Why:   Opens on a blunt contradiction of common advice, gives one concrete
       example, and lands a clean conclusion. Needs no earlier context.
Title: "The biggest mistake people make when they start"
Transcript: "..."
```

Exported as **JSON**, **CSV** and a **copy-paste text** block, with an optional
one-click FFmpeg cut of the selected clips — **with animated captions burned in**.

---

## Animated captions

The exported clips can carry word-by-word highlight captions: the phrase sits in
white, and each word turns green and pops slightly **exactly as it is spoken**.

```
IN  MARCH  I WALKED          ← "MARCH" is green + 118% size on that frame
```

This is driven by Whisper's **word-level timestamps**, so the highlight lands on
the real word, not on an estimate. Rendering is done by generating an ASS
subtitle file (one event per word, with `\t` easing the scale) and burning it in
with FFmpeg/libass.

Toggle it with the **Captions** checkbox above the results, and tune it in
**Settings → Animated captions**: font, size, the three colours, position,
words-on-screen, uppercase, pop size and pop speed. **▶ Preview captions** in the
player renders a 6-second sample so you can dial the style in without
re-exporting everything.

| | |
|---|---|
| Default font | `Arial Rounded MT Bold` — the roundest face Windows ships |
| Other installed options | Segoe UI Black, Arial Black, Impact, Bahnschrift, Cooper Black |
| Custom fonts | drop a `.ttf` in `assets/fonts/` — no install needed ([details](assets/fonts/README.md)) |
| Highlight colour | `#22C55E` |

Burning captions **re-encodes** the video (you cannot draw on pixels with a
stream copy), so it is slower than a plain cut — but it uses **NVENC** on an
NVIDIA card: two ~50-second clips took **3.3 s** on an RTX 3050. Unchecking
Captions goes back to instant lossless stream-copy cuts.

### Caption sync

Whisper infers word times from attention alignment, not from acoustic onsets,
so individual words scatter. Measured against real audio onsets (193 samples,
`small` model): the median word timestamp is actually ~67 ms *early*, but the
p10→p90 spread is **226 ms** and the worst word lands **+223 ms late**. Late is
what reads as "the caption is behind the speaker" — early is barely noticeable.

Two things counteract it:

- **Switching at the middle of the pause between two words** instead of at the
  next word's start time. Both ends of a pause are estimates, so the midpoint
  averages the two errors — and it is invisible, because nobody is speaking
  during the gap.
- **`captions.time_offset`** (default `-0.05 s`), a global shift, exposed as the
  **Sync** slider in Settings. Drag it left if captions still feel behind.

Measured effect of the two together: words landing noticeably late (>60 ms)
dropped from **22% → 8%**, worst-case late from +223 ms → +173 ms.

Two things that do *not* help, both tested and rejected:

- `vad_filter: false` — makes alignment **worse** (spread 226 ms → 258 ms, one
  word +1149 ms late). Leave VAD on.
- Blaming FFmpeg's seek — measured at +0 to +16 ms, i.e. under one frame.

Perfect sync would need forced alignment (WhisperX and friends). If you want to
push accuracy further without that, `whisper.model: turbo` aligns better than
`small`.

---

## Quick start (Windows)

```bash
git clone https://github.com/Quearth4i8/Local-Ai-Clipper.git
cd Local-Ai-Clipper
```

Then:

```
1. Run install.bat        (Python deps + FFmpeg + Ollama + the LLM model)
2. Run start.bat          (opens http://127.0.0.1:8420 in your browser)
3. Select video → Analyse video
```

That's it. `install.bat` is idempotent — safe to re-run if something fails
partway through. It installs, in order:

| Step | What | Notes |
|---|---|---|
| 1 | Finds Python 3.10–3.12 | falls back to whatever `python` is on PATH |
| 2 | Creates `.venv` | reused if it already exists |
| 3 | `requirements-core.txt` | required |
| 4 | `requirements-cuda.txt` | optional GPU support, ~1 GB — failure here is **not** fatal |
| 5 | FFmpeg via winget | skipped if already on PATH |
| 6 | Ollama + `qwen2.5:7b-instruct` | ~4.7 GB model download |

The Whisper model (~460 MB for `small`) downloads itself on the first analysis
into `cache/models/`. Nothing else is fetched at runtime — after this, the app
works fully offline.

Verify your setup any time with:

```bash
.venv\Scripts\python main.py check
```

---

## How it works

```
VIDEO
  │  FFmpeg  (stream to disk, never loaded into RAM)
  ▼
16 kHz mono WAV
  │  faster-whisper on CUDA, word-level timestamps
  ▼
Timestamped transcript  ───────────────► cached in cache/<hash>/
  │  sentence + pause + topic-shift segmentation
  ▼
Natural speech units  (every clip edge lands on one of these)
  │  offline pattern detector: hooks, emotion, story, curiosity, payoff (EN + FR)
  ▼
Heuristic hotspots  ──► used to guide the LLM, and as a no-LLM fallback
  │
  ▼
PASS 1 — local LLM reads the transcript in ~7 min blocks and returns
         scored candidate moments (strict JSON, sentence indices)
  │
  ▼
BOUNDARY OPTIMISATION — for each top candidate, the LLM sees the clip plus the
         surrounding lines and moves the in/out points to the best natural cut
  │
  ▼
OVERLAP MERGING — 01:20-02:00 + 01:24-02:08 + 01:29-02:15 become ONE clip
  │
  ▼
PASS 2 — the finalists are compared head-to-head, re-scored, deduplicated
  │
  ▼
DIVERSITY SELECTION (MMR) — not 10 near-identical educational clips
  │
  ▼
TOP CLIPS  →  UI · JSON · CSV · TXT · optional MP4 cuts
```

### Why the scoring is structured

The LLM is never asked *"is this viral?"*. It scores six independent axes against
a written rubric:

| Category | Max | The question it answers |
|---|---|---|
| **Hook** | 25 | Do the first 3 seconds stop a scroll? |
| **Payoff** | 25 | Does the clip actually deliver something? |
| **Emotion** | 15 | Does it make you feel anything? |
| **Curiosity** | 15 | Do you need to know what happens next? |
| **Standalone** | 10 | Does a stranger understand it without the video? |
| **Editability** | 10 | Clean in-point, clean out-point? |

The prompt explicitly teaches the `HOOK → CONTEXT → TENSION → PAYOFF` shape, and
states that a brilliant moment needing three minutes of prior context must score
*below* a slightly duller moment that stands alone. Weights are configurable in
`config.yaml` and in the Settings panel.

---

## Model recommendations — RTX 3050 (8 GB)

Whisper and the LLM never run at the same time: Whisper is explicitly unloaded and
VRAM is released before the LLM stage starts.

### Whisper (`whisper.model`)

| Model | VRAM | Speed on a 3050 | When to use |
|---|---|---|---|
| `small` | ~1.0 GB | ~10-14× realtime | **Default.** Plenty for finding moments. |
| `medium` | ~2.8 GB | ~5-7× realtime | Better punctuation → better sentence edges. |
| `turbo` (large-v3-turbo) | ~4.0 GB | ~7-9× realtime | **Best quality/speed.** Recommended upgrade. |
| `large-v3` | ~5.5 GB | ~2-3× realtime | Only if accuracy is critical. |
| `tiny` / `base` | <0.6 GB | very fast | Quick tests only — weak punctuation hurts boundaries. |

A 2-hour podcast with `small` on the 3050 transcribes in roughly 10-15 minutes,
and is then cached forever.

### Local LLM (`llm.model`)

| Model | Size | Notes |
|---|---|---|
| `qwen2.5:7b-instruct` | 4.7 GB | **Default.** Excellent instruction-following and JSON, solid in French. |
| `llama3.1:8b-instruct-q4_K_M` | 4.9 GB | Strong English judgement. |
| `mistral-nemo:12b-instruct-q4_K_M` | 7.1 GB | Best French. Tight on 8 GB — drop `num_ctx` to 4096. |
| `qwen2.5:3b-instruct` | 2.0 GB | 2-3× faster, noticeably blunter judgement. Good for long videos. |
| `gemma2:9b-instruct-q4_K_M` | 5.8 GB | Good alternative if you dislike Qwen's phrasing. |

Pull any of them with `ollama pull <name>`, then pick it in **Settings → Local LLM**.

Two things worth knowing about 7B-class models: **emotion** is their weakest scoring
axis (they routinely give 0/15 to genuinely moving stories), and their titles are
serviceable but rarely great. If either matters to you, `mistral-nemo:12b` is a
clear step up — and both are cosmetic: clip *selection* and *boundaries*, which is
what this tool exists for, hold up well at 7B.

For **French**, `whisper.model: small` is the weak link, not the LLM — French
punctuation and accents come out noticeably better with `medium` or `turbo`, and
sentence boundaries are what every clip edge is built on.

---

## Changing the local LLM

**Ollama (default)**

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```
Then Settings → *Model*, or in `config.yaml`:

```yaml
llm:
  backend: ollama
  model: llama3.1:8b-instruct-q4_K_M
  base_url: http://127.0.0.1:11434
```

**llama.cpp / LM Studio / vLLM (any OpenAI-compatible local server)**

```yaml
llm:
  backend: openai_compatible
  model: my-local-model
  base_url: http://127.0.0.1:8080/v1
  api_key: ""          # leave empty; local servers don't need one
```

Everything else — prompts, JSON repair, scoring, ranking — is backend-agnostic.
Adding a third backend means subclassing `LLMBackend` in `app/llm/base.py`.

---

## Analysing a video

**UI:** `start.bat` → **Select video** → adjust clips / min / max / language → **Analyse video**.

Then, per clip: **▶ Preview** (plays the exact range in-page), **⧉ Copy timestamp**,
**⧉ Copy transcript**, **✂ Export this clip**. Top bar: **Export ▾** (JSON / CSV / TXT)
and **✂ Export selected clips**.

**Command line:**

```bash
.venv\Scripts\python main.py analyze "D:\videos\podcast.mp4"
.venv\Scripts\python main.py analyze podcast.mkv --clips 20 --language fr --whisper-model turbo
.venv\Scripts\python main.py analyze podcast.mp4 --force-transcribe --transcripts
.venv\Scripts\python main.py check          # environment diagnostics
```

Results are written to `output/<name>_<timestamp>.{json,csv,txt}`.

---

## Configuration (`config.yaml`)

Everything the spec asks for is configurable:

| Setting | Key |
|---|---|
| Whisper model / device / compute type | `whisper.model`, `whisper.device`, `whisper.compute_type` |
| LLM backend / model / server | `llm.backend`, `llm.model`, `llm.base_url` |
| GPU or CPU | `whisper.device: auto \| cuda \| cpu` |
| Min / max clip duration | `clips.min_duration`, `clips.max_duration` |
| Number of clips | `clips.target_count` |
| Scoring weights | `scoring.weights.*` |
| Candidate window size | `clips.block_seconds`, `clips.block_overlap_seconds` |
| Language | `general.language` (`auto`, `en`, `fr`, `ar`, …) |
| Diversity strength | `clips.diversity_lambda` |
| Export mode | `export.mode` (`copy` = instant, `precise` = frame-accurate) |
| Caption style | `captions.*` (font, colours, size, position, pop) |
| Caption encoder | `export.encoder` (`auto` uses NVENC when present) |

The Settings panel writes back to the same file.

### Adding a language

Whisper already handles ~100 languages via `general.language`. The offline pattern
detector ships English and French; add a third by adding one key to `PATTERNS` in
[`app/candidates/heuristics.py`](app/candidates/heuristics.py). Nothing else changes —
unknown languages fall back to the English patterns, and the LLM prompt always asks
for titles in the transcript's own language.

---

## Project structure

```
clipping/
├── app/
│   ├── config.py               config load/merge/save
│   ├── models.py               Word / Sentence / Transcript / Candidate / VideoInfo
│   ├── pipeline.py             orchestrator + job manager + progress
│   ├── video/ffmpeg_tools.py   probe, audio extraction, cutting, preview streaming
│   ├── transcription/
│   │   ├── whisper_engine.py   faster-whisper + VRAM release
│   │   └── cuda_setup.py       finds the pip-installed cuBLAS/cuDNN DLLs
│   ├── segmentation/segmenter.py   sentences, pauses, topic shifts, blocks
│   ├── candidates/
│   │   ├── heuristics.py       EN/FR pattern lexicons (hook, emotion, story, …)
│   │   └── generator.py        window enumeration + offline scoring
│   ├── llm/
│   │   ├── base.py             backend abstraction + retries
│   │   ├── ollama_backend.py   Ollama
│   │   ├── openai_backend.py   llama.cpp / LM Studio / vLLM
│   │   ├── prompts.py          the viral-editor prompts + JSON schemas
│   │   └── json_repair.py      salvages malformed / truncated LLM JSON
│   ├── scoring/scorer.py       pass 1 scoring + boundary optimisation
│   ├── ranking/ranker.py       dedup/merge, pass 2, diversity (MMR)
│   ├── cache/store.py          transcript + result cache
│   ├── export/exporters.py     JSON / CSV / TXT / FFmpeg cuts
│   └── ui/
│       ├── server.py           FastAPI (127.0.0.1 only)
│       ├── filedialog.py       native Windows file picker
│       └── static/             index.html · style.css · app.js
├── cache/      transcripts, Whisper models, thumbnails
├── output/     exported JSON/CSV/TXT and cut clips
├── config.yaml · requirements.txt · main.py · install.bat · start.bat
```

---

## Requirements

* **Windows 10/11** (the pipeline itself is cross-platform; the installer is Windows)
* **Python 3.10 – 3.12** (3.11 recommended; 3.13 wheels are not always available)
* **FFmpeg** on PATH — `winget install Gyan.FFmpeg`
* **Ollama** — https://ollama.com/download
* **GPU:** any NVIDIA card with ≥4 GB VRAM and a recent driver. The CUDA Toolkit
  is *not* required — `requirements-cuda.txt` pulls wheels that ship the DLLs, and
  `app/transcription/cuda_setup.py` registers them at runtime. In practice
  `nvidia-cublas-cu12` alone is enough; `nvidia-cudnn-cu12` is the big optional one.
* **CPU fallback** is automatic — everything still works, just slower
  (`int8` Whisper, and Ollama will use CPU inference).

Dependencies are split so a stalled 1 GB download can't break the install:

| File | Required? | What |
|---|---|---|
| `requirements-core.txt` | yes | faster-whisper, ctranslate2, FastAPI, PyYAML, requests |
| `requirements-cuda.txt` | no | cuBLAS + cuDNN, for GPU transcription |
| `requirements.txt` | — | just includes both, for a one-shot `pip install -r` |

Disk: ~2 GB for Python packages (+1 GB with CUDA), 0.5–5 GB per Whisper model,
~5 GB for the LLM.

---

## Performance notes

* Whisper and the LLM are **never resident at the same time**; Whisper is deleted
  and VRAM freed before pass 1, and Ollama is asked to unload after the run
  (`llm.unload_after_analysis`).
* Transcripts are cached by content fingerprint — re-analysing the same file with
  different scoring settings **skips Whisper entirely**.
* The video is never loaded into memory: FFmpeg streams audio to a temp WAV which
  is deleted immediately after transcription.
* Long videos are processed block by block, so RAM stays flat whether the file is
  20 minutes or 5 hours.

**Measured on this machine** (RTX 3050 8 GB, `small` + `qwen2.5:7b-instruct`), a
7.5-minute two-speaker interview:

| Stage | GPU | CPU fallback |
|---|---|---|
| Transcription | ~15 s | ~50 s |
| Pass 1 (2 blocks) | ~29 s | same (LLM-bound) |
| Boundary optimisation (5 clips) | ~13 s | same |
| Pass 2 | ~8 s | same |
| **Total** | **63 s** | ~100 s |

Extrapolated to a **2-hour podcast**: transcription 12–18 min (once, then cached),
pass 1 ≈ 8–12 min, boundaries ≈ 3–4 min, pass 2 < 1 min — roughly **25–35 minutes
the first time, ~15 minutes on any re-run** since the transcript is cached.

To go faster: `whisper.model: small`, `clips.block_seconds: 600`,
`clips.boundary_candidates: 12`, or switch to `qwen2.5:3b-instruct`.

---

## Troubleshooting

**"FFmpeg not found"**
`winget install Gyan.FFmpeg`, then open a **new** terminal (PATH is only read at
launch). Or set `general.ffmpeg_path` in `config.yaml` to the folder containing
`ffmpeg.exe`.

**"Cannot reach Ollama at http://127.0.0.1:11434"**
Run `ollama serve` in a terminal and leave it open (`start.bat` tries to do this
for you). Check with `ollama list`.

**"Model 'qwen2.5:7b-instruct' is not installed"**
`ollama pull qwen2.5:7b-instruct`.

**GPU pill says "CPU mode", or the log says `cublas64_12.dll is not found`**
CTranslate2 could not load cuBLAS/cuDNN. Those come from two ~500 MB wheels that
`install.bat` installs separately, and they are the most likely thing to stall on
a slow connection. Retry them on their own:

```
.venv\Scripts\pip install -r requirements-cuda.txt --timeout 30 --retries 20
```

then run `main.py check`. Update your NVIDIA driver if it persists. Nothing
breaks meanwhile — the app detects the failure mid-transcription and finishes the
job on the CPU automatically (you will see "Retrying on the CPU…" in the log).

**Out of VRAM during pass 1**
Lower `llm.num_ctx` to 4096, use a smaller LLM (`qwen2.5:3b-instruct`), and make
sure nothing else is using the GPU. Check with `nvidia-smi`.

**The LLM returns nothing / analysis falls back to heuristics**
Open the live log in the progress panel. Usually the model isn't installed, or
`num_ctx` is too small for a 7-minute block — lower `clips.block_seconds` to 300.

**Video preview is black / won't play**
Browsers only decode MP4-H.264/AAC and WebM natively. For MKV, HEVC or AV1 press
**⟳ Transcoded preview** — FFmpeg transcodes just that section on the fly.

**Captions show the wrong font / fall back to something plain**
libass matches on the font's **family name**, not the filename. Check the exact
name in Settings → Animated captions → Font (the dropdown only lists fonts that
exist). For a dropped-in file, `Poppins-ExtraBold.ttf` is usually the family
`Poppins ExtraBold`.

**Captions feel behind the speaker**
Drag **Settings → Animated captions → Sync** to the left (more negative). It
shifts every caption earlier; `-80 ms` cuts noticeably-late words to under 2%.
Going past about `-120 ms` starts to feel ahead of the voice instead. See
[Caption sync](#caption-sync) for the measurements behind the default.

**Two phrases drawn on top of each other**
Fixed. Phrase timelines are clamped so a phrase always leaves the screen before
the next one appears — `tail_hold + lead_in_max` used to be able to exceed a
short pause between phrases, and both drew at the same position.

**Captions run past the edge of the frame**
`captions.max_chars: 0` fits the line to the frame automatically (video width,
font size and margins). Set a number to override it. A word count alone is not
enough — four long words are far wider than four short ones.

**Words missing from the captions**
The caption layer never drops a transcribed word (there is one caption event per
word, asserted in the tests). If text is missing, Whisper did not transcribe it:
try `whisper.model: turbo`, which handles background music and fast speech much
better than `small`.

**Captions are too big / too small / cover the subject**
`captions.font_size_ratio` is a fraction of the *smaller* video side, so it
scales correctly for both 9:16 and 16:9. Move them with `captions.margin_v_ratio`
(fraction of height from the bottom) or set `captions.position: center`.

**Caption export is slow**
Burning re-encodes. Check that `export.encoder: auto` is picking up NVENC —
`main.py check` and the log will show it. Without an NVIDIA GPU it falls back to
libx264, which is several times slower.

**Exported clips start slightly early**
`export.mode: copy` snaps cuts to the nearest keyframe (instant, lossless). For
frame-accurate cuts set `export.mode: precise` — it re-encodes and is slower.

**Whisper produces repeated/hallucinated lines in silence**
Keep `whisper.vad_filter: true` and `condition_on_previous_text: false` (defaults).

**"Whisper found no speech at all"**
The video has no dialogue for the tool to read — gameplay footage, music, b-roll.
This app finds clips from what people *say*, so it needs talking: a podcast,
interview, stream commentary, lecture or vlog. If you are certain there is speech,
the app has already retried with voice-detection disabled, so check that the audio
track isn't a silent/wrong capture device, then try `whisper.model: turbo`.

**"This video is only Ns long, shorter than the minimum clip length"**
You are feeding it a clip, not a source video. Point it at the long original you
want to mine — or lower `clips.min_duration`.

**Analysis found very few clips**
That is often correct — the prompt is deliberately strict. If you want a wider
net, raise `clips.target_count`, raise `clips.max_candidates_pass1`, and lower
`clips.overlap_iou_threshold` to `0.3`.

---

## Design notes

* **Sentence-indexed boundaries.** The LLM never returns raw seconds — it returns
  indices into a numbered list of speech units. Timestamps come from Whisper's
  word timings, so a clip physically cannot start mid-word.
* **Degrades instead of failing.** No LLM? The offline detector still produces
  ranked candidates. Malformed JSON? `json_repair` salvages complete objects out
  of truncated arrays. CUDA missing? CPU int8.
* **Local by construction.** The server binds `127.0.0.1`, and the only two
  backends both point at localhost. There is no code path to a paid API.
