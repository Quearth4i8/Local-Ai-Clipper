/* ==========================================================================
   Local AI Clip Finder — front-end
   ========================================================================== */
const $ = (id) => document.getElementById(id);
const api = async (url, opts = {}) => {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
};

const state = {
  video: null,
  jobId: null,
  poll: null,
  result: null,
  clips: [],
  selected: new Set(),
  active: null,
  config: null,
  transcoded: false,
};

const STAGES = [
  ['probe', 'Read video'], ['audio', 'Audio'], ['transcribe', 'Transcribe'],
  ['segment', 'Segment'], ['pass1', 'Pass 1'], ['boundaries', 'Boundaries'],
  ['pass2', 'Pass 2'], ['finalize', 'Finalise'],
];
const SCORE_MAX = { hook: 25, payoff: 25, emotion: 15, curiosity: 15, standalone: 10, editability: 10 };
const SCORE_LABEL = { hook: 'Hook', payoff: 'Payoff', emotion: 'Emotion',
  curiosity: 'Curiosity', standalone: 'Standalone', editability: 'Editability' };

/* ------------------------------------------------------------------ utils */
function tc(s) {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  const mm = String(m).padStart(2, '0'), ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function fmtDur(s) {
  s = Math.max(0, Math.round(s || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${String(s % 60).padStart(2, '0')}s`
                : `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}
function slowHint(stage) {
  return {
    transcribe: 'Whisper reports progress only as speech is decoded; long silences look like pauses.',
    pass1: 'The first block also waits for the LLM to load into VRAM (20–90s the first time).',
    boundaries: 'Each clip is one LLM call, so this scales with clips.boundary_candidates.',
    pass2: 'All finalists are compared in a single large LLM call.',
  }[stage] || 'Check the live log below for the last thing that happened.';
}
function esc(t) {
  return String(t ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function toast(msg, kind = 'info', ms = 4200) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, ms);
}

/* ------------------------------------------------------------------ theme */
(function initTheme() {
  const saved = localStorage.getItem('cf-theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('btn-theme').onclick = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('cf-theme', next);
  };
})();

/* SVG gradient used by the score rings */
(function injectDefs() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '0'); svg.setAttribute('height', '0');
  svg.style.position = 'absolute';
  svg.innerHTML = `<defs><linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#8b7cff"/><stop offset="55%" stop-color="#ffb03d"/>
    <stop offset="100%" stop-color="#ff7a3d"/></linearGradient></defs>`;
  document.body.appendChild(svg);
})();

/* ----------------------------------------------------------------- health */
async function refreshHealth() {
  try {
    const h = await api('/api/health');
    state.config = h.config;
    setPill('pill-ffmpeg', h.ffmpeg.ok, h.ffmpeg.ok ? 'FFmpeg' : 'FFmpeg missing', h.ffmpeg.error);
    setPill('pill-gpu', h.cuda.ok, h.cuda.ok ? 'GPU · CUDA' : 'CPU mode',
      h.cuda.ok ? '' : 'CUDA is not available to CTranslate2 — Whisper will run on the CPU.',
      h.cuda.ok ? 'ok' : 'warn');
    const llmOk = h.llm.ok && h.llm.model_installed !== false;
    setPill('pill-llm', llmOk, llmOk ? `LLM · ${shortModel(h.llm.model || '')}` : 'LLM offline',
      h.llm.error || '');
    const list = $('model-list');
    if (list && h.llm.models) list.innerHTML = h.llm.models.map((m) => `<option value="${esc(m)}">`).join('');
    // Only ever populate the form from disk once, so the 30s refresh never
    // stomps on values the user is in the middle of editing.
    if (!state.formReady) { applyConfigToUI(h.config); state.formReady = true; }
  } catch (e) {
    ['pill-ffmpeg', 'pill-gpu', 'pill-llm'].forEach((id) => setPill(id, false, id.split('-')[1], e.message));
  }
}
function shortModel(m) { return m.length > 18 ? m.slice(0, 17) + '…' : m; }
function setPill(id, ok, label, title, cls) {
  const el = $(id);
  el.className = `pill ${cls || (ok ? 'ok' : 'bad')}`;
  el.textContent = label;
  el.title = title || '';
}

/* ---------------------------------------------------------- video picking */
async function loadVideo(path) {
  try {
    const data = await api('/api/video', { method: 'POST', body: { path } });
    if (data.cancelled) return;
    showVideo(data);
  } catch (e) { toast(e.message, 'err', 6000); }
}
function showVideo(data) {
  state.video = data.video;
  $('dropzone').classList.add('hidden');
  $('video-meta').classList.remove('hidden');
  $('vm-name').textContent = data.video.filename;
  const v = data.video;
  $('vm-chips').innerHTML = [
    ['⏱', v.duration_tc], ['⬛', v.resolution], ['🎞', `${v.fps} fps`],
    ['💾', `${v.size_mb} MB`], ['🔊', v.acodec || '—'],
  ].map(([i, t]) => `<span class="chip">${i} ${esc(t)}</span>`).join('');
  $('cache-note').classList.toggle('hidden', !data.cached_transcript);
}
$('btn-browse').onclick = async () => {
  const btn = $('btn-browse');
  btn.disabled = true; btn.textContent = 'Opening picker…';
  try {
    const data = await api('/api/browse', { method: 'POST' });
    if (!data.cancelled) showVideo(data);
  } catch (e) {
    toast(`${e.message} — paste the full path instead.`, 'err', 7000);
  } finally { btn.disabled = false; btn.textContent = 'Select video'; }
};
$('btn-load-path').onclick = () => {
  const p = $('path-input').value.trim();
  if (p) loadVideo(p);
};
$('path-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btn-load-path').click(); });
$('btn-change').onclick = () => {
  $('video-meta').classList.add('hidden');
  $('dropzone').classList.remove('hidden');
};

/* drag & drop — Explorer usually exposes the path as text; File objects do not */
let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault(); dragDepth++; $('drop-overlay').classList.add('on');
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('dragleave', (e) => {
  e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; $('drop-overlay').classList.remove('on'); }
});
window.addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0; $('drop-overlay').classList.remove('on');
  const dt = e.dataTransfer;
  let path = (dt.getData('text/plain') || dt.getData('text/uri-list') || '').trim();
  if (path.startsWith('file:///')) path = decodeURIComponent(path.slice(8)).replace(/\//g, '\\');
  if (path) { loadVideo(path); return; }
  const f = dt.files && dt.files[0];
  if (f && f.path) { loadVideo(f.path); return; }
  if (f) {
    $('path-input').value = '';
    $('path-input').placeholder = `Browsers hide the folder — use "Select video" for ${f.name}`;
    toast('Your browser does not expose the file path. Use "Select video" instead.', 'info', 6000);
  }
});

/* --------------------------------------------------------------- analysis */
$('btn-analyze').onclick = async () => {
  if (!state.video) return;
  const patch = {
    general: { language: $('opt-lang').value },
    clips: {
      target_count: +$('opt-count').value,
      min_duration: +$('opt-min').value,
      max_duration: +$('opt-max').value,
    },
  };
  try {
    await api('/api/config', { method: 'POST', body: { config: patch, save: true } });
    const res = await api('/api/analyze', {
      method: 'POST',
      body: { path: state.video.path, force_transcribe: $('opt-force').checked },
    });
    state.jobId = res.job_id;
    startProgressUI();
    state.poll = setInterval(pollJob, 700);
  } catch (e) { toast(e.message, 'err', 7000); }
};

function startProgressUI() {
  $('progress-card').classList.remove('hidden');
  $('progress-card').classList.remove('failed');
  $('pg-error').classList.add('hidden');
  $('pg-hint').classList.add('hidden');
  $('results').classList.add('hidden');
  $('btn-analyze').disabled = true;
  $('pg-steps').innerHTML = STAGES.map(([k, l]) => `<span class="step" data-k="${k}">${l}</span>`).join('');
  $('pg-log').textContent = '';
  $('progress-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function pollJob() {
  if (!state.jobId) return;
  let job;
  try { job = await api(`/api/job/${state.jobId}`); }
  catch { return; }

  $('pg-stage').textContent = job.stage_label || '…';
  $('pg-message').textContent = job.message || '';
  $('pg-pct').textContent = `${Math.round(job.progress * 100)}%`;
  $('pg-bar').style.width = `${Math.max(2, job.progress * 100)}%`;
  $('pg-elapsed').textContent = `${fmtDur(job.elapsed)} total`;
  $('pg-stage-elapsed').textContent =
    job.status === 'running' ? `${fmtDur(job.stage_elapsed || 0)} in this step` : '';

  // Some single steps legitimately take minutes (a cold 7B model load, one big
  // transcript block). Say so, rather than letting a still bar look like a hang.
  const quiet = job.status === 'running' && (job.since_update || 0) > 25;
  $('pg-hint').textContent = quiet
    ? `Still working — this step has no sub-progress to report. ${slowHint(job.stage)}`
    : '';
  $('pg-hint').classList.toggle('hidden', !quiet);

  const idx = STAGES.findIndex(([k]) => k === job.stage);
  document.querySelectorAll('.step').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
    el.classList.toggle('done', idx >= 0 && i < idx);
  });
  const log = $('pg-log');
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  log.textContent = (job.logs || []).join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;

  if (['done', 'error', 'cancelled'].includes(job.status)) {
    clearInterval(state.poll);
    state.poll = null;
    state.jobId = job.status === 'done' ? state.jobId : state.jobId;
    $('btn-analyze').disabled = false;
    $('pg-hint').classList.add('hidden');
    if (job.status === 'done') {
      document.querySelectorAll('.step').forEach((el) => { el.classList.add('done'); el.classList.remove('active'); });
      $('pg-bar').style.width = '100%';
      $('progress-card').classList.remove('failed');
      renderResults(job.result);
      toast(`${job.result.clips.length} clips found in ${Math.round(job.elapsed)}s`, 'ok');
    } else if (job.status === 'error') {
      showFailure(job.error || 'Analysis failed');
      toast('Analysis failed — see the panel for details', 'err', 9000);
    } else {
      $('pg-stage').textContent = 'Cancelled';
      toast('Analysis cancelled', 'info');
    }
  }
}

/* A failed run must LOOK failed. Leaving a half-full bar frozen mid-stage is
   indistinguishable from a hang, which is exactly how it gets reported. */
function showFailure(message) {
  $('progress-card').classList.add('failed');
  $('pg-stage').textContent = 'Analysis failed';
  $('pg-message').textContent = '';
  $('pg-pct').textContent = '—';
  $('pg-stage-elapsed').textContent = '';
  document.querySelectorAll('.step').forEach((el) => el.classList.remove('active'));
  const box = $('pg-error');
  box.textContent = message;
  box.classList.remove('hidden');
  $('progress-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

$('btn-cancel').onclick = async () => {
  if (state.jobId) await api(`/api/job/${state.jobId}/cancel`, { method: 'POST' });
};

/* ---------------------------------------------------------------- results */
function renderResults(result) {
  state.result = result;
  state.clips = result.clips || [];
  state.selected = new Set(state.clips.map((c) => c.rank));
  $('results').classList.remove('hidden');
  $('sel-all').checked = true;

  const st = result.stats || {};
  const types = Object.entries(st.types || {}).map(([k, v]) => `${v}× ${k.replace(/_/g, ' ')}`).join(' · ');
  $('res-summary').textContent =
    `${state.clips.length} clips · ${st.candidates_pass1 || 0} candidates analysed · `
    + `${st.llm_used ? st.llm_model : 'offline heuristics only'} · ${Math.round(st.elapsed || 0)}s`
    + (types ? ` — ${types}` : '');

  $('clip-list').innerHTML = state.clips.map(clipCard).join('');
  requestAnimationFrame(() => {
    document.querySelectorAll('.sbar .fill').forEach((el) => { el.style.width = el.dataset.w; });
    document.querySelectorAll('.score-ring .val').forEach((el) => { el.style.strokeDashoffset = el.dataset.off; });
  });
  wireClipCards();
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clipCard(c) {
  const bars = Object.keys(SCORE_MAX).map((k) => {
    const v = (c.scores && c.scores[k]) || 0;
    const pct = Math.round((v / SCORE_MAX[k]) * 100);
    return `<div class="sbar"><b>${SCORE_LABEL[k]}</b>
      <span class="track"><span class="fill" data-w="${pct}%"></span></span>
      <i>${Math.round(v)}/${SCORE_MAX[k]}</i></div>`;
  }).join('');
  const C = 151;
  const off = C - C * Math.max(0, Math.min(100, c.score)) / 100;
  return `
  <article class="clip" data-rank="${c.rank}">
    <input class="clip-sel" type="checkbox" checked data-rank="${c.rank}" title="Include when exporting clips">
    <div class="clip-top">
      <div class="rank">#${c.rank}</div>
      <div class="clip-head">
        <div class="clip-title">${esc(c.title || '(untitled)')}</div>
        <div class="clip-sub">
          <span class="tc">${c.start_tc} → ${c.end_tc}</span>
          <span>${Math.round(c.duration)}s</span>
          <span class="type-chip" data-t="${esc(c.type)}">${esc(String(c.type).replace(/_/g, ' '))}</span>
          ${c.confidence ? `<span title="LLM confidence">conf ${Math.round(c.confidence * 100)}%</span>` : ''}
        </div>
      </div>
      <div class="score-ring">
        <svg width="56" height="56"><circle class="track" cx="28" cy="28" r="24"></circle>
          <circle class="val" cx="28" cy="28" r="24" stroke-dasharray="${C}"
                  stroke-dashoffset="${C}" data-off="${off}"></circle></svg>
        <span>${Math.round(c.score)}</span>
      </div>
    </div>
    <div class="bars">${bars}</div>
    <div class="reason"><strong>Why:</strong> ${esc(c.reason || '—')}</div>
    ${c.verdict ? `<div class="verdict"><strong>Ranking verdict:</strong> ${esc(c.verdict)}</div>` : ''}
    <details class="transcript"><summary>Transcript</summary><p>${esc(c.transcript || '')}</p></details>
    <div class="clip-actions">
      <button class="btn primary small act-play" data-rank="${c.rank}">▶ Preview</button>
      <button class="btn ghost small act-copy" data-rank="${c.rank}">⧉ Copy timestamp</button>
      <button class="btn ghost small act-copytext" data-rank="${c.rank}">⧉ Copy transcript</button>
      <button class="btn ghost small act-cut" data-rank="${c.rank}">✂ Export this clip</button>
    </div>
  </article>`;
}

function wireClipCards() {
  document.querySelectorAll('.clip').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('button, input, summary, a')) return;
      playClip(+el.dataset.rank);
    });
  });
  document.querySelectorAll('.act-play').forEach((b) => b.onclick = () => playClip(+b.dataset.rank));
  document.querySelectorAll('.act-copy').forEach((b) => b.onclick = () => {
    const c = clipOf(+b.dataset.rank);
    navigator.clipboard.writeText(`${c.start_tc} → ${c.end_tc}  (${Math.round(c.duration)}s)  ${c.title}`);
    toast('Timestamp copied', 'ok', 2000);
  });
  document.querySelectorAll('.act-copytext').forEach((b) => b.onclick = () => {
    navigator.clipboard.writeText(clipOf(+b.dataset.rank).transcript || '');
    toast('Transcript copied', 'ok', 2000);
  });
  document.querySelectorAll('.act-cut').forEach((b) => b.onclick = () => cutClips([+b.dataset.rank], b));
  document.querySelectorAll('.clip-sel').forEach((cb) => cb.onchange = () => {
    const r = +cb.dataset.rank;
    cb.checked ? state.selected.add(r) : state.selected.delete(r);
    $('sel-all').checked = state.selected.size === state.clips.length;
  });
}
const clipOf = (rank) => state.clips.find((c) => c.rank === rank);

$('sel-all').onchange = (e) => {
  state.selected = e.target.checked ? new Set(state.clips.map((c) => c.rank)) : new Set();
  document.querySelectorAll('.clip-sel').forEach((cb) => { cb.checked = e.target.checked; });
};

/* ---------------------------------------------------------------- preview */
const video = $('video-el');
let stopAt = null;

function playClip(rank, transcoded = false) {
  const c = clipOf(rank);
  if (!c || !state.video) return;
  state.active = rank;
  state.transcoded = transcoded;
  document.querySelectorAll('.clip').forEach((el) =>
    el.classList.toggle('active', +el.dataset.rank === rank));

  $('pv-empty').classList.add('hidden');
  $('pv-player').classList.remove('hidden');
  $('pv-title').textContent = c.title || '(untitled)';
  $('pv-time').textContent = `${c.start_tc} → ${c.end_tc}`;
  $('pv-hint').textContent = '';

  const p = encodeURIComponent(state.video.path);
  if (transcoded) {
    stopAt = null;
    video.src = `/api/preview?path=${p}&start=${c.start}&end=${c.end}`;
    video.load();
    video.play().catch(() => {});
    $('pv-hint').textContent = 'Transcoding this section with FFmpeg — playback starts in a few seconds.';
  } else {
    stopAt = c.end;
    const want = `/api/media?path=${p}`;
    if (!video.src.includes('/api/media')) { video.src = want; video.load(); }
    const seek = () => { video.currentTime = c.start; video.play().catch(() => {}); };
    if (video.readyState >= 1) seek();
    else video.addEventListener('loadedmetadata', seek, { once: true });
  }
  $('preview-pane').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

video.addEventListener('timeupdate', () => {
  const c = clipOf(state.active);
  if (!c) return;
  const t = state.transcoded ? c.start + video.currentTime : video.currentTime;
  const pct = Math.max(0, Math.min(1, (t - c.start) / Math.max(0.1, c.end - c.start)));
  $('pv-cursor').style.left = `${pct * 100}%`;
  if (stopAt !== null && video.currentTime >= stopAt) { video.pause(); stopAt = null; }
});
video.addEventListener('error', () => {
  if (!state.transcoded && state.active !== null) {
    $('pv-hint').textContent =
      'This container/codec will not play natively in the browser. Use “Transcoded preview”.';
  }
});
$('btn-replay').onclick = () => { if (state.active !== null) playClip(state.active, state.transcoded); };
$('btn-transcode').onclick = () => { if (state.active !== null) playClip(state.active, true); };

/* ----------------------------------------------------------------- export */
$('btn-export').onclick = (e) => { e.stopPropagation(); $('export-menu').classList.toggle('hidden'); };
document.addEventListener('click', () => $('export-menu').classList.add('hidden'));
$('export-menu').onclick = async (e) => {
  const fmt = e.target.dataset.fmt;
  if (!fmt) return;
  $('export-menu').classList.add('hidden');
  if (fmt === 'save') {
    try {
      const r = await api('/api/export', { method: 'POST', body: { job_id: state.jobId } });
      toast(`Saved JSON + CSV + TXT to ${r.folder}`, 'ok', 6000);
      api('/api/open_folder', { method: 'POST', body: { path: r.folder } }).catch(() => {});
    } catch (err) { toast(err.message, 'err'); }
  } else {
    window.location = `/api/export/${state.jobId}.${fmt}`;
  }
};

$('btn-cut').onclick = () => cutClips([...state.selected], $('btn-cut'));

async function cutClips(ranks, btn) {
  if (!ranks.length) { toast('No clips selected', 'info'); return; }
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = `Cutting ${ranks.length}…`;
  try {
    const r = await api('/api/export_clips', { method: 'POST', body: { job_id: state.jobId, ranks } });
    toast(`${r.clips.length} clip(s) written to ${r.folder}`, 'ok', 6000);
    api('/api/open_folder', { method: 'POST', body: { path: r.folder } }).catch(() => {});
  } catch (e) { toast(e.message, 'err', 8000); }
  finally { btn.disabled = false; btn.textContent = label; }
}

/* ---------------------------------------------------------------- settings */
const WEIGHT_KEYS = ['hook', 'payoff', 'emotion', 'curiosity', 'standalone', 'editability'];

function applyConfigToUI(cfg) {
  if (!cfg) return;
  $('opt-count').value = cfg.clips.target_count;
  $('opt-min').value = cfg.clips.min_duration;
  $('opt-max').value = cfg.clips.max_duration;
  $('opt-lang').value = cfg.general.language;

  $('set-whisper').value = cfg.whisper.model;
  $('set-device').value = cfg.whisper.device;
  $('set-compute').value = cfg.whisper.compute_type;
  $('set-backend').value = cfg.llm.backend;
  $('set-model').value = cfg.llm.model;
  $('set-url').value = cfg.llm.base_url;
  $('set-ctx').value = cfg.llm.num_ctx;
  $('set-block').value = cfg.clips.block_seconds;
  $('set-cap').value = cfg.clips.max_candidates_pass1;
  $('set-boundary').value = String(!!cfg.clips.boundary_optimization);
  $('set-div').value = cfg.clips.diversity_lambda;
  $('val-div').textContent = (+cfg.clips.diversity_lambda).toFixed(2);
  $('set-blend').value = cfg.scoring.heuristic_blend;
  $('val-blend').textContent = (+cfg.scoring.heuristic_blend).toFixed(2);

  $('weights').innerHTML = WEIGHT_KEYS.map((k) => `
    <div class="wrow"><span>${SCORE_LABEL[k]}</span>
      <input type="range" min="0" max="40" step="1" data-w="${k}" value="${cfg.scoring.weights[k]}">
      <b data-wv="${k}">${cfg.scoring.weights[k]}</b></div>`).join('');
  document.querySelectorAll('[data-w]').forEach((el) => {
    el.oninput = () => { document.querySelector(`[data-wv="${el.dataset.w}"]`).textContent = el.value; };
  });
}
$('set-div').oninput = (e) => { $('val-div').textContent = (+e.target.value).toFixed(2); };
$('set-blend').oninput = (e) => { $('val-blend').textContent = (+e.target.value).toFixed(2); };

$('btn-settings').onclick = async () => {
  $('settings-modal').classList.remove('hidden');
  try {
    const c = await api('/api/cache');
    $('cache-info').textContent = `${c.entries.length} cached video(s) · ${c.total_mb} MB · ${c.folder}`;
    $('btn-open-cache').onclick = () => api('/api/open_folder', { method: 'POST', body: { path: c.folder } });
  } catch { $('cache-info').textContent = '—'; }
};
$('btn-close-settings').onclick = () => $('settings-modal').classList.add('hidden');
$('settings-modal').onclick = (e) => { if (e.target.id === 'settings-modal') e.target.classList.add('hidden'); };

$('btn-save-settings').onclick = async () => {
  const weights = {};
  document.querySelectorAll('[data-w]').forEach((el) => { weights[el.dataset.w] = +el.value; });
  const patch = {
    whisper: { model: $('set-whisper').value, device: $('set-device').value,
               compute_type: $('set-compute').value },
    llm: { backend: $('set-backend').value, model: $('set-model').value.trim(),
           base_url: $('set-url').value.trim(), num_ctx: +$('set-ctx').value },
    clips: { block_seconds: +$('set-block').value, max_candidates_pass1: +$('set-cap').value,
             boundary_optimization: $('set-boundary').value === 'true',
             diversity_lambda: +$('set-div').value },
    scoring: { weights, heuristic_blend: +$('set-blend').value },
  };
  try {
    await api('/api/config', { method: 'POST', body: { config: patch, save: true } });
    toast('Settings saved to config.yaml', 'ok');
    $('settings-modal').classList.add('hidden');
    refreshHealth();
  } catch (e) { toast(e.message, 'err'); }
};

$('btn-clear-cache').onclick = async () => {
  if (!confirm('Delete all cached transcripts? They will have to be re-generated.')) return;
  await api('/api/cache', { method: 'DELETE' });
  toast('Cache cleared', 'ok');
  $('cache-info').textContent = '0 cached video(s)';
};

/* ------------------------------------------------------------------- init */
refreshHealth();
setInterval(() => { if (!state.poll) refreshHealth(); }, 30000);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $('settings-modal').classList.add('hidden');
  if (e.key === ' ' && state.active !== null && e.target === document.body) {
    e.preventDefault(); video.paused ? video.play() : video.pause();
  }
});
