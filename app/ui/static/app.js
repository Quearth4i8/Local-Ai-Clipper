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
  if (!res.ok) throw new Error(errorText(data, res.status));
  return data;
};

/* FastAPI returns 422 `detail` as an ARRAY of objects. Passing that straight to
   new Error() stringifies it to "[object Object]", which tells you nothing. */
function errorText(data, status) {
  const d = data && data.detail;
  if (typeof d === 'string' && d) return d;
  if (Array.isArray(d)) {
    const parts = d.map((e) => {
      const where = Array.isArray(e.loc) ? e.loc.filter((x) => x !== 'body').join('.') : '';
      const got = e.input !== undefined ? ` (got ${JSON.stringify(e.input)})` : '';
      return `${where ? where + ': ' : ''}${e.msg || 'invalid'}${got}`;
    });
    return `Invalid request — ${parts.join('; ')}`;
  }
  if (d && typeof d === 'object') return JSON.stringify(d);
  return `HTTP ${status}`;
}

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
    state.pilAvailable = h.pillow ? !!h.pillow.ok : true;
    if (!state.pilAvailable) $('opt-thumbnail').disabled = true;
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
  wireClipCards();
  loadFormats().then(refreshSelCount);
  refreshSelCount();
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clipCard(c) {
  const mini = Object.keys(SCORE_MAX).map((k) => {
    const pct = Math.round((((c.scores && c.scores[k]) || 0) / SCORE_MAX[k]) * 100);
    return `<div title="${SCORE_LABEL[k]} ${Math.round((c.scores && c.scores[k]) || 0)}/${SCORE_MAX[k]}"><span style="width:${pct}%"></span></div>`;
  }).join('');
  const cls = c.score >= 80 ? '' : (c.score >= 60 ? 'mid' : 'low');
  const thumbAt = c.start + Math.min(2, c.duration / 3);
  const src = state.video
    ? `/api/thumb?path=${encodeURIComponent(state.video.path)}&at=${thumbAt.toFixed(2)}`
    : '';
  return `
  <article class="clip selected" data-rank="${c.rank}">
    <div class="thumb">
      ${src ? `<img loading="lazy" src="${src}" alt=""
                onerror="this.style.display='none';this.nextElementSibling.style.display='grid'">` : ''}
      <div class="thumb-fallback" style="display:${src ? 'none' : 'grid'}">🎬</div>
      <label class="sel-box" title="Include this clip when exporting">
        <input type="checkbox" checked data-rank="${c.rank}">
      </label>
      <div class="score-badge ${cls}">${Math.round(c.score)}<small>SCORE</small></div>
      <div class="thumb-play">▶</div>
      <span class="thumb-tc">${c.start_tc} → ${c.end_tc}</span>
      <span class="thumb-dur">${Math.round(c.duration)}s</span>
    </div>
    <div class="clip-body">
      <div class="clip-meta">
        <span class="rank">#${c.rank}</span>
        <span class="type-chip" data-t="${esc(c.type)}">${esc(String(c.type).replace(/_/g, ' '))}</span>
        ${c.virality >= 55 ? `<span class="chip viral" title="Virality score — how hard this clip grabs a scroll and drives replies/arguments">🔥 ${Math.round(c.virality)}</span>` : ''}
        ${c.confidence ? `<span class="chip" title="How confident the model was">conf ${Math.round(c.confidence * 100)}%</span>` : ''}
      </div>
      <div class="clip-title">${esc(c.title || '(untitled)')}</div>
      <div class="mini-bars">${mini}</div>
      <div class="reason">${esc(c.reason || '—')}</div>
      <div class="clip-actions">
        <button class="btn primary small act-play" data-rank="${c.rank}">▶ Preview</button>
        <button class="btn ghost small act-copy" data-rank="${c.rank}" title="Copy the timecodes">⧉ TC</button>
        <button class="btn ghost small act-cut" data-rank="${c.rank}">✂ Export</button>
      </div>
    </div>
  </article>`;
}

function wireClipCards() {
  document.querySelectorAll('.clip').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('button, input, label, a')) return;
      openPlayer(+el.dataset.rank);
    });
  });
  document.querySelectorAll('.act-play').forEach((b) => b.onclick = () => openPlayer(+b.dataset.rank));
  document.querySelectorAll('.act-copy').forEach((b) => b.onclick = () => {
    const c = clipOf(+b.dataset.rank);
    navigator.clipboard.writeText(`${c.start_tc} → ${c.end_tc}  (${Math.round(c.duration)}s)  ${c.title}`);
    toast('Timecode copied', 'ok', 2000);
  });
  document.querySelectorAll('.act-cut').forEach((b) => b.onclick = () => cutClips([+b.dataset.rank], b));
  document.querySelectorAll('.sel-box input').forEach((cb) => cb.onchange = () => {
    const r = +cb.dataset.rank;
    cb.checked ? state.selected.add(r) : state.selected.delete(r);
    cb.closest('.clip').classList.toggle('selected', cb.checked);
    refreshSelCount();
  });
}
const clipOf = (rank) => state.clips.find((c) => c.rank === rank);

function refreshSelCount() {
  const n = state.selected.size;
  const total = state.clips.length;
  const f = state.formats ? state.formats.size : 1;
  $('sel-count').textContent = n === total ? `All ${total} selected` : `${n} of ${total} selected`;
  $('sel-all').checked = n === total && total > 0;
  $('sel-all').indeterminate = n > 0 && n < total;
  $('btn-cut').disabled = n === 0;
  $('btn-cut').textContent = f > 1
    ? `✂ Export ${n} clips × ${f} formats`
    : `✂ Export ${n} clip${n === 1 ? '' : 's'}`;
}

$('sel-all').onchange = (e) => {
  const on = e.target.checked;
  state.selected = on ? new Set(state.clips.map((c) => c.rank)) : new Set();
  document.querySelectorAll('.sel-box input').forEach((cb) => {
    cb.checked = on;
    cb.closest('.clip').classList.toggle('selected', on);
  });
  refreshSelCount();
};

/* -------------------------------------------------------- campaign rules */
$('btn-campaign-toggle').onclick = () => {
  const open = $('campaign-body').classList.toggle('hidden') === false;
  $('btn-campaign-toggle').setAttribute('aria-expanded', String(open));
};
function syncCampaignState() {
  const on = $('opt-metadata').checked || $('opt-watermark').checked || $('opt-music').checked;
  $('campaign-state').textContent = on ? 'on' : 'off';
  $('campaign-state').classList.toggle('on', on);
}
function saveCampaign() {
  api('/api/config', {
    method: 'POST',
    body: { config: {
      campaign: { rules: $('campaign-rules').value,
                 generate_metadata: $('opt-metadata').checked },
      watermark: {
        enabled: $('opt-watermark').checked,
        path: $('wm-path').value.trim(),
        position: $('wm-position').value,
        scale: +$('wm-scale').value,
        opacity: +$('wm-opacity').value,
        margin: +$('wm-margin').value,
      },
      music: {
        enabled: $('opt-music').checked,
        path: $('music-path').value.trim(),
        volume: +$('music-volume').value,
        fade_seconds: +$('music-fade').value,
      },
    }, save: true },
  }).catch(() => {});
}
let campaignSaveTimer = null;
function saveCampaignDebounced() {
  clearTimeout(campaignSaveTimer);
  campaignSaveTimer = setTimeout(saveCampaign, 800);
}
$('campaign-rules').addEventListener('input', saveCampaignDebounced);
$('opt-metadata').addEventListener('change', () => { syncCampaignState(); saveCampaign(); });

/* ------------------------------------------------------------ watermark */
function syncWatermarkUI() {
  const on = $('opt-watermark').checked;
  $('watermark-controls').classList.toggle('hidden', !on);
  syncCampaignState();
}
function setWatermarkThumb(path) {
  const img = $('wm-thumb');
  const empty = $('wm-thumb-empty');
  if (path) {
    img.src = `/api/media?path=${encodeURIComponent(path)}`;
    img.classList.remove('hidden');
    empty.classList.add('hidden');
  } else {
    img.classList.add('hidden');
    img.removeAttribute('src');
    empty.classList.remove('hidden');
  }
}
$('wm-thumb').addEventListener('error', () => setWatermarkThumb(''));
function syncWatermarkLabels() {
  $('val-wm-scale').textContent = `${Math.round(+$('wm-scale').value * 100)}%`;
  $('val-wm-opacity').textContent = `${Math.round(+$('wm-opacity').value * 100)}%`;
  $('val-wm-margin').textContent = `${Math.round(+$('wm-margin').value * 100)}%`;
}
$('opt-watermark').addEventListener('change', () => { syncWatermarkUI(); saveCampaign(); });
$('wm-position').addEventListener('change', saveCampaign);
['wm-scale', 'wm-opacity', 'wm-margin'].forEach((id) => {
  $(id).addEventListener('input', () => { syncWatermarkLabels(); saveCampaignDebounced(); });
});
$('wm-path').addEventListener('input', () => {
  setWatermarkThumb($('wm-path').value.trim());
  saveCampaignDebounced();
});
/* ---------------------------------------------------------------- music */
function syncMusicUI() {
  const on = $('opt-music').checked;
  $('music-controls').classList.toggle('hidden', !on);
  syncCampaignState();
}
function syncMusicLabels() {
  $('val-music-volume').textContent = `${Math.round(+$('music-volume').value * 100)}%`;
  $('val-music-fade').textContent = `${(+$('music-fade').value).toFixed(1)}s`;
}
$('opt-music').addEventListener('change', () => { syncMusicUI(); saveCampaign(); });
['music-volume', 'music-fade'].forEach((id) => {
  $(id).addEventListener('input', () => { syncMusicLabels(); saveCampaignDebounced(); });
});
$('music-path').addEventListener('input', saveCampaignDebounced);
$('btn-music-browse').onclick = async () => {
  const btn = $('btn-music-browse');
  btn.disabled = true; btn.textContent = 'Opening…';
  try {
    const data = await api('/api/browse_audio', { method: 'POST' });
    if (!data.cancelled) {
      $('music-path').value = data.path;
      saveCampaign();
    }
  } catch (e) {
    toast(`${e.message} — paste the audio path instead.`, 'err', 7000);
  } finally { btn.disabled = false; btn.textContent = 'Browse…'; }
};
$('btn-music-play').onclick = () => {
  const el = $('music-audio-el');
  const path = $('music-path').value.trim();
  if (!path) { toast('No music file set', 'info', 2000); return; }
  if (!el.paused && el.src.includes(encodeURIComponent(path))) {
    el.pause();
    $('btn-music-play').textContent = '▶';
    return;
  }
  el.src = `/api/media?path=${encodeURIComponent(path)}`;
  el.play().catch((e) => toast(`Could not play: ${e.message}`, 'err', 4000));
  $('btn-music-play').textContent = '⏸';
};
$('music-audio-el').addEventListener('ended', () => { $('btn-music-play').textContent = '▶'; });
$('music-audio-el').addEventListener('pause', () => { $('btn-music-play').textContent = '▶'; });

/* ------------------------------------------------------ tighten/thumbnail */
$('opt-tighten').addEventListener('change', () => {
  api('/api/config', { method: 'POST',
    body: { config: { tightening: { enabled: $('opt-tighten').checked } }, save: true } }).catch(() => {});
});
$('opt-thumbnail').addEventListener('change', () => {
  api('/api/config', { method: 'POST',
    body: { config: { thumbnail: { enabled: $('opt-thumbnail').checked } }, save: true } }).catch(() => {});
});

$('btn-wm-browse').onclick = async () => {
  const btn = $('btn-wm-browse');
  btn.disabled = true; btn.textContent = 'Opening…';
  try {
    const data = await api('/api/browse_image', { method: 'POST' });
    if (!data.cancelled) {
      $('wm-path').value = data.path;
      setWatermarkThumb(data.path);
      saveCampaign();
    }
  } catch (e) {
    toast(`${e.message} — paste the image path instead.`, 'err', 7000);
  } finally { btn.disabled = false; btn.textContent = 'Browse…'; }
};

/* ---------------------------------------------------------- format picker */
const PLATFORM_ICON = { '9:16': '📱', '4:5': '🖼', '1:1': '⬛', '16:9': '🖥' };

async function loadFormats() {
  try {
    const f = await api('/api/formats');
    // Only trust ids the server actually offers - anything else would round-trip
    // back on export and be rejected.
    const known = new Set(f.formats.map((x) => x.id));
    const picked = (f.selected || []).map(String).filter((x) => known.has(x));
    state.formats = new Set(picked.length ? picked : ['9:16']);
    $('format-pills').innerHTML = f.formats.map((x) => `
      <label class="fpill ${state.formats.has(x.id) ? 'on' : ''}" data-fmt="${esc(x.id)}">
        <input type="checkbox" ${state.formats.has(x.id) ? 'checked' : ''}>
        <span class="shape" data-r="${esc(x.id)}"></span>
        <span>
          <b>${PLATFORM_ICON[x.id] || ''} ${esc(x.label)} · ${esc(x.id)}</b>
          <small>${esc(x.platforms)} · ${x.width}×${x.height}</small>
        </span>
      </label>`).join('');
    $('format-pills').querySelectorAll('.fpill').forEach((el) => {
      el.onclick = (e) => {
        e.preventDefault();
        const id = el.dataset.fmt;
        if (state.formats.has(id)) {
          if (state.formats.size === 1) { toast('Keep at least one format', 'info', 2500); return; }
          state.formats.delete(id);
        } else state.formats.add(id);
        el.classList.toggle('on', state.formats.has(id));
        el.querySelector('input').checked = state.formats.has(id);
        refreshSelCount();
        saveFormats();
      };
    });
    $('rf-layout').value = f.layout || 'crop';
    $('rf-layout').onchange = saveFormats;
    $('pv-format').innerHTML = f.formats
      .map((x) => `<option value="${esc(x.id)}">${esc(x.id)} · ${esc(x.label)}</option>`).join('');

    const t = f.tracking || {};
    $('fb-tracking').textContent = t.ok
      ? 'Subject tracking on — the crop follows the speaker'
      : (t.reason || 'Centre crop only');
    $('fb-tracking').className = t.ok ? '' : 'warn';
  } catch { /* leave defaults */ }
}

function saveFormats() {
  api('/api/config', {
    method: 'POST',
    body: { config: { reframe: { formats: [...state.formats], layout: $('rf-layout').value } },
            save: true },
  }).catch(() => {});
}

/* ---------------------------------------------------------------- preview */
const video = $('video-el');
let stopAt = null;

function openPlayer(rank) {
  $('player-modal').classList.remove('hidden');
  playClip(rank);
}

function closePlayer() {
  $('player-modal').classList.add('hidden');
  video.pause();
}

function playClip(rank, transcoded = false) {
  const c = clipOf(rank);
  if (!c || !state.video) return;
  state.active = rank;
  state.transcoded = transcoded;

  $('pv-rank').textContent = `#${c.rank}`;
  $('pv-title').textContent = c.title || '(untitled)';
  $('pv-time').textContent = `${c.start_tc} → ${c.end_tc} · ${Math.round(c.duration)}s`;
  $('pv-type').textContent = String(c.type).replace(/_/g, ' ');
  $('pv-type').dataset.t = c.type;
  $('pv-score-text').textContent = `${Math.round(c.score)}/100`;
  $('pv-hint').textContent = '';

  $('pv-bars').innerHTML = Object.keys(SCORE_MAX).map((k) => {
    const v = (c.scores && c.scores[k]) || 0;
    const pct = Math.round((v / SCORE_MAX[k]) * 100);
    return `<div class="sbar"><b>${SCORE_LABEL[k]}</b>
      <span class="track"><span class="fill" data-fill="${pct}%"></span></span>
      <i>${Math.round(v)}/${SCORE_MAX[k]}</i></div>`;
  }).join('');
  $('pv-reason').innerHTML = `<strong>Why:</strong> ${esc(c.reason || '—')}`;
  $('pv-verdict').classList.toggle('hidden', !c.verdict);
  if (c.verdict) $('pv-verdict').innerHTML = `<strong>Ranking verdict:</strong> ${esc(c.verdict)}`;
  $('pv-text').textContent = c.transcript || '';
  requestAnimationFrame(() => {
    $('pv-bars').querySelectorAll('.fill').forEach((el) => { el.style.width = el.dataset.fill; });
  });

  const idx = state.clips.findIndex((x) => x.rank === rank);
  $('btn-prev-clip').disabled = idx <= 0;
  $('btn-next-clip').disabled = idx < 0 || idx >= state.clips.length - 1;

  loadPerfHistory(rank);

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
}

function stepClip(delta) {
  const idx = state.clips.findIndex((c) => c.rank === state.active);
  const next = state.clips[idx + delta];
  if (next) playClip(next.rank);
}

$('btn-close-player').onclick = closePlayer;
$('btn-prev-clip').onclick = () => stepClip(-1);
$('btn-next-clip').onclick = () => stepClip(1);
$('player-modal').onclick = (e) => { if (e.target.id === 'player-modal') closePlayer(); };
$('btn-copy-text').onclick = () => {
  navigator.clipboard.writeText(clipOf(state.active)?.transcript || '');
  toast('Transcript copied', 'ok', 2000);
};
$('btn-cut-one').onclick = () => {
  if (state.active !== null) cutClips([state.active], $('btn-cut-one'));
};

/* ------------------------------------------------------------- performance */
const PLATFORM_LABEL = { tiktok: 'TikTok', instagram: 'Instagram', youtube: 'YouTube', other: 'Other' };

async function loadPerfHistory(rank) {
  const hash = state.result && state.result.stats && state.result.stats.video_hash;
  const box = $('perf-history');
  if (!hash) { box.innerHTML = ''; return; }
  box.innerHTML = '<span class="hint">Loading…</span>';
  try {
    const data = await api(`/api/performance?video_hash=${encodeURIComponent(hash)}`);
    const entries = (data.entries || []).filter((e) => String(e.rank) === String(rank));
    renderPerfHistory(entries);
  } catch { box.innerHTML = ''; }
}

function renderPerfHistory(entries) {
  const box = $('perf-history');
  if (!entries.length) { box.innerHTML = '<span class="hint">No results logged yet.</span>'; return; }
  box.innerHTML = entries.map((e) => `
    <div class="perf-entry" data-id="${e.id}">
      <span class="plat">${esc(PLATFORM_LABEL[e.platform] || e.platform || 'other')}</span>
      <span class="stats">${(e.views || 0).toLocaleString()} views · ${(e.likes || 0).toLocaleString()} likes
        · ${(e.comments || 0).toLocaleString()} comments${e.notes ? ` · ${esc(e.notes)}` : ''}</span>
      <button class="perf-del" data-id="${e.id}" title="Delete">✕</button>
    </div>`).join('');
  box.querySelectorAll('.perf-del').forEach((b) => {
    b.onclick = async () => {
      try {
        await api(`/api/performance/${b.dataset.id}`, { method: 'DELETE' });
        loadPerfHistory(state.active);
      } catch (e) { toast(e.message, 'err'); }
    };
  });
}

$('btn-perf-save').onclick = async () => {
  if (state.active === null) return;
  const btn = $('btn-perf-save');
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    await api('/api/performance', {
      method: 'POST',
      body: {
        job_id: state.jobId, rank: String(state.active),
        platform: $('perf-platform').value,
        views: +$('perf-views').value || 0,
        likes: +$('perf-likes').value || 0,
        comments: +$('perf-comments').value || 0,
        shares: +$('perf-shares').value || 0,
        notes: $('perf-notes').value.trim() || null,
      },
    });
    ['perf-views', 'perf-likes', 'perf-comments', 'perf-shares', 'perf-notes'].forEach((id) => { $(id).value = ''; });
    toast('Performance logged', 'ok', 2500);
    loadPerfHistory(state.active);
  } catch (e) { toast(e.message, 'err', 6000); }
  finally { btn.disabled = false; btn.textContent = label; }
};

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

/* Render a short sample with captions burned in, so the style can be dialled in
   without re-exporting every clip. */
$('btn-captest').onclick = async () => {
  const rank = state.active !== null ? state.active : (state.clips[0] && state.clips[0].rank);
  if (!rank) return;
  const btn = $('btn-captest');
  const label = btn.textContent;
  const fmt = $('pv-format').value || '9:16';
  btn.disabled = true; btn.textContent = 'Rendering…';
  $('pv-hint').textContent = `Rendering a 6-second ${fmt} sample exactly as it will export…`;
  try {
    const res = await fetch('/api/caption_preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: state.jobId, rank, seconds: 6, format: fmt,
                             layout: $('rf-layout').value,
                             captions: $('opt-captions').checked,
                             watermark: $('opt-watermark').checked,
                             music: $('opt-music').checked }),
    });
    if (!res.ok) throw new Error((await res.text()).slice(0, 300));
    const blob = await res.blob();
    if (state.capUrl) URL.revokeObjectURL(state.capUrl);
    state.capUrl = URL.createObjectURL(blob);
    state.transcoded = true;   // times are already clip-relative
    stopAt = null;
    video.src = state.capUrl;
    video.load();
    video.play().catch(() => {});
    $('pv-hint').textContent = `${fmt} export sample — this is exactly what the file will `
      + 'look like. Framing is set above the grid, caption style in Settings.';
  } catch (e) {
    $('pv-hint').textContent = '';
    toast(`Caption preview failed: ${e.message}`, 'err', 9000);
  } finally { btn.disabled = false; btn.textContent = label; }
};

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

/* Report exactly what happened to the metadata step - a silent miss (LLM
   offline, model not installed, a clip's call failing) is the #1 reason it
   looks like "nothing was generated". */
function reportMetadata(metadata) {
  if (!metadata) return;
  if (metadata.error) {
    toast(`Metadata not written: ${metadata.error}`, 'err', 9000);
    return;
  }
  const ok = (metadata.written || []).length;
  const failed = (metadata.failed || []).length;
  if (ok) {
    toast(`Wrote metadata for ${ok}/${metadata.requested} clip(s)`
      + (failed ? ` — ${failed} failed` : ''), 'ok', 6000);
  } else if (failed) {
    toast(`Metadata generation failed for all ${failed} clip(s) — check the local LLM`, 'err', 9000);
  }
}

$('btn-cut').onclick = () => cutClips([...state.selected], $('btn-cut'));

async function cutClips(ranks, btn) {
  if (!ranks.length) { toast('No clips selected', 'info'); return; }
  const withCaptions = $('opt-captions').checked;
  const withMetadata = $('opt-metadata').checked;
  const withWatermark = $('opt-watermark').checked;
  const withTighten = $('opt-tighten').checked;
  const withThumbnail = $('opt-thumbnail').checked;
  const withMusic = $('opt-music').checked;
  const formats = state.formats ? [...state.formats] : ['9:16'];
  const label = btn.textContent;
  const jobs = ranks.length * formats.length;
  btn.disabled = true;
  btn.textContent = `Rendering ${jobs}…`;
  toast(`Rendering ${jobs} file(s): ${formats.join(', ')}`
    + (withCaptions ? ' with captions' : '')
    + (withTighten ? ' + tightened pacing' : '')
    + (withWatermark ? ' + watermark' : '')
    + (withMusic ? ' + music' : '')
    + (withThumbnail ? ' + thumbnails' : '')
    + (withMetadata ? ' + writing metadata' : '')
    + '. Reframing analyses the video, so give it a moment.', 'info', 6000);
  try {
    const r = await api('/api/export_clips', {
      method: 'POST',
      body: { job_id: state.jobId, ranks, captions: withCaptions,
              formats, layout: $('rf-layout').value,
              generate_metadata: withMetadata, campaign_rules: $('campaign-rules').value,
              watermark: withWatermark, tighten: withTighten, thumbnail: withThumbnail,
              music: withMusic },
    });
    toast(`${r.clips.length} file(s) written to ${r.folder}`, 'ok', 6000);
    if (withMetadata) reportMetadata(r.metadata);
    api('/api/open_folder', { method: 'POST', body: { path: r.folder } }).catch(() => {});
  } catch (e) { toast(e.message, 'err', 10000); }
  finally { btn.disabled = false; btn.textContent = label; refreshSelCount(); }
}

/* ------------------------------------------------------- viral compilation */
$('btn-compilation').onclick = () => buildCompilation([...state.selected]);

async function buildCompilation(ranks) {
  // Fewer than 2 picks isn't really a "compilation" choice, it's the whole
  // set — fall back to the strongest moments by virality automatically.
  const useAuto = ranks.length < 2;
  const btn = $('btn-compilation');
  const label = btn.textContent;
  const withCaptions = $('opt-captions').checked;
  const withMetadata = $('opt-metadata').checked;
  const withWatermark = $('opt-watermark').checked;
  const withTighten = $('opt-tighten').checked;
  const withMusic = $('opt-music').checked;
  const formats = state.formats ? [...state.formats] : ['9:16'];
  btn.disabled = true;
  btn.textContent = 'Building…';
  toast(useAuto
    ? 'No 2+ clips selected — auto-picking the highest-virality moments.'
    : `Stitching ${ranks.length} clips into one viral-format cut…`, 'info', 5000);
  try {
    const r = await api('/api/export_compilation', {
      method: 'POST',
      body: {
        job_id: state.jobId,
        ranks: useAuto ? null : ranks,
        captions: withCaptions,
        formats,
        layout: $('rf-layout').value,
        generate_metadata: withMetadata,
        campaign_rules: $('campaign-rules').value,
        watermark: withWatermark,
        tighten: withTighten,
        music: withMusic,
      },
    });
    const order = (r.files[0] && r.files[0].order) || [];
    const seq = order.map((o) => `#${o.rank}`).join(' → ');
    toast(`Compilation saved to ${r.folder} — order: ${seq}`, 'ok', 8000);
    if (withMetadata) reportMetadata(r.metadata);
    api('/api/open_folder', { method: 'POST', body: { path: r.folder } }).catch(() => {});
  } catch (e) { toast(e.message, 'err', 10000); }
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

  applyCaptionsToUI(cfg.captions || {});

  const camp = cfg.campaign || {};
  $('campaign-rules').value = camp.rules || '';
  $('opt-metadata').checked = !!camp.generate_metadata;

  const wm = cfg.watermark || {};
  $('opt-watermark').checked = !!wm.enabled;
  $('wm-path').value = wm.path || '';
  $('wm-position').value = wm.position || 'top_right';
  $('wm-scale').value = wm.scale ?? 0.18;
  $('wm-opacity').value = wm.opacity ?? 0.85;
  $('wm-margin').value = wm.margin ?? 0.04;
  syncWatermarkLabels();
  syncWatermarkUI();
  setWatermarkThumb(wm.path || '');

  const mu = cfg.music || {};
  $('opt-music').checked = !!mu.enabled;
  $('music-path').value = mu.path || '';
  $('music-volume').value = mu.volume ?? 0.15;
  $('music-fade').value = mu.fade_seconds ?? 0.6;
  syncMusicLabels();
  syncMusicUI();

  syncCampaignState();

  const tg = cfg.tightening || {};
  $('opt-tighten').checked = !!tg.enabled;
  $('set-tighten-fillers').checked = tg.remove_fillers !== false;
  $('set-tighten-gap').value = tg.min_gap ?? 0.6;
  $('set-tighten-keep').value = tg.keep_pause ?? 0.35;
  syncTightenLabels();

  const th = cfg.thumbnail || {};
  $('opt-thumbnail').checked = !!th.enabled;
  $('thumb-text-color').value = th.text_color || '#FFFFFF';
  $('thumb-outline-color').value = th.outline_color || '#000000';
  $('thumb-upper').checked = th.uppercase !== false;
  $('thumb-hint').textContent = state.pilAvailable === false
    ? 'Pillow is not installed on the server - thumbnails will not generate. Run: pip install Pillow'
    : 'Uses the strongest detected frame plus the clip\'s hook line.';

  applyWeightsToUI(cfg.scoring.weights);
}

function applyWeightsToUI(weights) {
  $('weights').innerHTML = WEIGHT_KEYS.map((k) => `
    <div class="wrow"><span>${SCORE_LABEL[k]}</span>
      <input type="range" min="0" max="40" step="1" data-w="${k}" value="${weights[k]}">
      <b data-wv="${k}">${weights[k]}</b></div>`).join('');
  // Scoped to #weights: an unscoped [data-w] also matched the score bars in the
  // clip cards, which wrote junk keys like "84%": null into scoring.weights.
  $('weights').querySelectorAll('[data-w]').forEach((el) => {
    el.oninput = () => { document.querySelector(`[data-wv="${el.dataset.w}"]`).textContent = el.value; };
  });
}

function syncTightenLabels() {
  $('val-tighten-gap').textContent = `${(+$('set-tighten-gap').value).toFixed(2)}s`;
  $('val-tighten-keep').textContent = `${(+$('set-tighten-keep').value).toFixed(2)}s`;
}
$('set-tighten-gap').addEventListener('input', syncTightenLabels);
$('set-tighten-keep').addEventListener('input', syncTightenLabels);
$('set-div').oninput = (e) => { $('val-div').textContent = (+e.target.value).toFixed(2); };
$('set-blend').oninput = (e) => { $('val-blend').textContent = (+e.target.value).toFixed(2); };

/* ---------------------------------------------------------------- captions */
async function applyCaptionsToUI(c) {
  $('opt-captions').checked = c.enabled !== false;
  $('cap-size').value = c.font_size_ratio ?? 0.07;
  $('cap-words').value = c.max_words ?? 4;
  $('cap-pos').value = c.position || 'bottom';
  $('cap-margin').value = c.margin_v_ratio ?? 0.16;
  $('cap-scale').value = c.highlight_scale ?? 118;
  $('cap-anim').value = c.animation_ms ?? 130;
  $('cap-sync').value = Math.round((c.time_offset ?? -0.05) * 1000);
  $('cap-base').value = c.base_color || '#FFFFFF';
  $('cap-hl').value = c.highlight_color || '#22C55E';
  $('cap-outline').value = c.outline_color || '#000000';
  $('cap-upper').checked = c.uppercase !== false;
  $('cap-strip').checked = c.strip_punctuation !== false;
  syncCaptionLabels();

  try {
    const f = await api('/api/fonts');
    const opts = [];
    if (f.dropped.length) {
      opts.push(`<optgroup label="From assets/fonts">`
        + f.dropped.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join('')
        + `</optgroup>`);
    }
    opts.push(`<optgroup label="Installed on Windows">`
      + f.builtin.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join('')
      + `</optgroup>`);
    const current = c.font || 'Arial Rounded MT Bold';
    if (![...f.dropped, ...f.builtin].includes(current)) {
      opts.unshift(`<option value="${esc(current)}">${esc(current)}</option>`);
    }
    $('cap-font').innerHTML = opts.join('');
    $('cap-font').value = current;
    $('cap-fonthint').textContent = f.dropped.length
      ? `${f.dropped.length} custom font(s) loaded from assets/fonts/`
      : 'Want Poppins or Nunito? Drop the .ttf into assets/fonts/ — no install needed.';
  } catch { /* keep whatever is in the DOM */ }
}

function syncCaptionLabels() {
  $('val-capsize').textContent = `${(+$('cap-size').value * 100).toFixed(1)}%`;
  $('val-capmargin').textContent = `${Math.round(+$('cap-margin').value * 100)}%`;
  $('val-capscale').textContent = `${$('cap-scale').value}%`;
  $('val-capanim').textContent = `${$('cap-anim').value}ms`;
  const s = +$('cap-sync').value;
  $('val-capsync').textContent = `${s > 0 ? '+' : ''}${s}ms`;
}
['cap-size', 'cap-margin', 'cap-scale', 'cap-anim', 'cap-sync'].forEach((id) => {
  $(id).oninput = syncCaptionLabels;
});

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

/* ---------------------------------------------------------------- insights */
let lastInsights = null;

$('btn-insights').onclick = async () => {
  $('insights-modal').classList.remove('hidden');
  const box = $('insights-content');
  box.innerHTML = '<span class="hint">Loading…</span>';
  $('btn-apply-weights').classList.add('hidden');
  $('insights-count').textContent = '';
  try {
    const d = await api('/api/performance/insights');
    lastInsights = d;
    renderInsights(d);
  } catch (e) {
    box.innerHTML = `<span class="hint">${esc(e.message)}</span>`;
  }
};
$('btn-close-insights').onclick = () => $('insights-modal').classList.add('hidden');
$('insights-modal').onclick = (e) => { if (e.target.id === 'insights-modal') e.target.classList.add('hidden'); };

function renderInsights(d) {
  const box = $('insights-content');
  if (!d.ready) {
    box.innerHTML = `<p class="hint">${esc(d.reason || 'Not enough data logged yet.')}</p>`;
    $('insights-count').textContent = `${d.count || 0} clip(s) logged so far`;
    return;
  }
  $('insights-count').textContent = `${d.count} clip(s) logged · ${d.with_signal} used for the trend below`;

  const rows = Object.entries(d.categories).map(([key, v]) => {
    const pct = Math.min(50, Math.abs(v.correlation) * 50);
    const cls = v.correlation >= 0 ? 'pos' : 'neg';
    return `<div class="insight-row">
      <b>${esc(key)}</b>
      <span class="track"><span class="zero"></span>
        <span class="fill ${cls}" style="width:${pct}%"></span></span>
      <span class="note">${esc(v.note)} (${v.correlation.toFixed(2)})</span>
    </div>`;
  }).join('');

  const typeRows = (d.by_clip_type || []).map((t) => `
    <div class="insight-type-row"><b>${esc(t.clip_type.replace(/_/g, ' '))}</b>
      <span>${t.avg_z >= 0 ? '+' : ''}${t.avg_z} avg (n=${t.count})</span></div>`).join('');

  box.innerHTML = `
    <h4 style="margin:14px 0 4px;font-size:13px;">Score category vs. engagement</h4>
    ${rows}
    <h4 style="margin:16px 0 4px;font-size:13px;">By clip type</h4>
    ${typeRows || '<span class="hint">—</span>'}
  `;
  $('btn-apply-weights').classList.remove('hidden');
}

$('btn-apply-weights').onclick = async () => {
  if (!lastInsights || !lastInsights.suggested_weights) return;
  try {
    await api('/api/config', {
      method: 'POST',
      body: { config: { scoring: { weights: lastInsights.suggested_weights } }, save: true },
    });
    applyWeightsToUI(lastInsights.suggested_weights);
    toast('Scoring weights updated from performance data', 'ok', 5000);
  } catch (e) { toast(e.message, 'err'); }
};

$('btn-save-settings').onclick = async () => {
  const weights = {};
  $('weights').querySelectorAll('[data-w]').forEach((el) => { weights[el.dataset.w] = +el.value; });
  const patch = {
    whisper: { model: $('set-whisper').value, device: $('set-device').value,
               compute_type: $('set-compute').value },
    llm: { backend: $('set-backend').value, model: $('set-model').value.trim(),
           base_url: $('set-url').value.trim(), num_ctx: +$('set-ctx').value },
    clips: { block_seconds: +$('set-block').value, max_candidates_pass1: +$('set-cap').value,
             boundary_optimization: $('set-boundary').value === 'true',
             diversity_lambda: +$('set-div').value },
    scoring: { weights, heuristic_blend: +$('set-blend').value },
    captions: {
      enabled: $('opt-captions').checked,
      font: $('cap-font').value,
      font_size_ratio: +$('cap-size').value,
      max_words: +$('cap-words').value,
      position: $('cap-pos').value,
      margin_v_ratio: +$('cap-margin').value,
      highlight_scale: +$('cap-scale').value,
      animation_ms: +$('cap-anim').value,
      base_color: $('cap-base').value,
      highlight_color: $('cap-hl').value,
      outline_color: $('cap-outline').value,
      uppercase: $('cap-upper').checked,
      strip_punctuation: $('cap-strip').checked,
      time_offset: +$('cap-sync').value / 1000,
    },
    tightening: {
      remove_fillers: $('set-tighten-fillers').checked,
      min_gap: +$('set-tighten-gap').value,
      keep_pause: +$('set-tighten-keep').value,
    },
    thumbnail: {
      text_color: $('thumb-text-color').value,
      outline_color: $('thumb-outline-color').value,
      uppercase: $('thumb-upper').checked,
    },
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
  if (e.target.matches('input, select, textarea')) return;
  const playerOpen = !$('player-modal').classList.contains('hidden');
  if (e.key === 'Escape') {
    $('settings-modal').classList.add('hidden');
    if (playerOpen) closePlayer();
  }
  if (!playerOpen) return;
  if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); stepClip(1); }
  if (e.key === 'ArrowUp' || e.key === 'k') { e.preventDefault(); stepClip(-1); }
});
