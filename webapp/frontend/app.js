// ── Canvas & Drawing State ──
const c = document.getElementById('c');
const ctx = c.getContext('2d');
let drawing = false;
let strokes = [];
let redoStack = [];
let cur = [];
let tool = 'pen';

ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, c.width, c.height);

let history = [];

// ── Drawing ──
function drawStroke(pts, color) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.strokeStyle = color;
  ctx.lineWidth = color === '#fff' ? 12 : 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();
}

function redraw() {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  for (const s of strokes) drawStroke(s.pts, s.color);
}

c.onpointerdown = e => {
  drawing = true;
  cur = [{ x: e.offsetX, y: e.offsetY }];
  c.setPointerCapture(e.pointerId);
};
c.onpointermove = e => {
  if (!drawing) return;
  cur.push({ x: e.offsetX, y: e.offsetY });
  drawStroke(cur.slice(-2), tool === 'eraser' ? '#fff' : '#000');
};
c.onpointerup = () => {
  if (cur.length) {
    strokes.push({ pts: cur, color: tool === 'eraser' ? '#fff' : '#000' });
    redoStack = [];
  }
  drawing = false;
  cur = [];
};

function setTool(t) {
  tool = t;
  document.getElementById('btn-pen').classList.toggle('active', t === 'pen');
  document.getElementById('btn-eraser').classList.toggle('active', t === 'eraser');
  c.style.cursor = t === 'eraser' ? 'cell' : 'crosshair';
}
function undo() { if (strokes.length) { redoStack.push(strokes.pop()); redraw(); } }
function redo() { if (redoStack.length) { strokes.push(redoStack.pop()); redraw(); } }
function clearCanvas() { strokes = []; redoStack = []; redraw(); }

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.shiftKey && e.key === 'Z') { e.preventDefault(); redo(); }
  else if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); }
});

// ── Pipeline stage definitions ──
const STAGES = ['classify', 'domainnet', 'generate', 'score', 'background'];
const STAGE_LABELS = {
  classify: 'BEiT Classification',
  domainnet: 'DomainNet Matching',
  generate: 'ControlNet + Diffusion',
  score: 'Scoring & Selection',
  background: 'Background Removal',
};

// ── Helpers ──
function stageIndicator(stageStates) {
  let h = '<div class="progress-stages">';
  for (const s of STAGES) {
    const state = stageStates[s] || 'pending';
    h += `<div class="progress-stage ${state}"><span class="dot"></span><span>${STAGE_LABELS[s]}</span></div>`;
  }
  h += '</div>';
  return h;
}

function imgTag(b64, alt) {
  return `<img src="data:image/png;base64,${b64}" alt="${alt || ''}">`;
}

function skeletonCards(n) {
  let h = '';
  for (let i = 0; i < n; i++) h += '<div class="skeleton skeleton-card"></div>';
  return h;
}

// ── Progressive render state ──
let pipeState = {};

function resetPipeState() {
  pipeState = {
    stageStates: {},
    config: {},
    sketch: null,
    label: null,
    predictions: null,
    dnImages: [],
    linearts: [],
    candidates: [],
    pickIdx: null,
    siglipScores: null,
    selector: null,
    reason: null,
    final: null,
    removeBg: true,
  };
}

function renderPipeState() {
  const s = pipeState;
  const out = document.getElementById('out');
  const pred = document.getElementById('pred');
  let h = '';

  // Progress indicator at top
  h += '<div class="section">' + stageIndicator(s.stageStates) + '</div>';

  // Section 1: Input + Classification
  h += '<div class="section"><div class="section-title">Input</div><div class="grid">';
  if (s.sketch) {
    h += `<div class="stage">${imgTag(s.sketch, 'sketch')}<div class="cap">Your Sketch</div></div>`;
  } else {
    h += '<div class="skeleton skeleton-card"></div>';
  }
  h += '</div></div>';

  // Update pred panel
  if (s.label) {
    pred.style.display = 'block';
    let predHtml = `<div class="lbl">${s.label}</div>`;
    if (s.predictions) {
      predHtml += `<div class="rest">${s.predictions.slice(1).map(p => p.label + ' ' + (p.score * 100).toFixed(0) + '%').join(' · ')}</div>`;
    }
    predHtml += `<div class="rest">seed ${s.config.seed || '?'} / ${s.config.model || '?'}</div>`;
    pred.innerHTML = predHtml;
  }

  // Section 2: DomainNet
  const dnState = s.stageStates.domainnet;
  if (dnState) {
    h += '<div class="section"><div class="section-title">DomainNet Matches</div><div class="grid">';
    if (dnState === 'done') {
      s.dnImages.forEach((b, i) => {
        h += `<div class="stage">${imgTag(b, 'dn' + i)}<div class="cap">#${i + 1}</div></div>`;
      });
    } else {
      h += skeletonCards(s.config.best_of || 4);
    }
    h += '</div></div>';
  }

  // Section 3: Lineart + Candidates (progressive)
  const genState = s.stageStates.generate;
  if (genState) {
    const total = s.config.best_of || 4;

    // Lineart controls
    h += '<div class="section"><div class="section-title">Lineart Controls</div><div class="grid">';
    for (let i = 0; i < total; i++) {
      if (s.linearts[i]) {
        h += `<div class="stage">${imgTag(s.linearts[i], 'ctrl' + i)}<div class="cap">#${i + 1}</div></div>`;
      } else {
        h += '<div class="skeleton skeleton-card"></div>';
      }
    }
    h += '</div></div>';

    // Candidates
    const selectorLabel = s.selector === 'kimi' ? 'Kimi' : s.selector === 'siglip_multi' ? 'SigLIP2 Multi' : 'SigLIP2';
    const pickStr = s.pickIdx !== null ? ` — ${selectorLabel} picked #${s.pickIdx + 1}` : '';
    h += `<div class="section"><div class="section-title">Candidates${pickStr}</div><div class="grid">`;
    for (let i = 0; i < total; i++) {
      if (s.candidates[i]) {
        const isWinner = s.pickIdx !== null && i === s.pickIdx;
        const scoreBadge = s.siglipScores ? `<span class="score-badge">${s.siglipScores[i].toFixed(2)}</span>` : '';
        h += `<div class="stage${isWinner ? ' winner' : ''}">${imgTag(s.candidates[i], 'cand' + i)}<div class="cap">#${i + 1}${isWinner ? ' ★' : ''}${scoreBadge}</div></div>`;
      } else {
        h += '<div class="skeleton skeleton-card"></div>';
      }
    }
    h += '</div>';
    if (s.reason) h += `<div class="kimi-reason">${selectorLabel}: ${s.reason}</div>`;
    h += '</div>';
  }

  // Section 4: Final
  const bgState = s.stageStates.background;
  if (bgState) {
    h += '<div class="section"><div class="section-title">Final Result</div>';
    if (bgState === 'done' && s.final) {
      h += '<div class="final-section">';
      h += `<div class="stage final-stage"><img id="final-img" src="data:image/png;base64,${s.final}"><div class="cap">${s.removeBg ? 'BG removed' : 'No BG removal'}</div></div>`;
      h += '</div>';
      h += `<button class="btn-download" onclick="downloadFinal()">Download PNG</button>`;
    } else {
      h += '<div class="grid">' + skeletonCards(1) + '</div>';
    }
    h += '</div>';
  }

  // Gallery
  h += '<div id="gallery-container"></div>';

  out.innerHTML = h;
  renderGallery();
}

// ── Download ──
function downloadFinal() {
  const img = document.getElementById('final-img');
  if (!img) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = `sketch_bomb_${Date.now()}.png`;
  a.click();
}

// ── Gallery ──
function addToHistory(s) {
  if (!s.final || !s.label) return;
  history.unshift({
    label: s.label,
    model: s.config.model,
    seed: s.config.seed,
    finalB64: s.final,
    timestamp: new Date().toLocaleTimeString(),
    fullState: JSON.parse(JSON.stringify(s)),
  });
  if (history.length > 20) history.pop();
}

function renderGallery() {
  const container = document.getElementById('gallery-container');
  if (!container || history.length === 0) return;
  let h = `<div class="gallery-toggle" onclick="toggleGallery(this)">`;
  h += `<span class="arrow">&#9654;</span> History (${history.length})`;
  h += '</div><div class="gallery" id="gallery-list">';
  for (let i = 0; i < history.length; i++) {
    const item = history[i];
    h += `<div class="gallery-item" onclick="showHistoryItem(${i})">`;
    h += `<img src="data:image/png;base64,${item.finalB64}">`;
    h += `<div class="meta"><span class="label">${item.label}</span>`;
    h += `<span>${item.model} / seed ${item.seed} / ${item.timestamp}</span></div></div>`;
  }
  h += '</div>';
  container.innerHTML = h;
}

function toggleGallery(el) {
  el.classList.toggle('open');
  const list = document.getElementById('gallery-list');
  if (list) list.classList.toggle('open');
}

function showHistoryItem(idx) {
  const item = history[idx];
  pipeState = JSON.parse(JSON.stringify(item.fullState));
  // Mark all stages as done
  for (const s of STAGES) pipeState.stageStates[s] = 'done';
  renderPipeState();
}

// ── SSE Generation ──
async function generate() {
  if (!strokes.length) return;
  const btn = document.getElementById('btn');
  const pred = document.getElementById('pred');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  pred.style.display = 'none';

  const model = document.getElementById('sel-model').value;
  const bestOf = parseInt(document.getElementById('sel-bestof').value);
  const selector = document.getElementById('sel-selector').value;
  const removeBg = document.getElementById('chk-rembg').checked;
  const seedInput = document.getElementById('inp-seed').value.trim();
  const seed = seedInput === '' ? -1 : parseInt(seedInput);

  resetPipeState();
  renderPipeState();

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: c.toDataURL('image/png'),
        model, best_of: bestOf, selector, remove_bg: removeBg, seed,
      }),
    });

    if (!resp.ok) {
      let msg = `Server error ${resp.status}`;
      try { msg = (await resp.json()).detail || msg; } catch {}
      throw new Error(msg);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const evt = JSON.parse(payload);
          handleSSE(evt);
        } catch {}
      }
    }

    // Finished — save to history
    addToHistory(pipeState);
    renderPipeState();

  } catch (e) {
    const out = document.getElementById('out');
    out.innerHTML = `<div class="error-msg">Error: ${e.message}</div><div id="gallery-container"></div>`;
    renderGallery();
  }

  btn.disabled = false;
  btn.textContent = 'Generate';
}

function handleSSE(evt) {
  const s = pipeState;

  switch (evt.type) {
    case 'config':
      s.config = evt;
      break;

    case 'stage_start':
      s.stageStates[evt.stage] = 'active';
      renderPipeState();
      break;

    case 'stage_done':
      s.stageStates[evt.stage] = 'done';
      const d = evt.data;

      if (evt.stage === 'classify') {
        s.sketch = d.sketch;
        s.label = d.label;
        s.predictions = d.predictions;
      } else if (evt.stage === 'domainnet') {
        s.dnImages = d.images || [];
      } else if (evt.stage === 'score') {
        s.pickIdx = d.pick_idx;
        s.siglipScores = d.siglip_scores;
        s.selector = d.selector;
        s.reason = d.reason;
      } else if (evt.stage === 'background') {
        s.final = d.final;
        s.removeBg = d.remove_bg;
      }

      renderPipeState();
      break;

    case 'candidate':
      s.linearts[evt.index] = evt.lineart;
      s.candidates[evt.index] = evt.image;
      renderPipeState();
      break;

    case 'error':
      throw new Error(evt.message || 'Pipeline error');
  }
}
