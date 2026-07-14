/* ============================================================
   OptiCrop – Smart Agricultural Production Optimization Engine
   predict.js – Crop Prediction Page Logic
   Author   : Principal Frontend Engineer
   Version  : 1.2.0 (ML Integration Release – live fetch() active)

   NOTE: Global concerns (preloader, navbar scroll, smooth scroll,
   scroll animations, back-to-top) are now managed by script.js.

   Sections:
   1.  Field Configuration (ranges, validation rules)
   2.  DOM Ready Bootstrapper
   3.  Form Progress Tracker
   4.  Input Validation Engine
   5.  Range Bar Updater
   6.  Button Ripple
   7.  Loading Overlay
   8.  Prediction Submit Handler (live fetch() → POST /predict)
   9.  Result Renderer (consumes Flask JSON response)
   10. Reset / Clear
   11. FAQ Keyboard Accessibility
   ============================================================ */

'use strict';

/* ──────────────────────────────────────────────────────────
   1. FIELD CONFIGURATION (matches preprocessing.py constants)
   NUMERICAL_FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
   ────────────────────────────────────────────────────────── */
const FIELD_CONFIG = {
  nitrogen:     { min: 0,   max: 140,  step: 1,    label: 'Nitrogen (N)',     unit: 'ppm' },
  phosphorus:   { min: 5,   max: 145,  step: 1,    label: 'Phosphorus (P)',   unit: 'ppm' },
  potassium:    { min: 5,   max: 205,  step: 1,    label: 'Potassium (K)',    unit: 'ppm' },
  temperature:  { min: 8,   max: 44,   step: 0.1,  label: 'Temperature',      unit: '°C'  },
  humidity:     { min: 14,  max: 100,  step: 0.1,  label: 'Humidity',         unit: '%'   },
  ph:           { min: 3.5, max: 10,   step: 0.01, label: 'pH Level',         unit: 'pH'  },
  rainfall:     { min: 20,  max: 300,  step: 0.1,  label: 'Rainfall',         unit: 'mm'  },
};

/** Loading stage messages displayed sequentially */
const LOADING_STAGES = [
  { msg: 'Analyzing soil nutrient profile (N, P, K)…',   pct: 20 },
  { msg: 'Processing environmental parameters…',          pct: 40 },
  { msg: 'Running Machine Learning prediction pipeline…', pct: 62 },
  { msg: 'Evaluating Random Forest decision trees…',      pct: 80 },
  { msg: 'Preparing intelligent crop recommendation…',    pct: 95 },
  { msg: 'Finalizing results…',                           pct: 100 },
];


/* ──────────────────────────────────────────────────────────
   2. DOM READY BOOTSTRAPPER
   ────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Global: preloader, navbar, scroll animations, back-to-top,
  // smooth scroll — all managed by script.js.
  initFormProgress();
  initRangeBarUpdaters();
  initInputValidation();
  initRipple();
  initFormHandlers();
  initFaqKeyboard();
});

/* ──────────────────────────────────────────────────────────
   3. FORM PROGRESS TRACKER
      Calculates % of valid fields filled in
   ────────────────────────────────────────────────────── */
function initFormProgress() {
  const fill   = document.getElementById('formProgressFill');
  const label  = document.getElementById('formProgressLabel');
  const inputs = document.querySelectorAll('.predict-input');
  if (!fill || !inputs.length) return;

  function updateProgress() {
    const total = inputs.length;
    const filled = [...inputs].filter(inp => inp.value.trim() !== '' && inp.classList.contains('is-valid')).length;
    const pct = Math.round((filled / total) * 100);
    fill.style.width = pct + '%';
    if (label) label.querySelector('.pct-val').textContent = `${filled}/${total} fields complete`;
  }

  inputs.forEach(inp => inp.addEventListener('input', updateProgress));
  updateProgress();
}

/* ──────────────────────────────────────────────────────────
   7. INPUT VALIDATION ENGINE
   ────────────────────────────────────────────────────── */
const validityMap = {}; // fieldId → boolean

function initInputValidation() {
  const submitBtn = document.getElementById('predictBtn');

  Object.keys(FIELD_CONFIG).forEach(fieldId => {
    validityMap[fieldId] = false;
    const input = document.getElementById(fieldId);
    if (!input) return;

    const cfg = FIELD_CONFIG[fieldId];

    input.addEventListener('input',  () => validateField(fieldId, cfg));
    input.addEventListener('blur',   () => validateField(fieldId, cfg));
    input.addEventListener('change', () => validateField(fieldId, cfg));
  });

  // Re-evaluate submit button state whenever any field changes
  document.getElementById('cropForm')?.addEventListener('input', () => {
    if (submitBtn) submitBtn.disabled = !allFieldsValid();
  });
}

function validateField(fieldId, cfg) {
  const input    = document.getElementById(fieldId);
  const errEl    = document.getElementById(`err-${fieldId}`);
  const okEl     = document.getElementById(`ok-${fieldId}`);
  const val      = input.value.trim();
  let   errorMsg = '';

  if (val === '') {
    errorMsg = `${cfg.label} is required.`;
  } else if (isNaN(val) || val === '') {
    errorMsg = `Please enter a valid numeric value.`;
  } else {
    const num = parseFloat(val);
    if (num < cfg.min) errorMsg = `Minimum value is ${cfg.min} ${cfg.unit}.`;
    else if (num > cfg.max) errorMsg = `Maximum value is ${cfg.max} ${cfg.unit}.`;
  }

  if (errorMsg) {
    input.classList.add('is-invalid');
    input.classList.remove('is-valid');
    if (errEl) { errEl.querySelector('.err-text').textContent = errorMsg; errEl.classList.add('show'); }
    if (okEl)  okEl.classList.remove('show');
    validityMap[fieldId] = false;
  } else {
    input.classList.remove('is-invalid');
    input.classList.add('is-valid');
    if (errEl) errEl.classList.remove('show');
    if (okEl)  okEl.classList.add('show');
    validityMap[fieldId] = true;
  }

  updateRangeBar(fieldId);
  updateSubmitBtn();
}

function allFieldsValid() {
  return Object.values(validityMap).every(Boolean);
}

function updateSubmitBtn() {
  const btn = document.getElementById('predictBtn');
  if (btn) btn.disabled = !allFieldsValid();
}

/* ──────────────────────────────────────────────────────────
   8. RANGE BAR UPDATER
   ────────────────────────────────────────────────────── */
function initRangeBarUpdaters() {
  Object.keys(FIELD_CONFIG).forEach(id => {
    const input = document.getElementById(id);
    if (!input) return;
    input.addEventListener('input', () => updateRangeBar(id));
  });
}

function updateRangeBar(fieldId) {
  const cfg  = FIELD_CONFIG[fieldId];
  const fill = document.getElementById(`bar-${fieldId}`);
  const inp  = document.getElementById(fieldId);
  if (!fill || !inp) return;
  const val = parseFloat(inp.value);
  if (isNaN(val)) { fill.style.width = '0%'; return; }
  const clampedPct = Math.max(0, Math.min(100, ((val - cfg.min) / (cfg.max - cfg.min)) * 100));
  fill.style.width = clampedPct + '%';
}

/* ──────────────────────────────────────────────────────────
   9. BUTTON RIPPLE
   ────────────────────────────────────────────────────── */
function initRipple() {
  document.querySelectorAll('.btn-predict, .btn-pcta').forEach(btn => {
    btn.addEventListener('click', function(e) {
      if (this.disabled) return;
      const r    = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      r.className = 'ripple';
      r.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX-rect.left-size/2}px;top:${e.clientY-rect.top-size/2}px`;
      this.appendChild(r);
      r.addEventListener('animationend', () => r.remove());
    });
  });
}

/* ──────────────────────────────────────────────────────────
   10. LOADING OVERLAY
   ────────────────────────────────────────────────────── */
function showLoading() {
  const overlay  = document.getElementById('loading-overlay');
  const fill     = document.getElementById('loadingProgressFill');
  const msgEl    = document.getElementById('loadingMsg');
  const steps    = document.querySelectorAll('.lstep');
  if (!overlay) return;

  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';

  fill.style.width = '0%';
  steps.forEach(s => s.classList.remove('active','done'));

  let stageIndex = 0;

  function runStage() {
    if (stageIndex >= LOADING_STAGES.length) return;
    const stage = LOADING_STAGES[stageIndex];

    // Update message
    msgEl.style.opacity = '0';
    setTimeout(() => {
      msgEl.textContent  = stage.msg;
      msgEl.style.opacity = '1';
    }, 180);

    // Update progress
    fill.style.width = stage.pct + '%';

    // Update step dots
    steps.forEach((s, i) => {
      if (i < stageIndex)   s.classList.add('done');
      if (i === stageIndex) s.classList.add('active');
    });

    stageIndex++;
  }

  const interval = setInterval(() => {
    runStage();
    if (stageIndex >= LOADING_STAGES.length) clearInterval(interval);
  }, 620);

  // Return a resolver so submit handler can dismiss
  return {
    interval,
    dismiss() {
      clearInterval(interval);
      fill.style.width = '100%';
    }
  };
}

function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (!overlay) return;
  overlay.classList.remove('active');
  document.body.style.overflow = '';
}

/* ──────────────────────────────────────────────────────────
   11. PREDICTION SUBMIT HANDLER — FLASK INTEGRATED
       POSTs form data to /predict, receives JSON result_data,
       and renders result inline via renderResult().
   ────────────────────────────────────────────────────── */
function initFormHandlers() {
  const form      = document.getElementById('cropForm');
  const resetBtn  = document.getElementById('resetBtn');
  const clearBtn  = document.getElementById('clearBtn');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!allFieldsValid()) return;

    const formData = new FormData(form);
    const loader   = showLoading();

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body:   formData,
      });

      const data = await response.json();

      if (loader) loader.dismiss();
      hideLoading();

      if (data.success) {
        renderResult(data);
      } else if (response.status === 422 && data.errors) {
        // Server-side validation errors — mark fields with backend messages
        Object.entries(data.errors).forEach(([field, msg]) => {
          const inp    = document.getElementById(field);
          const errEl  = document.getElementById(`err-${field}`);
          const errTxt = errEl?.querySelector('.err-text');
          if (inp)    inp.classList.add('is-invalid');
          if (errTxt) errTxt.textContent = msg;
          if (errEl)  errEl.classList.add('show');
        });
        showError('Please correct the highlighted fields and try again.');
      } else {
        showError(data.message || 'Prediction failed. Please try again.');
      }
    } catch (err) {
      if (loader) loader.dismiss();
      hideLoading();
      console.error('Fetch error:', err);
      showError('Network error. Please check your connection and try again.');
    }
  });

  /* Reset: re-validate all fields */
  resetBtn?.addEventListener('click', () => {
    form.reset();
    Object.keys(FIELD_CONFIG).forEach(id => {
      const inp = document.getElementById(id);
      if (!inp) return;
      inp.classList.remove('is-valid','is-invalid');
      validityMap[id] = false;
      const fill = document.getElementById(`bar-${id}`);
      if (fill) fill.style.width = '0%';
      document.getElementById(`err-${id}`)?.classList.remove('show');
      document.getElementById(`ok-${id}`)?.classList.remove('show');
    });
    updateSubmitBtn();
    updateFormProgress();
    hideResult();
  });

  /* Clear All */
  clearBtn?.addEventListener('click', () => resetBtn?.click());
}

function showError(msg) {
  const container = document.getElementById('predict-error-msg');
  const textEl    = document.getElementById('predict-error-text');
  if (container && textEl) {
    textEl.textContent   = msg;
    container.style.display = 'flex';
    // Scroll error into view for accessibility
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    // Auto-dismiss after 6 seconds
    setTimeout(() => { container.style.display = 'none'; }, 6000);
  } else {
    // Last-resort fallback — never loses the error silently
    console.warn('OptiCrop prediction error:', msg);
  }
}

function updateFormProgress() {
  const fill  = document.getElementById('formProgressFill');
  if (fill) fill.style.width = '0%';
  const label = document.getElementById('formProgressLabel');
  if (label) label.querySelector('.pct-val').textContent = `0/${Object.keys(FIELD_CONFIG).length} fields complete`;
}

/* ──────────────────────────────────────────────────────────
   12. RESULT RENDERER
       Consumes the full result_data JSON from POST /predict.
       Backend contract keys used:
         crop, confidence, season, soil_type, water_req,
         pred_time_ms, why, fertilizer, yield_exp,
         irrigation, advice, emoji, scientific, category
   ────────────────────────────────────────────────────── */
function renderResult(data) {
  const section = document.getElementById('result-section');
  if (!section) return;

  /* ── Core crop info ────────────────────────────────────────── */
  setText('resultCropName',    data.crop    ?? '—');
  setText('resultCropNameRec', data.crop    ?? '—');
  setText('resultEmoji',       data.emoji   ?? '🌱');
  setText('resultScientific',  data.scientific ?? '');
  setText('resultCategory',    data.category   ?? '');

  /* ── Agronomic details ─────────────────────────────────────── */
  setText('resultSeason',    data.season    ?? '—');
  setText('resultSoilType',  data.soil_type ?? '—');
  setText('resultWaterReq',  data.water_req ?? '—');
  setText('resultFertilizer',data.fertilizer ?? '—');
  setText('resultYieldExp',  data.yield_exp ?? '—');
  setText('resultIrrigation',data.irrigation ?? '—');
  setText('resultAdvice',    data.advice    ?? '');
  setText('resultWhy',       data.why       ?? '');

  /* ── Prediction meta ───────────────────────────────────────── */
  setText('resultPredTime',  `${data.pred_time_ms ?? '—'} ms`);
  setText('resultModel',     data.model     ?? 'Random Forest');
  setText('resultTimestamp', data.timestamp ?? '');

  /* ── Confidence bar ────────────────────────────────────────── */
  const confNum  = document.getElementById('resultConfNum');
  const confFill = document.getElementById('resultConfFill');
  const confLbl  = document.getElementById('resultConfLabel');
  const pct      = data.confidence ?? 0;
  if (confNum)  confNum.textContent  = pct.toFixed(1) + '%';
  if (confLbl)  confLbl.textContent  = pct > 90 ? 'Very High Confidence' : pct > 75 ? 'High Confidence' : 'Moderate Confidence';
  if (confFill) { confFill.style.width = '0%'; setTimeout(() => { confFill.style.width = pct + '%'; }, 300); }

  /* ── Feature importance bars ───────────────────────────────── */
  const fiContainer = document.getElementById('featureImportanceChart');
  if (fiContainer && Array.isArray(data.feature_importance)) {
    fiContainer.innerHTML = data.feature_importance.map(f => `
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:3px;">
          <span style="color:var(--text-secondary,#94a3b8);">${f.label}</span>
          <span style="color:${f.color};font-weight:700;">${f.pct}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:6px;overflow:hidden;">
          <div style="height:100%;width:0%;background:${f.color};border-radius:99px;transition:width 0.8s ease;" data-w="${f.pct}%"></div>
        </div>
      </div>`).join('');
    setTimeout(() => {
      fiContainer.querySelectorAll('[data-w]').forEach(b => { b.style.width = b.dataset.w; });
    }, 400);
  }

  /* ── Probability bars ──────────────────────────────────────── */
  const probContainer = document.getElementById('probabilityChart');
  if (probContainer && Array.isArray(data.probabilities)) {
    probContainer.innerHTML = data.probabilities.map((p,i) => `
      <div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;">
          <span style="color:var(--text-secondary,#94a3b8);font-weight:${i===0?700:400};">${p.crop}</span>
          <span style="background:${p.color};-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;">${p.pct.toFixed(1)}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:8px;overflow:hidden;">
          <div style="height:100%;width:0%;background:${p.color};border-radius:99px;transition:width 0.9s ease ${i*0.1}s;" data-w="${p.pct}%"></div>
        </div>
      </div>`).join('');
    setTimeout(() => {
      probContainer.querySelectorAll('[data-w]').forEach(b => { b.style.width = b.dataset.w; });
    }, 500);
  }

  /* ── Show section and scroll ───────────────────────────────── */
  section.classList.add('visible');
  setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function hideResult() {
  document.getElementById('result-section')?.classList.remove('visible');
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

/* ──────────────────────────────────────────────────────────
   11. FAQ KEYBOARD ACCESSIBILITY
   ────────────────────────────────────────────────────── */
function initFaqKeyboard() {
  document.querySelectorAll('.pfaq-button').forEach(btn => {
    btn.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); btn.click(); }
    });
  });
}

/* ══════════════════════════════════════════════════════════
   END predict.js  v1.1.0
══════════════════════════════════════════════════════════ */
