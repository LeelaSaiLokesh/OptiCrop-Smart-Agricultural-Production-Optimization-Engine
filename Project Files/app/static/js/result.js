/* ============================================================
   OptiCrop – Smart Agricultural Production Optimization Engine
   result.js – Prediction Result Page Logic
   Author   : Principal Frontend Engineer
   Version  : 1.1.0 (Integration Release – global inits removed)

   NOTE: Global concerns (preloader, navbar scroll, smooth scroll,
   back-to-top) are now managed by script.js.
   initScrollAnimations is page-specific (handles .rv-fade-up bar
   triggers) and is intentionally kept in this file.

   Sections:
   1.  Result Data (placeholder – Flask will inject via Jinja2)
   2.  DOM Ready Bootstrapper
   3.  Inject Result Data Into DOM
   4.  Scroll Animation Observer (page-specific)
   5.  Confidence Bar & Gauge Animator
   6.  Soil Parameter Bar Animator
   7.  Feature Importance Bar Animator
   8.  Probability Chart Renderer
   9.  Bar Chart Renderer (environmental conditions)
   10. Radar Chart (Canvas-based)
   11. Button Ripple
   12. Print Report
   13. FAQ Keyboard
   ============================================================ */

'use strict';

/* ──────────────────────────────────────────────────────────
   1. RESULT DATA
      The Jinja2 template injects RESULT as a global const when
      result_data is provided by Flask (see {% block extra_js %}).
      The object below is a dev-time placeholder used when no
      Flask context is present (e.g. opening the HTML directly).
   ────────────────────────────────────────────────────── */
// If Flask injected `const RESULT = {{ result_data | tojson }};` in the
// template's extra_js block (before this file loads), we use that value.
// Otherwise we fall back to this dev-time placeholder.
// Note: var is intentional here — it won't conflict with a prior const RESULT.
if (typeof RESULT === 'undefined') { var RESULT; }
RESULT = RESULT || {
  crop:          'Rice',
  scientific:    'Oryza sativa',
  category:      'Cereal / Staple Grain',
  emoji:         '🌾',
  confidence:    92.6,
  model:         'Random Forest (300 estimators)',
  pred_time_ms:  38,
  timestamp:     new Date().toISOString(),

  /* Input parameters (from predict form) */
  inputs: {
    nitrogen:    90,
    phosphorus:  42,
    potassium:   43,
    temperature: 20.8,
    humidity:    82.0,
    ph:          6.50,
    rainfall:    202.9,
  },

  /* Agronomic recommendation */
  why:          'The Random Forest model identified Rice as the optimal crop based on high nitrogen (90 ppm) and humidity (82%) values, combined with moderate pH (6.5) and substantial rainfall (202.9 mm). These conditions closely match the ideal pedoclimatic signature in the Rice cluster within the training dataset.',
  season:       'Kharif (June – September)',
  soil_type:    'Alluvial / Clayey Loam',
  climate:      'Tropical / Humid (24–35°C)',
  water_req:    'High — 200–300 mm per season',
  fertilizer:   'NPK 80:40:40 kg/ha (Urea + DAP + MOP)',
  yield_exp:    '4–6 tonnes per hectare',
  irrigation:   'Continuous flooding / Alternate Wetting and Drying (AWD)',
  advice:       'Transplant seedlings at 21–25 days age. Apply basal fertilizer before puddling. Maintain 5 cm water depth during vegetative stage. Drain field 10 days before harvest.',

  /* Feature importances (placeholder – from model.feature_importances_) */
  feature_importance: [
    { label: 'Rainfall',     pct: 28, color: '#38bdf8' },
    { label: 'Humidity',     pct: 24, color: '#2dd4bf' },
    { label: 'Temperature',  pct: 18, color: '#fb7185' },
    { label: 'pH',           pct: 12, color: '#a78bfa' },
    { label: 'Nitrogen',     pct: 10, color: '#22c55e' },
    { label: 'Phosphorus',   pct:  5, color: '#38bdf8' },
    { label: 'Potassium',    pct:  3, color: '#fbbf24' },
  ],

  /* Top-5 prediction probabilities */
  probabilities: [
    { crop: 'Rice',       pct: 92.6, color: 'var(--grad-primary)' },
    { crop: 'Jute',       pct:  4.2, color: 'linear-gradient(135deg,#38bdf8,#0ea5e9)' },
    { crop: 'Maize',      pct:  1.8, color: 'linear-gradient(135deg,#fbbf24,#f59e0b)' },
    { crop: 'Coconut',    pct:  0.9, color: 'linear-gradient(135deg,#a78bfa,#7c3aed)' },
    { crop: 'Blackgram',  pct:  0.5, color: 'linear-gradient(135deg,#f87171,#ef4444)' },
  ],
};

/* Soil param display config (matches FIELD_CONFIG in predict.js) */
const SOIL_PARAMS = [
  { id:'nitrogen',    label:'Nitrogen (N)',    unit:'ppm', min:0,   max:140,  emoji:'🟢', color:'#22c55e', gradient:'linear-gradient(90deg,#16a34a,#22c55e)' },
  { id:'phosphorus',  label:'Phosphorus (P)',  unit:'ppm', min:5,   max:145,  emoji:'🔵', color:'#38bdf8', gradient:'linear-gradient(90deg,#0ea5e9,#38bdf8)' },
  { id:'potassium',   label:'Potassium (K)',   unit:'ppm', min:5,   max:205,  emoji:'🟡', color:'#fbbf24', gradient:'linear-gradient(90deg,#f59e0b,#fbbf24)' },
  { id:'temperature', label:'Temperature',     unit:'°C',  min:8,   max:44,   emoji:'🌡️', color:'#fb7185', gradient:'linear-gradient(90deg,#e11d48,#fb7185)' },
  { id:'humidity',    label:'Humidity',        unit:'%',   min:14,  max:100,  emoji:'💧', color:'#2dd4bf', gradient:'linear-gradient(90deg,#0d9488,#2dd4bf)' },
  { id:'ph',          label:'pH Level',        unit:'pH',  min:3.5, max:10,   emoji:'🧫', color:'#a78bfa', gradient:'linear-gradient(90deg,#7c3aed,#a78bfa)' },
  { id:'rainfall',    label:'Rainfall',        unit:'mm',  min:20,  max:300,  emoji:'🌧️', color:'#38bdf8', gradient:'linear-gradient(90deg,#0369a1,#38bdf8)' },
];

/* ──────────────────────────────────────────────────────────
   2. DOM READY BOOTSTRAPPER
   ────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  injectResultData();
  // Global: preloader, navbar, back-to-top, smooth scroll
  // — all managed by script.js.
  initScrollAnimations(); // page-specific: also triggers bar fills
  initRipple();
  initPrintBtn();
  initFaqKeyboard();

  // Animated elements triggered after page settles
  setTimeout(() => {
    animateConfidenceBar();
    animateConfidenceGauge();
  }, 400);
});

/* ──────────────────────────────────────────────────────────
   3. INJECT RESULT DATA INTO DOM
   ────────────────────────────────────────────────────── */
function injectResultData() {
  // Crop identity
  setText('cropName',       RESULT.crop);
  setText('cropNameBadge',  RESULT.crop);
  setText('cropScientific', RESULT.scientific);
  setText('cropCategory',   RESULT.category);
  setText('cropEmoji',      RESULT.emoji);

  // Confidence
  setText('confPct',        RESULT.confidence.toFixed(1) + '%');
  const quality = RESULT.confidence >= 90 ? 'Very High Confidence'
    : RESULT.confidence >= 75 ? 'High Confidence' : 'Moderate Confidence';
  setText('confQuality', quality);

  // Summary cards
  setText('summaryModel',    RESULT.model);
  setText('summaryTime',     RESULT.pred_time_ms + ' ms');
  setText('summaryStatus',   '✅ Successful');
  setText('summaryConf',     RESULT.confidence.toFixed(1) + '%');
  setText('summaryTimestamp', formatTimestamp(RESULT.timestamp));

  // Soil inputs
  SOIL_PARAMS.forEach(p => {
    const val = RESULT.inputs[p.id];
    if (val === undefined) return;
    setText(`soilVal-${p.id}`, val);
    const pct = Math.max(0, Math.min(100, ((val - p.min) / (p.max - p.min)) * 100));
    const fill = document.getElementById(`soilBar-${p.id}`);
    if (fill) {
      fill.style.background = p.gradient;
      setTimeout(() => { fill.style.width = pct + '%'; }, 600);
    }
    const status = document.getElementById(`soilStatus-${p.id}`);
    if (status) {
      const s = pct > 65 ? ['status-optimal','Optimal'] : pct > 30 ? ['status-moderate','Moderate'] : ['status-low','Low'];
      status.className = `soil-param-status ${s[0]}`;
      status.innerHTML = `<i class="bi bi-circle-fill"></i> ${s[1]}`;
    }
  });

  // Why this crop
  setText('whyCropText', RESULT.why);

  // Agri sidebar
  setText('recSeason',    RESULT.season);
  setText('recSoil',      RESULT.soil_type);
  setText('recClimate',   RESULT.climate);
  setText('recWater',     RESULT.water_req);
  setText('recFertilizer',RESULT.fertilizer);
  setText('recYield',     RESULT.yield_exp);
  setText('recIrrigation',RESULT.irrigation);
  setText('recAdvice',    RESULT.advice);

  // Confidence gauge value
  setText('gaugeVal', RESULT.confidence.toFixed(0) + '%');

  // Feature importance bars
  const fiContainer = document.getElementById('featureImportanceBars');
  if (fiContainer) {
    fiContainer.innerHTML = RESULT.feature_importance.map(f =>
      `<div class="feature-bar-row rv-fade-up">
        <div class="feature-bar-label">${f.label}</div>
        <div class="feature-bar-track">
          <div class="feature-bar-fill" style="width:0%;background:${f.color};" data-target="${f.pct}"></div>
        </div>
        <div class="feature-bar-pct">${f.pct}%</div>
      </div>`
    ).join('');
    setTimeout(animateFeatureBars, 800);
  }

  // Probability bars
  const probContainer = document.getElementById('probabilityBars');
  if (probContainer) {
    probContainer.innerHTML = RESULT.probabilities.map(p =>
      `<div class="prob-bar-row rv-fade-up">
        <div class="prob-bar-crop">${p.crop}</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:0%;background:${p.color};" data-target="${p.pct}"></div>
        </div>
        <div class="prob-bar-pct">${p.pct}%</div>
      </div>`
    ).join('');
    setTimeout(animateProbBars, 900);
  }

  // Bar chart (environmental)
  renderBarChart();

  // Radar chart
  setTimeout(renderRadarChart, 500);

  // Timestamp on hero
  setText('heroTimestamp', formatTimestamp(RESULT.timestamp));
}

/* ──────────────────────────────────────────────────────────
   4. SCROLL ANIMATIONS (page-specific)
   ────────────────────────────────────────────────────── */
function initScrollAnimations() {
  const els = document.querySelectorAll('.rv-fade-up, .rv-fade-left, .rv-fade-right');
  if (!els.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        obs.unobserve(e.target);
        // If it's a feature/prob bar, trigger its animation
        const fill = e.target.querySelector('[data-target]');
        if (fill) animateSingleBar(fill);
      }
    });
  }, { rootMargin:'0px 0px -50px 0px', threshold:0.08 });
  els.forEach(el => obs.observe(el));
}

/* ──────────────────────────────────────────────────────────
   7. CONFIDENCE BAR & GAUGE
   ────────────────────────────────────────────────────── */
function animateConfidenceBar() {
  const fill = document.getElementById('confBarFill');
  if (fill) fill.style.width = RESULT.confidence + '%';
}

function animateConfidenceGauge() {
  const gaugeFill = document.getElementById('gaugeFill');
  if (!gaugeFill) return;
  // 0% → -180deg, 100% → 0deg
  const angle = -180 + (RESULT.confidence / 100) * 180;
  gaugeFill.style.transform = `rotate(${angle}deg)`;

  // colour: green > 80, amber 60-80, red < 60
  const col = RESULT.confidence >= 80 ? '#22c55e' : RESULT.confidence >= 60 ? '#f59e0b' : '#ef4444';
  gaugeFill.style.borderTopColor   = col;
  gaugeFill.style.borderRightColor = col;
  document.getElementById('gaugeVal')?.style && (document.getElementById('gaugeVal').style.color = col);
}

/* ──────────────────────────────────────────────────────────
   8. FEATURE IMPORTANCE BARS
   ────────────────────────────────────────────────────── */
function animateFeatureBars() {
  document.querySelectorAll('#featureImportanceBars .feature-bar-fill').forEach(el => animateSingleBar(el));
}

function animateProbBars() {
  document.querySelectorAll('#probabilityBars .prob-bar-fill').forEach(el => animateSingleBar(el));
}

function animateSingleBar(el) {
  const target = el.getAttribute('data-target');
  if (target !== null) el.style.width = target + '%';
}

/* ──────────────────────────────────────────────────────────
   9. BAR CHART (Environmental Conditions)
   ────────────────────────────────────────────────────── */
function renderBarChart() {
  const container = document.getElementById('envBarChart');
  if (!container) return;

  const data = [
    { label:'N',    value:RESULT.inputs.nitrogen,    max:140,  color:'#22c55e' },
    { label:'P',    value:RESULT.inputs.phosphorus,  max:145,  color:'#38bdf8' },
    { label:'K',    value:RESULT.inputs.potassium,   max:205,  color:'#fbbf24' },
    { label:'Temp', value:RESULT.inputs.temperature, max:44,   color:'#fb7185' },
    { label:'Hum',  value:RESULT.inputs.humidity,    max:100,  color:'#2dd4bf' },
    { label:'pH',   value:RESULT.inputs.ph,          max:10,   color:'#a78bfa' },
    { label:'Rain', value:RESULT.inputs.rainfall,    max:300,  color:'#38bdf8' },
  ];

  container.innerHTML = data.map(d => {
    const pct = Math.round((d.value / d.max) * 100);
    return `<div class="bar-col">
      <div class="bar-col-fill" style="background:${d.color};height:${pct}%;opacity:0.85;border-radius:4px 4px 0 0;" data-target="${pct}"></div>
      <div class="bar-col-label">${d.label}</div>
    </div>`;
  }).join('');
}

/* ──────────────────────────────────────────────────────────
   10. RADAR CHART (Canvas — pure JS, no library needed for
       this simple 7-axis polygon)
   ────────────────────────────────────────────────────── */
function renderRadarChart() {
  const canvas = document.getElementById('radarCanvas');
  if (!canvas || !canvas.getContext) return;

  canvas.width  = 200;
  canvas.height = 200;

  const ctx    = canvas.getContext('2d');
  const cx     = 100;
  const cy     = 100;
  const R      = 80;
  const labels = ['N','P','K','Temp','Hum','pH','Rain'];
  const values = [
    RESULT.inputs.nitrogen    / 140,
    RESULT.inputs.phosphorus  / 145,
    RESULT.inputs.potassium   / 205,
    RESULT.inputs.temperature / 44,
    RESULT.inputs.humidity    / 100,
    RESULT.inputs.ph          / 10,
    RESULT.inputs.rainfall    / 300,
  ];
  const n = labels.length;

  function getPoint(i, r) {
    const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  }

  // Draw grid rings
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75, 1].forEach(ratio => {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = getPoint(i, R * ratio);
      i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.stroke();
  });

  // Draw axes
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  for (let i = 0; i < n; i++) {
    const p = getPoint(i, R);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p.x, p.y); ctx.stroke();
  }

  // Draw data polygon
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  grad.addColorStop(0, 'rgba(22,163,74,0.4)');
  grad.addColorStop(1, 'rgba(14,165,233,0.15)');
  ctx.fillStyle = grad;
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const p = getPoint(i, R * Math.min(v, 1));
    i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Draw dots
  ctx.fillStyle = '#22c55e';
  values.forEach((v, i) => {
    const p = getPoint(i, R * Math.min(v, 1));
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // Labels
  ctx.fillStyle = 'rgba(148,163,184,0.9)';
  ctx.font = 'bold 10px "Outfit", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  labels.forEach((lbl, i) => {
    const p = getPoint(i, R + 14);
    ctx.fillText(lbl, p.x, p.y);
  });
}

/* ──────────────────────────────────────────────────────────
   11. BUTTON RIPPLE
   ────────────────────────────────────────────────────── */
function initRipple() {
  document.querySelectorAll('.btn-action-primary').forEach(btn => {
    btn.addEventListener('click', function(e) {
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
   12. PRINT REPORT
   ────────────────────────────────────────────────────── */
function initPrintBtn() {
  document.getElementById('printReportBtn')?.addEventListener('click', () => window.print());
}

/* ──────────────────────────────────────────────────────────
   15. FAQ KEYBOARD
   ────────────────────────────────────────────────────── */
function initFaqKeyboard() {
  document.querySelectorAll('.rfaq-button').forEach(btn => {
    btn.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); btn.click(); }
    });
  });
}

/* ──────────────────────────────────────────────────────────
   UTILITIES
   ────────────────────────────────────────────────────── */
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function formatTimestamp(iso) {
  try {
    return new Date(iso).toLocaleString('en-IN', {
      day:'numeric', month:'short', year:'numeric',
      hour:'2-digit', minute:'2-digit', hour12:true
    });
  } catch { return iso; }
}

/* ══════════════════════════════════════════════════════════
   END result.js  v1.1.0
══════════════════════════════════════════════════════════ */
