/* ============================================================
   OptiCrop – Smart Agricultural Production Optimization Engine
   about.js – About Page Vanilla JavaScript
   Author   : Principal Frontend Engineer
   Version  : 1.1.0  (Integration Release – global inits removed)

   NOTE: Global concerns (preloader, navbar scroll, scroll animations
   for .fade-in-up/.fade-in-left/.fade-in-right, counter animation,
   standard progress bars, back-to-top, smooth scroll, active nav)
   are now managed exclusively by script.js.

   This file contains ONLY about-page-specific logic:
   1.  Button Ripple (about-page selectors)
   2.  Algorithm Progress Bars (.algo-prog-fill)
   3.  ML Timeline stagger animation
   4.  FAQ Accordion keyboard accessibility
   ============================================================ */

'use strict';

/* ── DOM Ready ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Global: preloader, navbar, scroll animations (.fade-in-up / .fade-in-left
  // / .fade-in-right), counter animation, standard progress bars,
  // back-to-top, smooth scroll, active nav — all managed by script.js.
  initAlgoProgressBars();   // about-specific: .algo-prog-fill selector
  initAboutRipple();        // about-specific button selectors
  initMlTimeline();         // stagger animation for ML timeline items
  initFaqKeyboard();        // keyboard accessibility for .faq-button
});

/* ══════════════════════════════════════════════════════════
   1. ALGORITHM ACCURACY PROGRESS BARS
      (.algo-prog-fill selector is about-page specific;
       script.js handles .progress-fill – different class)
══════════════════════════════════════════════════════════ */
function initAlgoProgressBars() {
  const bars = document.querySelectorAll('.algo-prog-fill[data-width]');
  if (!bars.length) return;

  // Start all at zero width
  bars.forEach(bar => { bar.style.width = '0%'; });

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        setTimeout(() => {
          e.target.style.width = e.target.dataset.width + '%';
        }, 250);
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });

  bars.forEach(b => obs.observe(b));
}

/* ══════════════════════════════════════════════════════════
   2. BUTTON RIPPLE
      Targets about-page-specific button classes.
      The global script.js handles generic .btn-predict / .btn-pcta
      ripples; this handles about CTA buttons.
══════════════════════════════════════════════════════════ */
function initAboutRipple() {
  document.querySelectorAll('.btn-primary-custom, .btn-about-cta').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const r    = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      r.className = 'ripple';
      r.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - rect.left - size / 2}px;top:${e.clientY - rect.top - size / 2}px`;
      this.appendChild(r);
      r.addEventListener('animationend', () => r.remove());
    });
  });
}

/* ══════════════════════════════════════════════════════════
   3. ML TIMELINE – stagger entrance animation
      Adds .fade-in-up to each item and staggers visibility
      via IntersectionObserver + setTimeout delay.
══════════════════════════════════════════════════════════ */
function initMlTimeline() {
  const items = document.querySelectorAll('.ml-timeline-item');
  if (!items.length) return;

  items.forEach(item => item.classList.add('fade-in-up'));

  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        setTimeout(() => e.target.classList.add('visible'), i * 120);
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });

  items.forEach(item => obs.observe(item));
}

/* ══════════════════════════════════════════════════════════
   4. FAQ ACCORDION – keyboard accessibility
      Ensures Enter / Space trigger click on .faq-button
══════════════════════════════════════════════════════════ */
function initFaqKeyboard() {
  document.querySelectorAll('.faq-button').forEach(btn => {
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        btn.click();
      }
    });
  });
}

/* ══════════════════════════════════════════════════════════
   END about.js  v1.1.0
══════════════════════════════════════════════════════════ */
