/* ============================================================
   OptiCrop – Smart Agricultural Production Optimization Engine
   script.js – Global JavaScript Controller (ALL Pages)
   Author   : Senior Frontend Engineer
   Version  : 2.0.0  (Integration Release)

   Responsibilities:
     • Preloader     • Page Transitions   • Navbar Scroll
     • Scroll Animations (IntersectionObserver)
     • Counter Animation  • Progress Bars
     • Ripple Effect   • Smooth Scroll
     • Back-to-Top     • Active Nav
     • Toast Utility    • Skip Navigation
     • Hero Canvas      • Typed Effect
     • Bootstrap Tooltips
   ============================================================ */

'use strict';

/* ══════════════════════════════════════════════════════════
   UTILITY: DOM Ready Bootstrapper
══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initPreloader();        // Must be first
  initPageTransition();  // Intercept navigation links
  initNavbar();          // Sticky scroll behaviour
  initScrollAnimations();
  initCounterAnimation();
  initProgressBars();
  initRippleEffect();
  initActiveNav();
  initSmoothScroll();
  initBackToTop();
  initSkipNav();
  initTechTooltips();
});


/* ══════════════════════════════════════════════════════════
   1. NAVBAR – Scroll behaviour & active state
══════════════════════════════════════════════════════════ */
function initNavbar() {
  const nav = document.getElementById('mainNav');
  if (!nav) return;

  const handleScroll = () => {
    if (window.scrollY > 60) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  };

  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll(); // Run once on load
}

/* ── Active nav link based on scroll position ─────────────── */
function initActiveNav() {
  const sections  = document.querySelectorAll('section[id]');
  const navLinks  = document.querySelectorAll('.navbar-nav .nav-link[href^="#"]');

  if (!sections.length || !navLinks.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '-30% 0px -60% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));
}

/* ══════════════════════════════════════════════════════════
   2. SCROLL ANIMATIONS (Intersection Observer)
══════════════════════════════════════════════════════════ */
function initScrollAnimations() {
  const animatedElements = document.querySelectorAll(
    '.fade-in-up, .fade-in-left, .fade-in-right'
  );

  if (!animatedElements.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -60px 0px',
    threshold: 0.1
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target); // Animate once
      }
    });
  }, observerOptions);

  animatedElements.forEach(el => observer.observe(el));
}

/* ══════════════════════════════════════════════════════════
   3. COUNTER ANIMATION
══════════════════════════════════════════════════════════ */
function initCounterAnimation() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.4
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  counters.forEach(counter => observer.observe(counter));
}

/**
 * Smoothly counts up a number element to its target value.
 * Supports suffixes like "+", "%", "k".
 * @param {HTMLElement} el – The element with data-counter attribute
 */
function animateCounter(el) {
  const target   = parseFloat(el.dataset.counter);
  const suffix   = el.dataset.suffix || '';
  const prefix   = el.dataset.prefix || '';
  const duration = 2000; // ms
  const steps    = 60;
  const stepTime = Math.floor(duration / steps);
  let  current   = 0;
  const increment = target / steps;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      current = target;
      clearInterval(timer);
    }
    el.textContent = prefix + formatNumber(current, target) + suffix;
  }, stepTime);
}

/**
 * Formats the number – if target has decimal, show one decimal place;
 * otherwise show integer.
 */
function formatNumber(val, target) {
  if (Number.isInteger(target)) {
    return Math.floor(val).toLocaleString();
  }
  return val.toFixed(1);
}

/* ══════════════════════════════════════════════════════════
   4. PROGRESS BARS
══════════════════════════════════════════════════════════ */
function initProgressBars() {
  const bars = document.querySelectorAll('.progress-fill[data-width]');
  if (!bars.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.3
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const bar = entry.target;
        const targetWidth = bar.dataset.width;
        // Delay slightly for visual effect
        setTimeout(() => {
          bar.style.width = targetWidth + '%';
        }, 200);
        observer.unobserve(bar);
      }
    });
  }, observerOptions);

  // Set initial width to 0 for animation
  bars.forEach(bar => {
    bar.style.width = '0%';
    observer.observe(bar);
  });
}

/* ══════════════════════════════════════════════════════════
   5. BUTTON RIPPLE EFFECT
══════════════════════════════════════════════════════════ */
function initRippleEffect() {
  const rippleButtons = document.querySelectorAll('.btn-primary-custom');

  rippleButtons.forEach(btn => {
    btn.addEventListener('click', function (e) {
      const ripple = document.createElement('span');
      ripple.classList.add('ripple');

      const rect   = this.getBoundingClientRect();
      const size   = Math.max(rect.width, rect.height);
      const x      = e.clientX - rect.left - size / 2;
      const y      = e.clientY - rect.top  - size / 2;

      ripple.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
      `;

      this.appendChild(ripple);

      ripple.addEventListener('animationend', () => ripple.remove());
    });
  });
}

/* ══════════════════════════════════════════════════════════
   6. SMOOTH SCROLL for anchor links
══════════════════════════════════════════════════════════ */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;

      const target = document.querySelector(href);
      if (!target) return;

      e.preventDefault();

      // Close mobile menu if open
      const navbarCollapse = document.querySelector('.navbar-collapse');
      if (navbarCollapse && navbarCollapse.classList.contains('show')) {
        const bsCollapse = bootstrap.Collapse.getInstance(navbarCollapse);
        if (bsCollapse) bsCollapse.hide();
      }

      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

/* ══════════════════════════════════════════════════════════
   7. HERO – Typed text effect (subtitle cycling)
══════════════════════════════════════════════════════════ */
(function initTypedEffect() {
  const el = document.getElementById('typed-subtitle');
  if (!el) return;

  const phrases = [
    'AI-Powered Crop Recommendation',
    'Precision Farming Intelligence',
    'Data-Driven Agricultural Optimization',
    'Smart Soil & Climate Analysis',
    'Predictive Modeling for Farmers'
  ];

  let phraseIndex = 0;
  let charIndex   = 0;
  let isDeleting  = false;
  let typingDelay = 80;

  function type() {
    const current = phrases[phraseIndex];

    if (!isDeleting) {
      el.textContent = current.substring(0, charIndex + 1);
      charIndex++;

      if (charIndex === current.length) {
        isDeleting  = true;
        typingDelay = 2500; // Pause before deleting
      } else {
        typingDelay = 75;
      }
    } else {
      el.textContent = current.substring(0, charIndex - 1);
      charIndex--;

      if (charIndex === 0) {
        isDeleting  = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        typingDelay = 400;
      } else {
        typingDelay = 40;
      }
    }

    setTimeout(type, typingDelay);
  }

  // Start after hero animation completes
  setTimeout(type, 1200);
})();

/* ══════════════════════════════════════════════════════════
   8. PARTICLE / FLOATING DOT ANIMATION (Hero Canvas)
══════════════════════════════════════════════════════════ */
(function initHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;

  const ctx    = canvas.getContext('2d');
  let   width, height, particles;

  const COLORS = ['rgba(22,163,74,', 'rgba(14,165,233,', 'rgba(245,158,11,'];

  class Particle {
    constructor() { this.reset(); }

    reset() {
      this.x    = Math.random() * width;
      this.y    = Math.random() * height;
      this.size = Math.random() * 2.5 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.4;
      this.speedY = (Math.random() - 0.5) * 0.4;
      this.color = COLORS[Math.floor(Math.random() * COLORS.length)];
      this.alpha = Math.random() * 0.4 + 0.1;
    }

    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      if (this.x < 0 || this.x > width || this.y < 0 || this.y > height) {
        this.reset();
      }
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color + this.alpha + ')';
      ctx.fill();
    }
  }

  function resize() {
    width  = canvas.width  = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  }

  function init() {
    resize();
    const count = Math.floor((width * height) / 18000);
    particles   = Array.from({ length: count }, () => new Particle());
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    particles.forEach(p => { p.update(); p.draw(); });

    // Draw connecting lines between nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx   = particles[i].x - particles[j].x;
        const dy   = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(22,163,74,${0.08 * (1 - dist / 100)})`;
          ctx.lineWidth   = 0.5;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', () => {
    resize();
    particles.forEach(p => p.reset());
  }, { passive: true });

  init();
  animate();
})();

/* ══════════════════════════════════════════════════════════
   9. NAVBAR TOGGLER ICON ANIMATION
══════════════════════════════════════════════════════════ */
(function initTogglerAnimation() {
  const toggler = document.querySelector('.navbar-toggler');
  if (!toggler) return;

  toggler.addEventListener('click', function () {
    this.classList.toggle('is-active');
  });
})();

/* ══════════════════════════════════════════════════════════
   10. BACK TO TOP BUTTON (scroll-triggered visibility)
══════════════════════════════════════════════════════════ */
(function initBackToTop() {
  const btn = document.querySelector('.back-to-top');
  if (!btn) return;

  const handleScroll = () => {
    if (window.scrollY > 500) {
      btn.style.opacity = '1';
      btn.style.pointerEvents = 'auto';
    } else {
      btn.style.opacity = '0';
      btn.style.pointerEvents = 'none';
    }
  };

  // Initial state
  btn.style.opacity = '0';
  btn.style.pointerEvents = 'none';
  btn.style.transition = 'opacity 0.3s ease';

  window.addEventListener('scroll', handleScroll, { passive: true });
})();

/* ══════════════════════════════════════════════════════════
   11. TECH BADGE – Tooltip on hover (vanilla)
══════════════════════════════════════════════════════════ */
(function initTechTooltips() {
  // Bootstrap tooltips init if available
  if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
  }
})();

/* ══════════════════════════════════════════════════════════
   12. PRELOADER (fades out when page is fully loaded)
══════════════════════════════════════════════════════════ */
function initPreloader() {
  const preloader = document.getElementById('preloader');
  if (!preloader) return;

  const hide = () => {
    preloader.classList.add('hidden');
    setTimeout(() => {
      preloader.style.display = 'none';
    }, 500);
  };

  if (document.readyState === 'complete') {
    hide();
  } else {
    window.addEventListener('load', hide);
    // Fallback: always hide after 3s even if load event is slow
    setTimeout(hide, 3000);
  }
}

/* ══════════════════════════════════════════════════════════
   13. PAGE TRANSITION – Smooth fade on internal navigation
══════════════════════════════════════════════════════════ */
function initPageTransition() {
  const overlay = document.getElementById('page-transition');
  if (!overlay) return;

  // Intercept all anchor clicks that navigate to another page (not anchors)
  document.addEventListener('click', (e) => {
    const anchor = e.target.closest('a');
    if (!anchor) return;

    const href = anchor.getAttribute('href');
    if (!href) return;

    // Skip: external links, anchor-only links, javascript:, mailto:, tel:
    const isExternal  = anchor.target === '_blank' || /^https?:///.test(href);
    const isAnchor    = href.startsWith('#');
    const isSpecial   = /^(javascript:|mailto:|tel:)/.test(href);

    if (isExternal || isAnchor || isSpecial) return;

    e.preventDefault();

    // Apply leaving transition
    overlay.classList.add('leaving');

    setTimeout(() => {
      window.location.href = href;
    }, 280);
  });

  // On back/forward navigation, remove overlay
  window.addEventListener('pageshow', () => {
    overlay.classList.remove('leaving');
  });
}

/* ══════════════════════════════════════════════════════════
   14. SKIP NAVIGATION (Accessibility – WCAG 2.1 SC 2.4.1)
══════════════════════════════════════════════════════════ */
function initSkipNav() {
  const skipLink   = document.querySelector('.skip-nav');
  const mainContent = document.getElementById('main-content');
  if (!skipLink || !mainContent) return;

  skipLink.addEventListener('click', (e) => {
    e.preventDefault();
    mainContent.setAttribute('tabindex', '-1');
    mainContent.focus();
    mainContent.scrollIntoView({ behavior: 'smooth' });
  });
}

/* ══════════════════════════════════════════════════════════
   15. TOAST NOTIFICATION UTILITY (Global)
       Usage: window.showToast('Message', 'success'|'error'|'warning'|'info')
══════════════════════════════════════════════════════════ */
window.showToast = function(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const iconMap = {
    success: 'bi-check-circle-fill toast-icon-success',
    error:   'bi-exclamation-circle-fill toast-icon-error',
    warning: 'bi-exclamation-triangle-fill toast-icon-warning',
    info:    'bi-info-circle-fill toast-icon-info'
  };

  const icon = iconMap[type] || iconMap.info;
  const id   = 'toast-' + Date.now();

  const toastHTML = `
    <div id="${id}" class="toast opticrop-toast align-items-center" role="alert"
         aria-live="assertive" aria-atomic="true" data-bs-delay="${duration}">
      <div class="toast-header">
        <i class="bi ${icon} me-2" aria-hidden="true"></i>
        <strong class="me-auto">OptiCrop</strong>
        <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
      <div class="toast-body">${message}</div>
    </div>`;

  container.insertAdjacentHTML('beforeend', toastHTML);

  const toastEl = document.getElementById(id);
  const bsToast = new bootstrap.Toast(toastEl, { delay: duration });
  bsToast.show();

  // Auto-remove from DOM after hiding
  toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
};

/* ══════════════════════════════════════════════════════════
   END OF GLOBAL SCRIPT (script.js v2.0.0)
══════════════════════════════════════════════════════════ */
