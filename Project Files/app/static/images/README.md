# OptiCrop – Static Images Directory

Place project image assets here.

## Required Images

| File | Purpose | Recommended Size |
|---|---|---|
| `og-cover.png` | Open Graph / Twitter Card preview image | 1200×630 px |
| `logo.png` | Brand logo (alternative to emoji) | 512×512 px |
| `favicon.ico` | Browser tab icon | 32×32 px |

## Usage in Flask Templates

```html
<!-- Open Graph image (base.html) -->
<meta property="og:image" content="{{ url_for('static', filename='images/og-cover.png') }}" />

<!-- Logo in template -->
<img src="{{ url_for('static', filename='images/logo.png') }}" alt="OptiCrop Logo" />
```

## Notes

- All images should be optimized for web (compress with TinyPNG or similar)
- Use WebP format where browser support allows for better performance
- The `og-cover.png` is referenced in `base.html` — add it before deploying to production
