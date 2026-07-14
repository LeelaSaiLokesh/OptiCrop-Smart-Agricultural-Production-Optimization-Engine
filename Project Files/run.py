"""
OptiCrop – run.py
Flask Development Server Entry Point

Usage:
    python run.py                          → development mode (default)
    FLASK_ENV=production python run.py     → production mode
    set FLASK_ENV=production && python run.py  (Windows)

The environment is read from:
  1. The FLASK_ENV shell/environment variable
  2. A .env file in the project root (loaded via python-dotenv)
  3. Default: 'development'

Environment Variables (see .env.example for full list):
    FLASK_ENV    → 'development' | 'production' | 'testing'
    SECRET_KEY   → Override default secret (required in production)
    HOST         → Bind address (default: 0.0.0.0)
    PORT         → Port number  (default: 5000)
"""

import os

# Load .env file if present — must run before importing create_app
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed — variables must be set in the shell

from app import create_app

# ── Determine environment ─────────────────────────────────────────────────────
config_name: str = os.environ.get('FLASK_ENV', 'development')

# ── Create the application ────────────────────────────────────────────────────
app = create_app(config_name)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    host: str = os.environ.get('HOST', '0.0.0.0')
    port: int = int(os.environ.get('PORT', 5000))
    debug: bool = app.config.get('DEBUG', False)

    url_str = f'http://localhost:{port}/'
    print()
    print('  +--------------------------------------------------------+')
    print('  |      OptiCrop - AI Crop Recommendation Engine         |')
    print('  +--------------------------------------------------------+')
    print(f'  |  Environment  :  {config_name:<35}|')
    print(f'  |  Debug Mode   :  {str(debug):<35}|')
    print(f'  |  URL          :  {url_str:<35}|')
    print('  +--------------------------------------------------------+')
    print()

    app.run(host=host, port=port, debug=debug)
