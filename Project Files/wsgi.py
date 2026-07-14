"""
wsgi.py — OptiCrop WSGI Entry Point
====================================
Production entry point for WSGI servers (Gunicorn, uWSGI, etc.).

Usage:
    # Gunicorn (recommended for production)
    gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4 --timeout 120

    # uWSGI
    uwsgi --http 0.0.0.0:5000 --module wsgi:app --processes 4

    # Development (use run.py instead)
    python run.py

Gunicorn worker notes:
    --workers 4         : 2 × CPU cores + 1 is a good starting point.
    --timeout 120       : First-request model loading can take 2–3s.
    --worker-class sync : Default; adequate for this prediction workload.
                          For async (WebSocket), use 'eventlet' or 'gevent'.
"""

import os
from dotenv import load_dotenv

# Load environment variables before creating the app
load_dotenv()

from app import create_app  # noqa: E402

flask_env = os.getenv('FLASK_ENV', 'production')
app = create_app(flask_env)

if __name__ == '__main__':
    # Fallback for direct execution; use run.py for development.
    app.run()
