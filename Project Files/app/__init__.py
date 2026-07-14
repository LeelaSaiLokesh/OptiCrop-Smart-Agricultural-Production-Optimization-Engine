"""
OptiCrop – app/__init__.py
Flask Application Factory

Creates, configures, and returns a fully wired Flask application instance.

Architecture:
  create_app(config_name)
    ├── _load_config()           → config.py registry
    ├── _configure_logging()     → app/utils/logger.py
    ├── _register_blueprints()   → app/routes.py (main Blueprint)
    ├── _register_error_handlers() → 400/404/405/500 + catchall
    └── _register_context_processors() → global Jinja2 vars

Usage:
    from app import create_app
    app = create_app('development')   # or 'production' / 'testing'
"""

import os
from flask import Flask, render_template


def create_app(config_name: str = 'default') -> Flask:
    """
    Application factory — the single entry point for Flask app creation.

    Args:
        config_name: Environment name — 'development', 'production',
                     'testing', or 'default' (maps to DevelopmentConfig)

    Returns:
        Fully configured Flask application instance
    """
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    _load_config(app, config_name)
    _configure_logging(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_context_processors(app)

    app.logger.info(
        'OptiCrop initialized  |  env=%-12s  |  debug=%s',
        config_name, app.config.get('DEBUG')
    )

    return app


# ═══════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════

def _load_config(app: Flask, config_name: str) -> None:
    """
    Loads the matching Config class from config.py and applies any
    SECRET_KEY override from the runtime environment.
    """
    from config import config as cfg_registry  # project-root config.py

    app.config.from_object(cfg_registry[config_name])

    # Runtime environment always wins over class defaults
    env_secret = os.environ.get('SECRET_KEY')
    if env_secret:
        app.config['SECRET_KEY'] = env_secret


def _configure_logging(app: Flask) -> None:
    """Delegates logging setup to the dedicated logger utility."""
    from app.utils.logger import configure_logging
    configure_logging(app)


def _register_blueprints(app: Flask) -> None:
    """Imports and registers all application Blueprints."""
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    app.logger.debug('Blueprint registered: main')


def _register_error_handlers(app: Flask) -> None:
    """
    Registers custom HTTP error handlers.

    All handlers render templates from templates/errors/ so the user
    sees a branded, helpful error page rather than the Flask default.
    """

    @app.errorhandler(400)
    def bad_request(error):
        app.logger.warning('400 Bad Request | %s', error)
        return render_template(
            'errors/error.html',
            code=400,
            title='Bad Request',
            headline='Invalid Request',
            message=(
                'The server could not process your request. '
                'Please check your input and try again.'
            ),
            icon='bi-exclamation-circle-fill',
        ), 400

    @app.errorhandler(404)
    def not_found(error):
        app.logger.warning('404 Not Found | %s', error)
        return render_template(
            'errors/404.html',
            code=404,
            title='Page Not Found',
            headline='Page Not Found',
            message=(
                "The page you're looking for doesn't exist or may have been moved. "
                "Let's get you back to the crop recommendation engine."
            ),
        ), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        app.logger.warning('405 Method Not Allowed | %s', error)
        return render_template(
            'errors/error.html',
            code=405,
            title='Method Not Allowed',
            headline='Method Not Allowed',
            message='This HTTP method is not supported for the requested endpoint.',
            icon='bi-slash-circle-fill',
        ), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error('500 Internal Server Error | %s', error, exc_info=True)
        return render_template(
            'errors/500.html',
            code=500,
            title='Server Error',
            headline='Something Went Wrong',
            message=(
                'An internal server error occurred. '
                'Please try again in a moment.'
            ),
        ), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """Catch-all for any unhandled Python exception."""
        app.logger.exception('Unhandled exception | %s', str(error))
        return render_template(
            'errors/500.html',
            code=500,
            title='Unexpected Error',
            headline='Unexpected Error',
            message=(
                'An unexpected error occurred. '
                'Our engineering team has been notified.'
            ),
        ), 500


def _register_context_processors(app: Flask) -> None:
    """
    Injects global Jinja2 template variables available on every page.

    Variables injected:
      {{ app_name }}    → 'OptiCrop'
      {{ app_version }} → '1.0.0'
    """

    @app.context_processor
    def inject_app_globals() -> dict:
        return {
            'app_name':    app.config.get('APP_NAME',    'OptiCrop'),
            'app_version': app.config.get('APP_VERSION', '1.0.0'),
        }
