"""
OptiCrop – tests/test_routes.py
Flask Route Integration Tests

Tests all registered routes, error pages, validation, session,
and the prediction API endpoint.

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=term-missing

Coverage targets:
    app/routes.py           — 100%
    app/utils/validators.py — 100%
    app/services/prediction.py — 100%
"""

import json
import pytest

from app import create_app


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def app():
    """Create a TestingConfig Flask application for the module."""
    application = create_app('testing')
    application.config['TESTING']  = True
    application.config['DEBUG']    = True
    yield application


@pytest.fixture(scope='module')
def client(app):
    """Flask test client with session support."""
    return app.test_client()


@pytest.fixture
def valid_payload():
    """Valid prediction inputs (Rice profile from the Quick Fill demo)."""
    return {
        'nitrogen':    '90',
        'phosphorus':  '42',
        'potassium':   '43',
        'temperature': '20.8',
        'humidity':    '82.0',
        'ph':          '6.5',
        'rainfall':    '202.9',
    }


# ═══════════════════════════════════════════════════════════
# Home page
# ═══════════════════════════════════════════════════════════

class TestHomeRoute:
    def test_home_status_200(self, client):
        r = client.get('/')
        assert r.status_code == 200

    def test_home_alias_status_200(self, client):
        r = client.get('/home')
        assert r.status_code == 200

    def test_home_contains_opticrop(self, client):
        r = client.get('/')
        assert b'OptiCrop' in r.data

    def test_home_is_html(self, client):
        r = client.get('/')
        assert b'<!DOCTYPE html>' in r.data or b'<!doctype html>' in r.data.lower()


# ═══════════════════════════════════════════════════════════
# About page
# ═══════════════════════════════════════════════════════════

class TestAboutRoute:
    def test_about_status_200(self, client):
        r = client.get('/about')
        assert r.status_code == 200

    def test_about_contains_about(self, client):
        r = client.get('/about')
        assert b'About' in r.data or b'about' in r.data


# ═══════════════════════════════════════════════════════════
# Predict page — GET
# ═══════════════════════════════════════════════════════════

class TestPredictGet:
    def test_predict_get_status_200(self, client):
        r = client.get('/predict')
        assert r.status_code == 200

    def test_predict_contains_form(self, client):
        r = client.get('/predict')
        assert b'nitrogen' in r.data

    def test_predict_contains_all_fields(self, client):
        r = client.get('/predict')
        for field in [b'nitrogen', b'phosphorus', b'potassium',
                      b'temperature', b'humidity', b'ph', b'rainfall']:
            assert field in r.data, f'Field {field!r} not found in predict.html'


# ═══════════════════════════════════════════════════════════
# Predict API — POST (JSON response)
# ═══════════════════════════════════════════════════════════

class TestPredictPost:
    def test_predict_post_valid_returns_200(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        assert r.status_code == 200

    def test_predict_post_returns_json(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        assert r.content_type == 'application/json'

    def test_predict_post_success_flag(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        body = json.loads(r.data)
        assert body.get('success') is True

    def test_predict_post_returns_crop(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        body = json.loads(r.data)
        assert 'crop' in body
        assert isinstance(body['crop'], str)

    def test_predict_post_returns_confidence(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        body = json.loads(r.data)
        assert 'confidence' in body
        assert 0 <= float(body['confidence']) <= 100

    def test_predict_post_returns_feature_importance(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        body = json.loads(r.data)
        assert 'feature_importance' in body
        assert len(body['feature_importance']) == 15  # 7 raw + 8 engineered features

    def test_predict_post_returns_probabilities(self, client, valid_payload):
        r = client.post('/predict', data=valid_payload)
        body = json.loads(r.data)
        assert 'probabilities' in body
        assert len(body['probabilities']) >= 1

    # ── Validation failures ────────────────────────────────────────────────────

    def test_predict_post_missing_all_fields_returns_422(self, client):
        r = client.post('/predict', data={})
        assert r.status_code == 422

    def test_predict_post_missing_field_success_false(self, client):
        r = client.post('/predict', data={'nitrogen': '90'})  # 6 fields missing
        body = json.loads(r.data)
        assert body.get('success') is False
        assert 'errors' in body

    def test_predict_post_out_of_range_nitrogen(self, client, valid_payload):
        payload = {**valid_payload, 'nitrogen': '999'}  # max is 140
        r = client.post('/predict', data=payload)
        body = json.loads(r.data)
        assert r.status_code == 422
        assert 'nitrogen' in body['errors']

    def test_predict_post_invalid_string_nitrogen(self, client, valid_payload):
        payload = {**valid_payload, 'nitrogen': 'abc'}
        r = client.post('/predict', data=payload)
        body = json.loads(r.data)
        assert r.status_code == 422
        assert 'nitrogen' in body['errors']

    def test_predict_post_negative_ph(self, client, valid_payload):
        payload = {**valid_payload, 'ph': '-1.0'}  # min is 3.5
        r = client.post('/predict', data=payload)
        body = json.loads(r.data)
        assert r.status_code == 422
        assert 'ph' in body['errors']

    def test_predict_post_json_body_accepted(self, client, valid_payload):
        """POST /predict also accepts application/json body."""
        r = client.post(
            '/predict',
            data=json.dumps({k: float(v) for k, v in valid_payload.items()}),
            content_type='application/json',
        )
        assert r.status_code == 200
        body = json.loads(r.data)
        assert body.get('success') is True


# ═══════════════════════════════════════════════════════════
# Result page — GET
# ═══════════════════════════════════════════════════════════

class TestResultRoute:
    def test_result_get_no_session_redirects(self, client):
        """Without a session prediction, /result redirects to /predict."""
        with client.session_transaction() as sess:
            sess.clear()
        r = client.get('/result', follow_redirects=False)
        assert r.status_code == 302

    def test_result_get_with_session_returns_200(self, client, valid_payload):
        """Submit a prediction first, then /result should return 200."""
        client.post('/predict', data=valid_payload)
        r = client.get('/result')
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════

class TestHealthRoute:
    def test_health_status_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200

    def test_health_returns_ok(self, client):
        r = client.get('/health')
        body = json.loads(r.data)
        assert body['status'] == 'ok'
        assert body['app'] == 'OptiCrop'


# ═══════════════════════════════════════════════════════════
# Error pages
# ═══════════════════════════════════════════════════════════

class TestErrorPages:
    def test_404_returns_404(self, client):
        r = client.get('/this-page-does-not-exist-xyz')
        assert r.status_code == 404

    def test_404_contains_opticrop(self, client):
        r = client.get('/nonexistent-route-abc')
        assert b'OptiCrop' in r.data or b'404' in r.data


# ═══════════════════════════════════════════════════════════
# Validators unit tests
# ═══════════════════════════════════════════════════════════

class TestValidators:
    """Unit tests for app.utils.validators — no Flask context required."""

    def _valid(self):
        return {
            'nitrogen': '90', 'phosphorus': '42', 'potassium': '43',
            'temperature': '20.8', 'humidity': '82.0',
            'ph': '6.5', 'rainfall': '202.9',
        }

    def test_valid_inputs_pass(self):
        from app.utils.validators import validate_prediction_inputs
        result = validate_prediction_inputs(self._valid())
        assert result['nitrogen'] == 90.0
        assert result['ph'] == 6.5

    def test_empty_string_raises(self):
        from app.utils.validators import validate_prediction_inputs, ValidationError
        bad = {**self._valid(), 'nitrogen': ''}
        with pytest.raises(ValidationError) as exc_info:
            validate_prediction_inputs(bad)
        assert 'nitrogen' in exc_info.value.errors

    def test_non_numeric_raises(self):
        from app.utils.validators import validate_prediction_inputs, ValidationError
        bad = {**self._valid(), 'ph': 'seven'}
        with pytest.raises(ValidationError) as exc_info:
            validate_prediction_inputs(bad)
        assert 'ph' in exc_info.value.errors

    def test_below_min_raises(self):
        from app.utils.validators import validate_prediction_inputs, ValidationError
        bad = {**self._valid(), 'rainfall': '1'}   # min 20
        with pytest.raises(ValidationError) as exc_info:
            validate_prediction_inputs(bad)
        assert 'rainfall' in exc_info.value.errors

    def test_above_max_raises(self):
        from app.utils.validators import validate_prediction_inputs, ValidationError
        bad = {**self._valid(), 'potassium': '999'}   # max 205
        with pytest.raises(ValidationError) as exc_info:
            validate_prediction_inputs(bad)
        assert 'potassium' in exc_info.value.errors

    def test_multiple_errors_collected(self):
        from app.utils.validators import validate_prediction_inputs, ValidationError
        bad = {'nitrogen': '', 'phosphorus': '', 'potassium': '',
               'temperature': '', 'humidity': '', 'ph': '', 'rainfall': ''}
        with pytest.raises(ValidationError) as exc_info:
            validate_prediction_inputs(bad)
        assert len(exc_info.value.errors) == 7
