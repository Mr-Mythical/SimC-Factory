from fastapi.testclient import TestClient

from web import credentials
from web.app import app


def test_core_dashboard_pages_render(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", str(tmp_path / ".credentials.json"))

    with TestClient(app) as client:
        expectations = {
            "/": "Dashboard",
            "/jobs/launch": "Launch Script",
            "/pipelines": "Pipelines",
            "/settings": "AWS Credentials",
        }
        for path, expected_text in expectations.items():
            response = client.get(path)
            assert response.status_code == 200
            assert expected_text in response.text


def test_settings_page_does_not_render_saved_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", str(tmp_path / ".credentials.json"))
    credentials.save({
        "aws_access_key_id": "TESTKEY",
        "aws_secret_access_key": "SUPERSECRET",
        "aws_session_token": "SESSIONSECRET",
        "aws_region": "us-east-1",
    })

    with TestClient(app) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert "TESTKEY" in response.text
    assert "SUPERSECRET" not in response.text
    assert "SESSIONSECRET" not in response.text
    assert "Leave blank to keep saved secret" in response.text


def test_basic_auth_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("SIMC_DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("SIMC_DASHBOARD_PASSWORD", "secret")

    with TestClient(app) as client:
        unauthenticated = client.get("/settings")
        authenticated = client.get("/settings", auth=("admin", "secret"))

    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["WWW-Authenticate"] == 'Basic realm="Mr. Mythical: SimC Factory"'
    assert authenticated.status_code == 200


def test_basic_auth_partial_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("SIMC_DASHBOARD_USERNAME", "admin")
    monkeypatch.delenv("SIMC_DASHBOARD_PASSWORD", raising=False)

    with TestClient(app) as client:
        response = client.get("/settings")

    assert response.status_code == 500
    assert "must be configured together" in response.text


def test_destructive_launch_requires_confirmation():
    with TestClient(app) as client:
        response = client.post("/jobs/launch", data={"preset_id": "wipe_all_data"})

    assert response.status_code == 400
    assert "WIPE ALL DATA" in response.text


def test_destructive_launch_form_shows_confirmation():
    with TestClient(app) as client:
        response = client.get("/jobs/launch/form?preset=terraform_destroy")

    assert response.status_code == 200
    assert "Destructive action confirmation required" in response.text
    assert "DESTROY INFRASTRUCTURE" in response.text


def test_specs_launch_rejects_non_spec_presets():
    with TestClient(app) as client:
        response = client.post("/specs/launch", data={"preset_id": "wipe_all_data", "specs": "all"})

    assert response.status_code == 400
    assert "not allowed" in response.text