import importlib
import os


def _reload_app():
    import video_shot_splitter.web.app as app
    return importlib.reload(app)


def test_max_upload_default(monkeypatch):
    monkeypatch.delenv("VSS_MAX_UPLOAD_MB", raising=False)
    app = _reload_app()
    assert app.MAX_UPLOAD_BYTES == 200 * 1024 * 1024


def test_max_upload_env_override(monkeypatch):
    monkeypatch.setenv("VSS_MAX_UPLOAD_MB", "50")
    app = _reload_app()
    assert app.MAX_UPLOAD_BYTES == 50 * 1024 * 1024


def test_server_config_defaults(monkeypatch):
    monkeypatch.delenv("VSS_HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    app = _reload_app()
    host, port = app.server_config()
    assert host == "0.0.0.0"
    assert port == 7860


def test_server_config_env_override(monkeypatch):
    monkeypatch.setenv("VSS_HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8000")
    app = _reload_app()
    host, port = app.server_config()
    assert host == "127.0.0.1"
    assert port == 8000
