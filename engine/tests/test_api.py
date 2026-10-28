"""Contract tests de la API HTTP (`api`).

Usan `fastapi.testclient.TestClient` con un motor falso parcheado para no
depender de `openvino_genai`, y validan:
  - `/api/tags` devuelve 200 con JSON (incluso sin modelos).
  - `/api/chat` requiere `messages` y `model` resueltos.
  - `/api/chat` con `stream=false` genera texto (motor falso).
  - El guard de seguridad rechaza Host/Origin no locales (SR-01).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from winllama_engine import api


class _FakeEngine:
    def __init__(self, text: str = "hola mundo") -> None:
        self.text = text

    def apply_chat_template(self, messages):  # noqa: ANN001
        return " | ".join(m["content"] for m in messages)

    def generate(self, prompt, options=None):  # noqa: ANN001
        return f"{self.text} ({prompt})"

    def stream(self, prompt, options=None):  # noqa: ANN001
        for piece in self.text.split():
            yield piece + " "


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> TestClient:
    model_dir = tmp_path / "qwen2-0_5b"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(
        api, "get_engine", lambda model_dir, device: _FakeEngine()
    )
    # El TestClient envia Host: testserver; se permite solo en tests.
    app = api.create_app(
        models_dir=str(tmp_path), extra_allowed_hosts={"testserver"}, **kwargs
    )
    return TestClient(app)


def test_tags_returns_200_json_even_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(api, "get_engine", lambda model_dir, device: _FakeEngine())
    app = api.create_app(
        models_dir=str(empty_dir), extra_allowed_hosts={"testserver"}
    )
    client = TestClient(app)
    response = client.get("/api/tags")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_tags_lists_registered_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.get("/api/tags")
    assert response.status_code == 200
    models = response.json()["models"]
    assert [m["name"] for m in models] == ["qwen2-0_5b"]
    assert models[0]["details"]["format"] == "openvino-ir"


def test_chat_requires_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.post("/api/chat", json={"model": "qwen2-0_5b"})
    assert response.status_code == 400


def test_chat_returns_generation_non_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "model": "qwen2-0_5b",
            "stream": False,
            "messages": [{"role": "user", "content": "hola"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert "hola mundo" in body["message"]["content"]


def test_chat_streams_ndjson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "model": "qwen2-0_5b",
            "stream": True,
            "messages": [{"role": "user", "content": "hola"}],
        },
    )
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.strip()]
    assert any('"done": true' in ln for ln in lines)


def test_chat_unknown_model_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/chat",
        json={"model": "nope", "messages": [{"role": "user", "content": "hola"}]},
    )
    assert response.status_code == 404


def test_security_guard_rejects_foreign_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.get("/api/tags", headers={"Host": "evil.example.com"})
    assert response.status_code == 403


def test_security_guard_rejects_browser_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.get("/api/tags", headers={"Origin": "http://evil.example.com"})
    assert response.status_code == 403


def test_doctor_reports_device_and_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    response = client.get("/api/doctor")
    assert response.status_code == 200
    body = response.json()
    assert "device" in body
    assert "models_dir" in body
