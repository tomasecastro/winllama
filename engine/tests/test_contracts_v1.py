"""Contract tests `/v1` (contrato tipo OpenAI) de la 002 (T-221 / T-225).

Shapes EXACTAS de `contracts/api.md` §3 según la referencia OpenAI:
  - POST /v1/chat/completions no-stream:
    {id, object:"chat.completion", created, model,
     choices:[{index, message{role, content}, finish_reason}],
     usage:{prompt_tokens, completion_tokens, total_tokens}}
  - POST /v1/chat/completions stream: SSE (`data: {..}\\n\\n`, cierre
    `data: [DONE]\\n\\n`) con `chat.completion.chunk` y `delta.content`.
  - GET /v1/models: {object:"list", data:[{id, object:"model", created, owned_by}]}.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import make_client


def test_chat_completions_non_stream_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2-0_5b",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "qwen2-0_5b"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"].strip() == "hola mundo"
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] >= 0
    assert usage["completion_tokens"] >= 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_stream_sse_with_done(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2-0_5b",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    raw = response.text
    # `data: {..}\n\n` por chunk; cierre exacto `data: [DONE]\n\n`.
    assert "data: [DONE]\n\n" in raw
    events = [ln.removeprefix("data: ") for ln in raw.split("\n\n") if ln.strip()]
    chunks = [json.loads(ev) if ev != "[DONE]" else "[DONE]" for ev in events]
    assert chunks[-1] == "[DONE]"
    body_chunks = [c for c in chunks if c != "[DONE]"]
    for chunk in body_chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")
        choice = chunk["choices"][0]
        assert choice["index"] == 0
        assert "delta" in choice
    # Al MENOS un delta de contenido + el cierre con finish_reason "stop".
    deltas = "".join(
        c["choices"][0]["delta"].get("content", "") for c in body_chunks
    )
    assert "hola " in deltas
    finish = body_chunks[-1]["choices"][0]["finish_reason"]
    assert finish == "stop"


def test_models_list_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and body["data"]
    row = body["data"][0]
    assert row["id"] == "qwen2-0_5b"
    assert row["object"] == "model"
    assert isinstance(row["created"], int)
    assert row["owned_by"] == "winllama-engine"


def test_models_list_empty_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile
    from pathlib import Path

    from winllama_engine import api
    from fastapi.testclient import TestClient

    tmp = Path(tempfile.mkdtemp(prefix="winllama-v1-empty-"))
    monkeypatch.setattr(
        api, "get_engine", lambda model_dir, device: None
    )
    app = api.create_app(models_dir=str(tmp), extra_allowed_hosts={"testserver"})
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_chat_completions_default_model_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _engine, _store = make_client(monkeypatch, default_model="qwen2-0_5b")
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}], "stream": False},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "qwen2-0_5b"


def test_chat_completions_v1_params_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """temperature/max_tokens/stop del contrato `/v1` llegan como options del core."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2-0_5b",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": False,
            "temperature": 0.4,
            "max_tokens": 42,
            "stop": ["END"],
        },
    )
    assert response.status_code == 200
    options = engine.requests[-1]["options"]
    from winllama_engine.pipeline import GenOptions

    assert isinstance(options, GenOptions)
    assert options.max_new_tokens == 42
    assert options.temperature == 0.4
    assert options.stop == ["END"]


def test_chat_completions_unknown_model_404(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404
    assert "error" in response.json()


def test_chat_completions_c101_traversal_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-101 aplicado tambien en la ruta `/v1` (T-220 pide su caso en T-225)."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "../escape",
            "messages": [{"role": "user", "content": "x"}],
            "stream": False,
        },
    )
    assert response.status_code == 400
    assert engine.requests == []


def test_chat_completions_options_cap_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-103: num_predict cap tambien en la ruta `/v1` (max_tokens)."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2-0_5b",
            "messages": [{"role": "user", "content": "x"}],
            "stream": False,
            "max_tokens": 100000,
        },
    )
    assert response.status_code == 400
    assert engine.requests == []
