"""Contract tests del contrato `/api` (tipo Ollama) de la 002 (T-220 / T-223 / T-229 / T-224).

Shapes EXACTAS de `contracts/api.md` §1 (generate NDJSON + timing ns),
§2 (chat), §4 (/api/ps), §7 (/api/show), §8 (DELETE /api/delete) y
controles de seguridad C-101/C-102/C-106 (CA-S16, CA-S17, CA-S21, CA-S22).

Usa `tests/conftest.FakeGenEngine` (no requiere GPU ni openvino_genai).
Duraciones SIEMPRE en nanosegundos (perf_counter_ns).
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from winllama_engine.pipeline import MAX_NUM_CTX, WRITE_TIMEOUT_S  # noqa: F401
from tests.conftest import TEST_HOST, FakeGenEngine, make_client

NDJSON_NEEDED_FINAL_KEYS = {
    "model",
    "created_at",
    "response",
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "context",
}


def _parse_ndjson(response_text: str) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in response_text.splitlines() if ln.strip()]


def test_generate_streams_ndjson_token_per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contrato §1 stream: {model, created_at, response, done:false} por token."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={"model": "qwen2-0_5b", "prompt": "Di hola", "stream": True},
    )
    assert response.status_code == 200
    lines = _parse_ndjson(response.text)
    assert len(lines) >= 2
    token_chunk = lines[0]
    assert token_chunk["model"] == "qwen2-0_5b"
    assert token_chunk["response"] == "hola "
    assert token_chunk["done"] is False
    assert token_chunk["created_at"]

    final = lines[-1]
    assert NDJSON_NEEDED_FINAL_KEYS <= set(final)
    assert final["response"] == ""
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    # Duraciones en NANOSEGUNDOS (contrato: SIEMPRE ns).
    assert all(
        isinstance(final[k], int) and final[k] >= 0
        for k in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    )
    assert final["context"] == []


def test_generate_non_stream_full_object_with_ns_timing(monkeypatch) -> None:
    """Contrato §1 no-stream: un objeto con timing completo + done_reason."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={"model": "qwen2-0_5b", "prompt": "Di hola", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert NDJSON_NEEDED_FINAL_KEYS <= set(body)
    assert body["response"].strip() == "hola mundo"
    assert body["done_reason"] == "stop"
    assert body["prompt_eval_count"] >= 0


def test_generate_empty_prompt_loads_model_done_reason_load(monkeypatch) -> None:
    """Edge §1: prompt vacio + sin keep_alive -> load (done_reason 'load')."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post("/api/generate", json={"model": "qwen2-0_5b", "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["done_reason"] == "load"
    assert body["response"] == ""
    assert body["eval_count"] == 0
    assert engine.requests == []  # la carga NO llama a generate/stream


def test_generate_keep_alive_zero_unloads(tmp_path_factory, monkeypatch) -> None:
    """Contrato §1: keep_alive:0 -> unload (done_reason 'unload')."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={
            "model": "qwen2-0_5b",
            "prompt": "Di hola",
            "stream": False,
            "keep_alive": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["done_reason"] == "unload"
    assert engine.unloaded is True


def test_generate_unknown_model_404(tmp_path_factory, monkeypatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate", json={"model": "nope", "prompt": "x", "stream": False}
    )
    assert response.status_code == 404
    assert "error" in response.json()


def test_generate_missing_model_no_default_400(tmp_path_factory, monkeypatch) -> None:
    """Sin model y sin default: 400 con lista de disponibles (semantica 400)."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post("/api/generate", json={"prompt": "x", "stream": False})
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert "available_models" in body


def test_chat_streams_message_chunks_and_empty_final_content(
    tmp_path_factory, monkeypatch
) -> None:
    """Contrato §2: chunk {model, created_at, message{role, content}, done:false};
    final con message.content:'' y las mismas metricas."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "model": "qwen2-0_5b",
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    lines = _parse_ndjson(response.text)
    assert lines[0]["model"] == "qwen2-0_5b"
    assert lines[0]["message"]["role"] == "assistant"
    assert lines[0]["message"]["content"] == "hola "
    assert lines[0]["done"] is False
    final = lines[-1]
    assert final["done"] is True
    assert final["message"]["content"] == ""
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        assert isinstance(final[key], int) and final[key] >= 0


def test_chat_multi_turn_accepts_history(monkeypatch) -> None:
    """Contrato §2: multi-turno (history aceptado) — el template recibe todo."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "model": "qwen2-0_5b",
            "stream": False,
            "messages": [
                {"role": "user", "content": "why is the sky blue?"},
                {"role": "assistant", "content": "rayleigh scattering."},
                {"role": "user", "content": "how is that different from mie?"},
            ],
        },
    )
    assert response.status_code == 200
    prompt_used = engine.requests[-1]["prompt"]
    assert "rayleigh scattering." in prompt_used
    assert "how is that different from mie?" in prompt_used


# ------------------------------------------------------------------ /api/ps
def test_ps_lists_loaded_model_with_details(
    tmp_path_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrato §4: name/model/size/digest/details/expires_at."""
    import re

    client, _engine, _store = make_client(monkeypatch)
    client.post(
        "/api/generate",
        json={"model": "qwen2-0_5b", "prompt": "Di hola", "stream": False},
    )
    ps = client.get("/api/ps")
    assert ps.status_code == 200
    body = ps.json()
    assert isinstance(body.get("models"), list) and body["models"], body
    row = body["models"][0]
    assert row["name"] == "qwen2-0_5b"
    assert row["model"] == "qwen2-0_5b"
    assert isinstance(row["size"], int) and row["size"] > 0
    assert isinstance(row["digest"], str) and row["digest"].startswith("sha256:")
    details = row["details"]
    assert details["format"] == "openvino-ir"
    assert "expires_at" in row  # RFC3339/ISO; sin expiracion matematica tests: no nulo
    assert isinstance(row.get("size_vram", 0), int)
    # sha256 hex 64 chars (o menor alinee 'digest' de ps usa 12 hex -> OK)
    assert re.fullmatch(r"sha256:[0-9a-f]{12}", row["digest"]) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", row["digest"]
    )


def test_ps_empty_when_nothing_loaded(tmp_path_factory, monkeypatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    ps = client.get("/api/ps")
    assert ps.status_code == 200
    assert ps.json() == {"models": []}


# ------------------------------------------------- /api/show y DELETE /api/delete
def test_show_returns_manifest_details(tmp_path_factory, monkeypatch) -> None:
    """Contrato §7: details/model_info/capabilities desde el manifest local."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post("/api/show", json={"model": "qwen2-0_5b"})
    assert response.status_code == 200
    body = response.json()
    assert body["details"]["format"] == "openvino-ir"
    assert "model_info" in body
    assert isinstance(body["capabilities"], list) and body["capabilities"] == ["completion"]


def test_show_unknown_model_404(tmp_path_factory, monkeypatch) -> None:
    client, _engine, _store = make_client(monkeypatch)
    response = client.post("/api/show", json={"model": "nope"})
    assert response.status_code == 404
    assert "error" in response.json()


def test_delete_removes_model_and_refreshes_tags(
    tmp_path_factory, monkeypatch
) -> None:
    """Contrato §8: 200 {status:success}; 404 si no existe; tags refresca."""
    client, _engine, _store = make_client(monkeypatch)
    before = client.get("/api/tags").json()["models"]
    assert [m["name"] for m in before] == ["qwen2-0_5b"]
    deleted = client.request(
        "DELETE", "/api/delete", json={"model": "qwen2-0_5b"}
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "success"}
    after = client.get("/api/tags").json()["models"]
    assert after == []
    gone = client.request("DELETE", "/api/delete", json={"model": "qwen2-0_5b"})
    assert gone.status_code == 404
    assert "error" in gone.json()


# ----------------------------------------------------------- seguridad (CAs)
def test_ca_s16_generate_rejects_foreign_origin(
    tmp_path_factory, monkeypatch
) -> None:
    """C-102/CA-S16: POSTs de control tambien con guard de Origin/Host."""
    client, _engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={"model": "qwen2-0_5b", "prompt": "x"},
        headers={"Origin": "https://evil.example.com"},
    )
    assert response.status_code == 403


def test_ca_s17_traversal_on_all_model_endpoints(
    tmp_path_factory, monkeypatch
) -> None:
    """C-101/CA-S17: `model='../x'` -> 400 en generate/chat/show/delete."""
    client, engine, _store = make_client(monkeypatch)
    for status in (
        client.post("/api/generate", json={"model": "../escape", "prompt": "x", "stream": False}),
        client.post("/api/chat", json={"model": "../escape", "messages": [{"role": "user", "content": "x"}]}),
        client.post("/api/show", json={"model": "../escape"}),
        client.request("DELETE", "/api/delete", json={"model": "../escape"}),
    ):
        assert status.status_code == 400, (status.request.url, status.status_code)
    assert engine.requests == []


def test_ca_s17_c101_containment_realpath(tmp_path_factory, monkeypatch) -> None:
    """C-101 capa 2: un nombre que resuelva FUERA del cache -> fail-closed."""
    client, _engine, store = make_client(monkeypatch)
    # `qwen2-0_5b` hacia un path con symlink no se produce aqui; se testea
    # la barrera sintetica directamente via nota '...' no-nombre:
    forbidden = ["C___root", "AUX", "aux", "a@@b", "qwen2/", "/absolute"]
    for name in forbidden:
        response = client.post(
            "/api/generate", json={"model": name, "prompt": "x", "stream": False}
        )
        assert response.status_code in {400, 404}, name


def test_ca_s21_num_ctx_cap_reject_400(tmp_path_factory, monkeypatch) -> None:
    """C-103/CA-S21: num_ctx 10_000_000 -> 400 ANTES de asignar KV."""
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={
            "model": "qwen2-0_5b",
            "prompt": "x",
            "stream": False,
            "options": {"num_ctx": 10_000_000},
        },
    )
    assert response.status_code == 400
    assert "num_ctx" in response.json()["error"]
    assert engine.requests == []


def test_ca_s21_num_predict_cap_reject_400(tmp_path_factory, monkeypatch) -> None:
    client, engine, _store = make_client(monkeypatch)
    response = client.post(
        "/api/generate",
        json={
            "model": "qwen2-0_5b",
            "prompt": "x",
            "stream": False,
            "options": {"num_predict": 4097},
        },
    )
    assert response.status_code == 400
    assert "num_predict" in response.json()["error"]


def test_rate_limit_429_second_burst(
    tmp_path_factory, monkeypatch
) -> None:
    """C-106: token bucket per-peer -> 429/503 en exceso."""
    client, _engine, _store = make_client(
        monkeypatch,
        rate_limit_requests=1,
        rate_limit_window=3600.0,
    )
    first = client.post(
        "/api/generate", json={"model": "qwen2-0_5b", "prompt": "x", "stream": False}
    )
    assert first.status_code == 200
    second = client.post(
        "/api/generate", json={"model": "qwen2-0_5b", "prompt": "x", "stream": False}
    )
    assert second.status_code == 429
    assert "error" in second.json()


def test_first_byte_before_generation_completes(
    tmp_path_factory, monkeypatch
) -> None:
    """FR-105 (T-212): el PRIMER byte llega sin esperar al fin del stream."""
    import time as _time

    client, engine, _store = make_client(monkeypatch)
    gate = threading.Event()
    engine.block_after_first_token(gate)

    with client.stream(
        "POST",
        "/api/generate",
        json={"model": "qwen2-0_5b", "prompt": "Di hola", "stream": True},
    ) as response:
        assert response.status_code == 200
        stream = response.iter_bytes()
        first_bytes = next(stream)
        assert b'"done": false' in first_bytes
        # El worker sigue BLOQUEADO en la puerta (generacion NO terminada):
        _time.sleep(0.15)
        assert engine.stream_finished_count == 0
        gate.set()
        rest = b"".join(stream)
        assert b'"done": true' in rest
