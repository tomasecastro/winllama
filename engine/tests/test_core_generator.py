"""Tests de `winllama_engine.core_generator` (T-222).

Generador interno compartido por `/api/*` (contrato tipo Ollama) y `/v1/*`
(contrato tipo OpenAI):
valida el nombre de modelo (control C-101), resuelve el cache, construye el
prompt (chat-template o raw), ejecuta la generacion y produce eventos genéricos
(token / end / error) aplicando la semantica `keep_alive` (derivada T-257).
"""

from __future__ import annotations

import pytest

from winllama_engine.core_generator import (
    StreamEvent,
    iter_generation,
    parse_keep_alive,
    validate_model_name,
)
from winllama_engine.models import ModelStore
from winllama_engine.pipeline import OptionCapError
from tests.conftest import FakeGenEngine


def _make_store(monkeypatch: pytest.MonkeyPatch, model_name: str = "qwen2-0_5b"):
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="winllama-core-tests-"))
    model_dir = tmp / model_name
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    return ModelStore(str(tmp))


def test_validate_model_name_accepts_normal_names() -> None:
    for name in ("llama-3.2-1B-int4-ov", "qwen2_0_5b", "Mi_Modelo.1"):
        validate_model_name(name)  # no levanta


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "..",
        "../escape",
        "a/../b",
        "sub/colacion",
        r"..\\windows",
        "espacio en el medio",
        "a:b",
        ".oculto",
        "-guion-principio",
        "trailing.",
        "a;b",
    ],
)
def test_validate_model_name_rejects_traversal_and_meta_chars(bad: str) -> None:
    """C-101: regex POR SEGMENTO, sin '..' y sin escapes de directorio."""
    with pytest.raises(ValueError):
        validate_model_name(bad)


def test_iter_generation_streams_tokens_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(monkeypatch)
    engine = FakeGenEngine(text="hola mundo")
    registry: dict = {}
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="qwen2-0_5b",
            prompt="Di hola",
            options={"temperature": 0},
            device="CPU",
            registry=registry,
        )
    )
    tokens = [e for e in events if e.kind == "token"]
    ends = [e for e in events if e.kind == "end"]
    errors = [e for e in events if e.kind == "error"]
    assert errors == []
    assert [t.text for t in tokens] == ["hola ", "mundo "]
    assert len(ends) == 1
    end = ends[0]
    assert end.done_reason == "stop"
    metrics = end.metrics
    assert set(metrics) == {
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    }
    assert all(isinstance(v, int) for v in metrics.values())


def test_iter_generation_chat_uses_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(monkeypatch)
    engine = FakeGenEngine(text="hola mundo")
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="qwen2-0_5b",
            messages=[{"role": "user", "content": "que?"}],
            options=None,
            device="CPU",
            registry={},
        )
    )
    assert any(e.kind == "token" for e in events)
    prompt_used = engine.requests[-1]["prompt"]
    assert prompt_used == "que?"  # el fake junta contents con espacios


def test_iter_generation_unknown_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)
    engine = FakeGenEngine()
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="nope",
            prompt="hola",
            options=None,
            device="CPU",
            registry={},
        )
    )
    ends = [e for e in events if e.kind == "error"]
    assert len(ends) == 1
    assert "no esta en el cache" in ends[0].error


def test_iter_generation_name_traversal_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-101: un `model` tipo '../x' error a NIVEL DE EVENTO (el mapper 400)."""
    store = _make_store(monkeypatch)
    engine = FakeGenEngine()
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="../escape",
            prompt="hola",
            options=None,
            device="CPU",
            registry={},
        )
    )
    errors = [e for e in events if e.kind == "error"]
    assert len(errors) == 1
    assert "no valido" in errors[0].error
    assert engine.requests == []


def test_iter_generation_empty_prompt_loads_only_load_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrato §1: prompt vacio + sin keep_alive -> load (done_reason:'load')."""
    store = _make_store(monkeypatch)
    engine = FakeGenEngine(text="hola mundo")
    registry: dict = {}
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="qwen2-0_5b",
            prompt="",
            options=None,
            device="CPU",
            registry=registry,
        )
    )
    assert [e.kind for e in events] == ["end"]
    end = events[0]
    assert end.done_reason == "load"
    assert end.metrics["load_duration"] >= 0
    assert end.metrics["eval_count"] == 0
    assert engine.requests == []  # la carga NO genera
    assert "qwen2-0_5b" in registry


def test_iter_generation_keep_alive_zero_unloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrato §1: keep_alive:0 -> unload (done_reason:'unload')."""
    store = _make_store(monkeypatch)
    engine = FakeGenEngine(text="hola mundo")
    registry: dict = {}
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="qwen2-0_5b",
            prompt="Di algo",
            options=None,
            device="CPU",
            keep_alive="0",
            registry=registry,
        )
    )
    ends = [e for e in events if e.kind == "end"]
    assert len(ends) == 1
    assert ends[0].done_reason == "unload"
    assert engine.unloaded is True
    assert "qwen2-0_5b" not in registry


def test_iter_generation_options_cap_bubbles(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-103: el cap se valida aqui y sale como evento error con code 400."""
    store = _make_store(monkeypatch)
    engine = FakeGenEngine()
    events = list(
        iter_generation(
            store,
            lambda p, d: engine,
            model="qwen2-0_5b",
            prompt="hola",
            options={"num_predict": 9999},
            device="CPU",
            registry={},
        )
    )
    errors = [e for e in events if e.kind == "error"]
    assert len(errors) == 1
    assert errors[0].status == 400
    assert "num_predict" in errors[0].error
    assert engine.requests == []


def test_parse_keep_alive_durations() -> None:
    assert parse_keep_alive("0") == 0.0
    assert parse_keep_alive(0) == 0.0
    assert parse_keep_alive("10s") == 10.0
    assert parse_keep_alive("5m") == 300.0
    assert parse_keep_alive("2h") == 7200.0
    assert parse_keep_alive(90) == 90.0
    assert parse_keep_alive(None) == 300.0  # default 5m
    assert parse_keep_alive(-1) == float("inf")  # infinito (no expira Monos)
    assert parse_keep_alive("garbage") == 300.0  # fallback default


def test_stream_event_shape() -> None:
    ev = StreamEvent(kind="token", text="hola")
    assert ev.kind == "token"
    assert ev.text == "hola"
