"""Tests de `winllama_engine.streaming` (T-212): adaptadores NDJSON y SSE.

El contrato `/api/*` usa NDJSON (1 JSON por linea); el contrato `/v1/*` usa SSE
(`data: {..}\\n\\n` con cierre `data: [DONE]\\n\\n`). El framer debe ceder
cada marco a medida que la fuente lo produce (sin buffering de salida, ISS-02)
y con flush por token (FR-105).
"""

from __future__ import annotations

import json
import threading
from typing import Iterator

import pytest

from winllama_engine.streaming import ndjson_framer, sse_framer


def test_ndjson_framer_one_json_per_line() -> None:
    chunks = [{"a": 1}, {"b": 2}]
    lines = list(ndjson_framer(iter(chunks)))
    assert b"\n" in lines[0]
    body = b"".join(lines).decode("utf-8")
    parsed = [json.loads(ln) for ln in body.splitlines() if ln.strip()]
    assert parsed == [{"a": 1}, {"b": 2}]


def test_sse_framer_data_prefixes_and_done() -> None:
    chunks = [{"delta": "The"}, {"delta": " sky"}]
    out = list(sse_framer(iter(chunks)))
    assert out[0].startswith(b"data: ")
    assert out[0].endswith(b"\n\n")
    tail = b"".join(out[-2:])
    assert b"data: [DONE]\n\n" in tail
    payload = json.loads(out[0].removeprefix(b"data: ").rstrip(b"\n"))
    assert payload == {"delta": "The"}


def test_framers_yield_incrementally_no_full_buffer() -> None:
    """El framer cede el primer marco ANTES de que la fuente termine (flush)."""
    gate = threading.Event()
    source_times: list[str] = []

    def source() -> Iterator[dict]:
        yield {"n": 1}
        source_times.append("esperando-gate")
        gate.wait(timeout=5)
        yield {"n": 2}

    def consumersequence() -> Iterator[bytes]:  # bacteriana sola para el flujo
        for piece in ndjson_framer(source()):
            source_times.append("consumidor-recibio-1" if b'"n": 1' in piece else "consumidor-recibio-2")
            yield piece

    out = list(consumersequence())
    gate.set()
    # El consumidor recibio el primer marco mientras la fuente aun no producia el 2.
    assert source_times.index("consumidor-recibio-1") < source_times.index("esperando-gate")
    assert len(out) == 2


def test_ndjson_framer_handles_non_ascii() -> None:
    out = list(ndjson_framer(iter([{"response": "á\r\nla"}])))
    body = b"".join(out).decode("utf-8")
    # \r\n debe escaparse dentro del JSON para NO romper el framing NDJSON.
    assert len(body.splitlines()) == 1
    assert json.loads(body) == {"response": "á\r\nla"}


def test_sse_framer_empty_source_still_emits_done() -> None:
    out = list(sse_framer(iter([])))
    assert out == [b"data: [DONE]\n\n"]
