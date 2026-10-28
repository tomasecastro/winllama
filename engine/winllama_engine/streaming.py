"""Adaptadores de framing para streaming (T-212).

Fijan el formato de línea del streaming sin conocer la API concreta:

  - `ndjson_framer`: el contrato `/api/*` emite **NDJSON** (un JSON por linea).
  - `sse_framer`:    el contrato `/v1/*` emite **SSE** (`data: {..}\\n\\n` y cierre
    `data: [DONE]\\n\\n`).

Ambos son perezosos: ceden cada marco a medida que la fuente lo produce
(sin buffering completo de la respuesta, ISS-02) para lograr flush por token
(FR-105 / CA-S23).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator


def _json_line(obj: dict[str, Any]) -> bytes:
    """Serializa `obj` a UNA linea NDJSON (el contenido nunca abre linea)."""
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def ndjson_framer(chunks: Iterable[dict[str, Any]]) -> Iterator[bytes]:
    """Envuelve un iterador de dicts a bytes NDJSON, marco por marco."""
    for chunk in chunks:
        yield _json_line(chunk)


def sse_framer(chunks: Iterable[dict[str, Any]]) -> Iterator[bytes]:
    """Envuelve un iterador de dicts a eventos SSE con cierre `data: [DONE]`."""
    for chunk in chunks:
        payload = json.dumps(chunk, ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
