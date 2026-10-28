"""API compatible OpenAI (`/v1/*`) — T-225 (spec 002, FR-104).

Montada por `api.create_app` (que inyecta store, engine factory y state)
para que comparta EXACTAMENTE el mismo generador (`core_generator`) y controles
(C-101/C-103/C-106, guard de Origin/Host del middleware da la app padre).

El contrato `/v1/*` sigue la forma de la referencia OpenAI (documentada en
`contracts/api.md` §3) para que los clientes existentes funcionen sin cambios.
Shapes (`contracts/api.md` §3 y la referencia OpenAI):
  - POST /v1/chat/completions no-stream: {id, object:"chat.completion",
    created, model, choices[{index, message{role, content}, finish_reason}],
    usage{prompt_tokens, completion_tokens, total_tokens}}.
  - stream: SSE con `chat.completion.chunk` y cierre `data: [DONE]\n\n`.
  - GET /v1/models: {object:"list", data[{id, object:"model", created, owned_by}]}.

Errores: shape del contrato {"error": {"message": ...}} con HTTP 400/404/500.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Iterator

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from . import core_generator

logger = logging.getLogger("winllama_engine.api_v1")

SSE_MEDIA_TYPE = "text/event-stream"
DEFAULT_CREATED = 1700000000
OWNED_BY = "winllama-engine"


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _error_response(status: int, message: str) -> JSONResponse:
    """Errores con el shape del contrato {"error": {"message"..., "code"...}}."""
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "code": status}}
    )


def _translate_options(payload: dict[str, Any]) -> dict[str, Any]:
    """Traduce params `/v1` -> bloque `options` del contrato del core.

    Las keys estilo contrato (`num_predict`, `raw`...) SOLO se heredan si el
    cliente `/v1` no envio su equivalente explicito (max_completion_tokens gana
    a max_tokens por ser el moderno en la referencia OpenAI).
    """
    options = dict(payload.get("options") or {})
    if "temperature" in payload and isinstance(payload["temperature"], (int, float)):
        options["temperature"] = float(payload["temperature"])
    if "top_p" in payload and isinstance(payload["top_p"], (int, float)):
        options["top_p"] = float(payload["top_p"])
    if "max_completion_tokens" in payload and payload["max_completion_tokens"] is not None:
        options["num_predict"] = int(payload["max_completion_tokens"])
    elif "max_tokens" in payload and payload["max_tokens"] is not None:
        options["num_predict"] = int(payload["max_tokens"])
    if "stop" in payload and payload["stop"] is not None:
        stop = payload["stop"]
        options["stop"] = stop if isinstance(stop, list) else [str(stop)]
    if payload.get("presence_penalty") is not None:
        options["presence_penalty"] = float(payload["presence_penalty"])
    if payload.get("frequency_penalty") is not None:
        options["frequency_penalty"] = float(payload["frequency_penalty"])
    if payload.get("seed") is not None:
        options["seed"] = int(payload["seed"])
    # `n`, `logprobs`, `logit_bias`, `tools`: fuera de US1 (respuestas n);
    # si el cliente las envia, se IGNORAN EXPLICITAMENTE con log (no error).
    for extra in ("n", "logprobs", "logit_bias", "tools"):
        if extra in payload and payload[extra] not in (None, 1, [], {}):
            logger.warning(
                "Param del contrato %r fuera del alcance de US1: se ignora "
                "(NO es descarte silencioso: warning visiblemente logeado).", extra
            )
    return options


def _epoch_from_iso(value: Any) -> int:
    try:
        return int(datetime_from_iso(value))
    except Exception:
        return int(time.time())


def datetime_from_iso(value: Any) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(str(value)).timestamp()


def build_v1_router(
    store: Any,
    engine_factory: Callable[[Any, str], Any],
    state: dict[str, Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    def v1_models() -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        for entry in store.list_models():
            data.append(
                {
                    "id": entry.name,
                    "object": "model",
                    "created": _epoch_from_iso(entry.added_at),
                    "owned_by": OWNED_BY,
                }
            )
        return {"object": "list", "data": data}

    @router.post("/v1/chat/completions")
    def v1_chat_completions(payload: dict = Body(default_factory=dict)) -> Any:
        messages = payload.get("messages") or []
        if not messages:
            return _error_response(400, "El campo 'messages' es obligatorio.")
        model_name = payload.get("model") or state["default_model"]
        if not model_name:
            return _error_response(
                400,
                "No se especifico 'model' y no hay modelo por defecto.",
            )
        stream = bool(payload.get("stream", False))
        events = core_generator.iter_generation(
            store,
            engine_factory,
            model=str(model_name),
            messages=messages,
            options=_translate_options(payload),
            device=payload.get("device") or state["device"],
            registry=state["loaded"],
        )
        event_iter = iter(events)
        first = next(event_iter, None)
        if first is None:
            return _error_response(500, "Sin eventos de generacion.")
        if first.kind == "error":
            return _error_response(first.status or 500, first.error or "error")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created = int(time.time())
        model_str = str(model_name)

        if not stream:
            text_parts: list[str] = []
            if first.kind == "token":
                text_parts.append(first.text or "")
            for ev in event_iter:
                if ev.kind == "token":
                    text_parts.append(ev.text or "")
                elif ev.kind == "end":
                    metrics = ev.metrics or {}
                    prompt_tokens = int(metrics.get("prompt_eval_count", 0) or 0)
                    completion_tokens = int(metrics.get("eval_count", 0) or 0)
                    return {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_str,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "".join(text_parts),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                else:  # pragma: no cover - defensivo
                    return _error_response(ev.status or 500, ev.error or "error")
            return _error_response(500, "Generacion sin final.")  # pragma: no cover

        def sse_stream() -> Iterator[str]:
            def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
                return {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_str,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish_reason}
                    ],
                }

            first_emitted = False
            if first.kind == "token":
                first_emitted = True
                yield _sse(
                    json.dumps(
                        chunk({"role": "assistant", "content": first.text}),
                        ensure_ascii=False,
                    )
                )
            elif first.kind == "end":
                yield _sse(json.dumps(chunk({}, finish_reason="stop")))
                yield _sse("[DONE]")
                return
            sent_including_usage: dict[str, int] | None = None
            for ev in event_iter:
                if ev.kind == "token":
                    yield _sse(
                        json.dumps(
                            chunk({"content": ev.text}), ensure_ascii=False
                        )
                    )
                elif ev.kind == "end":
                    metrics = ev.metrics or {}
                    sent_including_usage = {
                        "prompt_tokens": int(metrics.get("prompt_eval_count", 0) or 0),
                        "completion_tokens": int(metrics.get("eval_count", 0) or 0),
                    }
                    yield _sse(
                        json.dumps(chunk({}, finish_reason="stop"), ensure_ascii=False)
                    )
                else:
                    error_chunk = {"error": {"message": ev.error or "error"}}
                    yield _sse(json.dumps(error_chunk, ensure_ascii=False))
            if not first_emitted and not sent_including_usage:  # pragma: no cover
                yield _sse(json.dumps(chunk({}, finish_reason="stop")))
            yield _sse("[DONE]")

        return StreamingResponse(sse_stream(), media_type=SSE_MEDIA_TYPE)

    return router
