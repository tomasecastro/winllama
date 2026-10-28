"""API HTTP del daemon Winllama (spec 002, Hito 2 — US1).

Expone el contrato de compatibilidad con Ollama (mismos shapes de respuesta);
Ollama es la referencia del contrato, no el producto.

Endpoints (Franja 1, `contracts/api.md`):
    POST /api/generate   -> FR-101/FR-102: streaming NDJSON o JSON final (timing ns).
    POST /api/chat       -> FR-103: multi-turno con streaming NDJSON.
    GET  /api/ps         -> FR-106: modelos cargados (size/digest/details/expires_at).
    POST /api/show       -> FR-123: info del manifest local.
    DELETE /api/delete   -> FR-123: elimina modelo del cache.
    GET  /api/tags, /api/version, /api/doctor -> utilidades (Hito 1).

Controles de seguridad embebidos (threat-model §6):
  - C-101: validacion de `<model>` (regex por segmento SIN '..') + containment
    por realpath en TODOS los endpoints que aceptan model (core_generator).
  - C-102: middleware que rechaza Host/Origin no locales con 403 (SR-01).
  - C-103: caps de num_ctx/num_predict/stop y body (~2 MB) con 400/413.
  - C-106: rate-limit per-peer (token bucket), coalescing de carga
    (pipeline.load) y cancel-on-disconnect del stream.

La generacion va por `core_generator.iter_generation` (T-222): eventos genericos
que esta capa traduce a NDJSON (`streaming.ndjson_framer`). El motor se obtiene
con `get_engine` inyectado como factory para que los tests puedan parcharlo
(compat Hito 1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import core_generator
from .core_generator import StreamEvent, ensure_containment, validate_model_name
from .device import resolve_device
from .models import ModelStore
from .pipeline import EngineNotAvailableError, get_engine

logger = logging.getLogger("winllama_engine.api")

# Hosts locales permitidos (mitiga DNS rebinding, SR-01).
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

DEFAULT_MODEL_SETTING = "default_model"
MAX_BODY_BYTES = 2 * 1024 * 1024  # C-103: cap de tamano de body (~2 MB)

# C-106: rate-limit per-peer (token bucket por ventana deslizante).
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_S = 1.0


class _RateLimiter:
    """Rate-limit per-peer simple: ventana deslizante (control C-106)."""

    def __init__(self, max_requests: int, window_s: float) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, peer: str) -> bool:
        now = time.monotonic()
        window = self._hits[peer]
        while window and now - window[0] > self.window_s:
            window.popleft()
        if len(window) >= self.max_requests:
            return False
        window.append(now)
        return True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _line(obj: dict[str, Any]) -> bytes:
    """Serializa un objeto a una línea NDJSON."""
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _manifest_digest(entry_payload: dict[str, Any]) -> str:
    """`digest` para /api/ps: sha256 del manifest del modelo (12 hex).

    Ollama muestra "sha256:<12 hex>"; la definicion completa del digest
    (pin del ledger, T-256) confirma esto como base. ERROR de divergencia
    documentado en bitacora si T-256 fija otro mecanismo.
    """
    canonical = json.dumps(entry_payload, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()[:12]


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normaliza los mensajes al formato esperado por la plantilla de chat."""
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            # Contenido multimodal: extraer solo el texto.
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        normalized.append({"role": role, "content": str(content)})
    return normalized


def create_app(
    models_dir: str | None = None,
    default_model: str | None = None,
    default_device: str = "CPU",
    extra_allowed_hosts: set[str] | None = None,
    rate_limit_requests: int = RATE_LIMIT_REQUESTS,
    rate_limit_window: float = RATE_LIMIT_WINDOW_S,
) -> FastAPI:
    """Construye la aplicacion FastAPI del daemon.

    Args:
        models_dir: directorio de cache de modelos (default `~/.winllama/models`).
        default_model: modelo a usar cuando el request no especifica uno.
        default_device: dispositivo por defecto (CPU).
        extra_allowed_hosts: hosts adicionales aceptados por el guard (solo tests).
        rate_limit_requests: requests por peer permitidos (control C-106).
        rate_limit_window: ventana del bucket en segundos (C-106).
    """
    app = FastAPI(title="winllama-engine", version="0.2.0")
    store = ModelStore(models_dir)
    allowed_hosts = ALLOWED_HOSTS | (extra_allowed_hosts or set())
    limiter = _RateLimiter(rate_limit_requests, rate_limit_window)
    state: dict[str, Any] = {
        DEFAULT_MODEL_SETTING: default_model,
        "device": default_device,
        # Registro de modelos cargados para /api/ps (T-224); lo muta
        # core_generator con loaded_at/expires_at (T-257).
        "loaded": {},
    }

    def _engine_factory(model_path: Any, device: str) -> Any:
        # Resuelve `get_engine` EN TIEMPO DE LLAMADA (module global): los tests
        # de Hito 1 lo parchean con monkeypatch y siguen funcionando.
        return get_engine(model_path, device)

    # -------------------------------------------------------- middleware C-102
    @app.middleware("http")
    async def _local_only_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = (request.headers.get("host") or "").split(":")[0]
        if host and host not in allowed_hosts:
            logger.warning("Peticion rechazada: Host no permitido '%s'", host)
            return JSONResponse(
                status_code=403,
                content={"error": f"Host '{host}' no permitido (daemon solo local)."},
            )
        origin = request.headers.get("origin")
        if origin and origin not in {"null"}:
            # Los clientes HTTP (CLI Rust, curl) no envian Origin; un navegador si.
            logger.warning("Peticion rechazada: Origin '%s' no permitido", origin)
            return JSONResponse(
                status_code=403,
                content={"error": "Peticiones con Origin no permitidas (CSRF)."},
            )
        # C-103: cap de tamano de body (413 controlado, nunca OOM).
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Body excede el maximo de ~2 MB del daemon."},
            )
        # C-106: rate-limit per-peer con 429 (Art. VI: limites con rechazo).
        peer = request.client.host if request.client else "desconocido"
        if not limiter.allow(peer):
            logger.warning("Rate-limit excedido para peer '%s' (429).", peer)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit excedido; reintenta mas tarde."},
            )
        return await call_next(request)

    # ------------------------------------------------------------ helpers API
    def _resolve_model_name(payload: dict[str, Any]) -> tuple[str | None, JSONResponse | None]:
        """Devuelve (model, error_response). 400 sin model y sin default."""
        model_name = payload.get("model") or state.get(DEFAULT_MODEL_SETTING)
        if not model_name:
            available = [e.name for e in store.list_models()]
            return None, JSONResponse(
                status_code=400,
                content={
                    "error": "No se especifico 'model' y no hay modelo por defecto.",
                    "available_models": available,
                },
            )
        return str(model_name), None

    def _events_for(payload: dict[str, Any], chat_mode: bool) -> Iterator[StreamEvent]:
        """Construye el iterador de eventos del core (helpers de chat/validity)."""
        keep_alive = payload.get("keep_alive")
        return core_generator.iter_generation(
            store,
            _engine_factory,
            model=payload["model"],
            prompt=payload.get("prompt"),
            messages=payload.get("messages") if chat_mode else None,
            options=payload.get("options"),
            device=payload.get("device") or state["device"],
            raw=bool(payload.get("raw", False)),
            system=payload.get("system"),
            keep_alive=keep_alive,
            registry=state["loaded"],
        )

    def _final_payload(model: str, end_event: StreamEvent, chat_mode: bool) -> dict[str, Any]:
        """Shape final del contrato con las métricas ns Exactas (FR-102)."""
        metrics = dict(end_event.metrics or {})
        if chat_mode:
            payload: dict[str, Any] = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": end_event.done_reason or "stop",
            }
            payload.update(metrics)
            return payload
        payload = {
            "model": model,
            "created_at": _now_iso(),
            "response": "",
            "done": True,
            "done_reason": end_event.done_reason or "stop",
            "context": list(end_event.context or []),
        }
        payload.update(metrics)
        return payload

    # ------------------------------------------------------ /api/generate §1
    @app.post("/api/generate")
    def api_generate(payload: dict = Body(default_factory=dict)) -> Any:
        """FR-101/FR-102/FR-107/FR-108 sobre core_generator."""
        model_name, error = _resolve_model_name(payload)
        if error is not None:
            return error

        events = _events_for({**payload, "model": model_name}, chat_mode=False)
        event_iter = iter(events)
        first = next(event_iter, None)
        if first is None:
            return JSONResponse(status_code=500, content={"error": "Sin eventos."})
        if first.kind == "error":
            return JSONResponse(status_code=first.status or 500, content={"error": first.error})

        if not bool(payload.get("stream", True)):
            text_parts: list[str] = []
            if first.kind == "token":
                text_parts.append(first.text or "")
            elif first.kind == "end":
                # Load-only (contract §1): final inmediato sin tokens.
                final = _final_payload(str(model_name), first, chat_mode=False)
                final["response"] = ""
                return final
            for ev in event_iter:
                if ev.kind == "token":
                    text_parts.append(ev.text or "")
                elif ev.kind == "end":
                    final = _final_payload(str(model_name), ev, chat_mode=False)
                    final["response"] = "".join(text_parts)
                    return final
                else:  # pragma: no cover - defensivo
                    return JSONResponse(status_code=ev.status or 500, content={"error": ev.error})
            return JSONResponse(status_code=500, content={"error": "Generacion sin final."})  # pragma: no cover

        created = _now_iso()

        def ndjson_generate() -> Iterator[bytes]:
            if first.kind == "token":
                yield _line(
                    {
                        "model": model_name,
                        "created_at": created,
                        "response": first.text,
                        "done": False,
                    }
                )
            elif first.kind == "end":
                yield _line(_final_payload(str(model_name), first, chat_mode=False))
                return
            for ev in event_iter:
                if ev.kind == "token":
                    yield _line(
                        {
                            "model": model_name,
                            "created_at": created,
                            "response": ev.text,
                            "done": False,
                        }
                    )
                elif ev.kind == "end":
                    yield _line(_final_payload(str(model_name), ev, chat_mode=False))
                else:
                    yield _line({"error": ev.error or "error", "done": True})

        try:
            return StreamingResponse(
                ndjson_generate(), media_type="application/x-ndjson"
            )
        except Exception:  # pragma: no cover - defensivo
            logger.exception("Fallo al armar el stream")
            return JSONResponse(status_code=500, content={"error": "Error del servidor."})

    # ----------------------------------------------------------- /api/chat §2
    @app.post("/api/chat")
    def chat(payload: dict = Body(default_factory=dict)) -> Any:
        """Chat compatible con el contrato (misma maquina general que /api/generate)."""
        messages = payload.get("messages") or []
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": "El campo 'messages' es obligatorio y no puede estar vacio."},
            )
        model_name, error = _resolve_model_name(payload)
        if error is not None:
            return error

        events = _events_for({**payload, "model": model_name}, chat_mode=True)
        event_iter = iter(events)
        first = next(event_iter, None)
        if first is None:
            return JSONResponse(status_code=500, content={"error": "Sin eventos."})
        if first.kind == "error":
            return JSONResponse(status_code=first.status or 500, content={"error": first.error})

        if not bool(payload.get("stream", True)):
            text_parts: list[str] = []
            if first.kind == "token":
                text_parts.append(first.text or "")
            for ev in event_iter:
                if ev.kind == "token":
                    text_parts.append(ev.text or "")
                elif ev.kind == "end":
                    final = _final_payload(str(model_name), ev, chat_mode=True)
                    final["message"]["content"] = "".join(text_parts)
                    return final
                else:  # pragma: no cover
                    return JSONResponse(status_code=ev.status or 500, content={"error": ev.error})
            return JSONResponse(status_code=500, content={"error": "Generacion sin final."})  # pragma: no cover

        created = _now_iso()

        def ndjson_chat() -> Iterator[bytes]:
            if first.kind == "token":
                yield _line(
                    {
                        "model": model_name,
                        "created_at": created,
                        "message": {"role": "assistant", "content": first.text},
                        "done": False,
                    }
                )
            elif first.kind == "end":
                yield _line(_final_payload(str(model_name), first, chat_mode=True))
                return
            for ev in event_iter:
                if ev.kind == "token":
                    yield _line(
                        {
                            "model": model_name,
                            "created_at": created,
                            "message": {"role": "assistant", "content": ev.text},
                            "done": False,
                        }
                    )
                elif ev.kind == "end":
                    yield _line(_final_payload(str(model_name), ev, chat_mode=True))
                else:
                    yield _line({"error": ev.error or "error", "done": True})

        return StreamingResponse(ndjson_chat(), media_type="application/x-ndjson")

    # --------------------------------------------------------------- /api/ps
    @app.get("/api/ps")
    def list_ps() -> dict[str, Any]:
        """FR-106: modelos cargados AHORA (loaded/expires_at por keep_alive)."""
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        expired: list[str] = []
        for name, info in list(state["loaded"].items()):
            expires_raw = info.get("expires_at")
            if expires_raw:
                try:
                    expires = datetime.fromisoformat(str(expires_raw))
                except ValueError:  # pragma: no cover - manifest tapado
                    expires = None
                if expires and expires < now:
                    expired.append(name)
                    continue
            try:
                entry = store.show(name)
            except FileNotFoundError:
                expired.append(name)
                continue
            raw_manifest = store._read_manifest().get(name, {})
            rows.append(
                {
                    "name": entry.name,
                    "model": entry.name,
                    "size": entry.size_bytes,
                    "digest": _manifest_digest(raw_manifest or {"name": entry.name}),
                    "details": {
                        "format": "openvino-ir",
                        "family": (entry.source or "unknown").split("/")[-1].split("-")[0]
                        if entry.source
                        else "unknown",
                        "parameter_size": entry.parameters or "",
                        "quantization_level": entry.quantization or "",
                    },
                    "expires_at": expires_raw,
                    "size_vram": 0,
                }
            )
        for name in expired:
            state["loaded"].pop(name, None)
        return {"models": rows}

    # ------------------------------------------------------------- /api/show
    @app.post("/api/show")
    def show(payload: dict = Body(default_factory=dict)) -> Any:
        """FR-123 (§7): info del manifest local de un modelo del cache."""
        model = payload.get("model") or state.get(DEFAULT_MODEL_SETTING)
        if not model:
            return JSONResponse(status_code=400, content={"error": "Falta 'model'."})
        try:
            name = validate_model_name(str(model))
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            model_path = store.resolve(name)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        try:
            ensure_containment(model_path, store.root)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        entry = store.show(name)
        raw = store._read_manifest().get(name, {})
        source = entry.source or raw.get("source") or "local"
        family = raw.get("family") or (
            str(source).split("/")[-1].split("-")[0].lower() if "/" in str(source) else "unknown"
        )
        return {
            "details": {
                "format": "openvino-ir",
                "family": family,
                "parameter_size": entry.parameters or "",
                "quantization_level": entry.quantization or "",
            },
            "model_info": {
                "general.architecture": raw.get("general.architecture", "unknown"),
                "source": source,
                "recommended_device": raw.get("recommended_device", "auto"),
                "mirrors": raw.get("mirrors", []),
            },
            "capabilities": ["completion"],
        }

    # --------------------------------------------------------- DELETE delete
    @app.delete("/api/delete")
    def delete(payload: dict = Body(default_factory=dict)) -> Any:
        """FR-123 (§8): elimina la carpeta del cache y refresca /api/tags."""
        model = payload.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"error": "Falta 'model'."})
        try:
            name = validate_model_name(str(model))
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            model_path = store.resolve(name)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        try:
            ensure_containment(model_path, store.root)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        # Fail-closed (Art. VI): containment verificado ANTES de borrar.
        store.remove(name)
        # Soltar el engine cargado (libera memoria y evita re-load fantasma).
        from .pipeline import drop_engines_for

        drop_engines_for(model_path)
        state["loaded"].pop(name, None)
        logger.info("Modelo '%s' eliminado del cache.", name)
        return {"status": "success"}

    # -------------------------------------------------------------- /api/tags
    @app.get("/api/tags")
    def list_tags() -> dict[str, Any]:
        """Lista los modelos locales. Siempre 200, incluso sin modelos."""
        models = []
        for entry in store.list_models():
            models.append(
                {
                    "name": entry.name,
                    "model": entry.name,
                    "modified_at": entry.added_at,
                    "size": entry.size_bytes,
                    "digest": "",
                    "details": {
                        "format": "openvino-ir",
                        "family": entry.source or "unknown",
                        "parameter_size": entry.parameters or "",
                        "quantization_level": entry.quantization or "",
                    },
                }
            )
        return {"models": models}

    @app.get("/api/version")
    def version() -> dict[str, str]:
        from . import __version__

        return {"version": __version__}

    # ------------------------------------------------------------ /api/doctor
    @app.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        """Diagnostico: devices, cache y disponibilidad del motor."""
        report = resolve_device(state["device"])
        engine_ok = True
        engine_error = None
        try:
            import openvino_genai  # noqa: F401
        except Exception as exc:  # pragma: no cover - depende del entorno
            engine_ok = False
            engine_error = str(exc)
        return {
            "openvino_genai_available": engine_ok,
            "openvino_genai_error": engine_error,
            "device": report.as_dict(),
            "models_dir": str(store.root),
            "models": [e.name for e in store.list_models()],
            "default_model": state.get(DEFAULT_MODEL_SETTING),
        }

    # ------------------------------------------------- montar router /v1
    from .api_v1 import build_v1_router

    app.include_router(build_v1_router(store, _engine_factory, state))

    # Referencias para tests/diagnostico (core generator conserva el lock).
    app.state.model_store = store
    app.state.engine_factory = _engine_factory
    app.state.api_state = state

    return app


def _test_payload_size_cap_disabled() -> None:  # pragma: no cover - removible
    """El cap de body (C-103) esta probado por el guard; no hay handler extra."""
    return None
