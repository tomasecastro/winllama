"""Generador interno compartido por las APIs (T-222).

Unica ruta de generacion para `/api/generate`, `/api/chat` (contrato de
compatibilidad tipo Ollama) y `/v1/chat/completions` (contrato tipo OpenAI):
el mapper de cada API solo traduce los eventos
genericos (`StreamEvent`) a su shape de contrato. Así la semantica de modelo,
keep_alive (derivada T-257), caps (C-103) y traversal (C-101) vive EN UN SOLO
lugar (plan.md Complexity Tracking; analogo a FR-113 para la compilacion).

Controles embebidos:
  - C-101: `validate_model_name` + `ensure_containment` (regex por segmento,
    sin `..`, negación explícita, containment por realpath).
  - C-103: caps al mapear options (delegados en `pipeline.GenOptions`).
  - C-106: cancel-on-disconnect (GeneratorExit del consumidor -> `cancel()`).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .models import ModelStore
from .pipeline import (
    EngineNotAvailableError,
    GenOptions,
    ModelNotFoundError,
)

logger = logging.getLogger("winllama_engine.core_generator")

DEFAULT_KEEP_ALIVE_S = 300.0  # "5m" (contratos/api.md §1)

# Primer caracter alfanumerico/underscore; resto alfanumerico/_/./- (C-101).
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$")


# ------------------------------------------------------------- StreamEvent
@dataclass
class StreamEvent:
    """Evento generico de generacion.

    kind:
      "token" -> `text` trae el fragmento de token.
      "end"   -> `metrics` (ns, nombres del contrato), `done_reason`, `context`.
      "error" -> `error` (mensaje) y `status` (HTTP code semantico).
    """

    kind: str
    text: str | None = None
    metrics: dict[str, Any] | None = None
    context: list[int] | None = None
    done_reason: str | None = None
    error: str | None = None
    status: int | None = None


class ModelNameError(ValueError):
    """Nombre de modelo invalido (control C-101: fail-closed)."""


def validate_model_name(name: object) -> str:
    """Valida el `<nombre>` de modelo local (control C-101).

    Reglas (regex POR SEGMENTO; la clase naive `^[\w\-.]+$` NO basta porque
    el '.' admite `..`):
      * string no vacio
      * rechaza separadores de ruta (`/`, `\\`), rutas absolutas, drive letters
        (`c:\\`), espacios y cualquier caracter fuera de [A-Za-z0-9_.-]
      * primer caracter alfanumerico o '_'
      * cada segmento separado por '.' es alfanumerico con '_' y '-', NO vacio
        => `..`, `a/../b`, `.` final quedan imposibilitados por construccion
    """
    if not isinstance(name, str) or not name or not name.strip():
        raise ModelNameError("El nombre de modelo es obligatorio.")
    if "/" in name or "\\" in name:
        # Los repos HF (`org/repo`) se validan aparte en el pull (Hito 4).
        raise ModelNameError(
            f"Nombre de modelo no valido (separador de ruta): {name!r}."
        )
    if ":" in name and len(name) > 1 and name[1] == ":":
        raise ModelNameError(f"Nombre de modelo no valido (unidad de disco): {name!r}.")
    if name.startswith("-") or name.endswith("."):
        raise ModelNameError(f"Nombre de modelo no valido (punto final): {name!r}.")
    if not _NAME_RE.fullmatch(name):
        raise ModelNameError(f"Nombre de modelo no valido (caracteres meta): {name!r}.")
    for segment in name.split("."):
        if not segment or not _SEGMENT_RE.fullmatch(segment):
            raise ModelNameError(
                f"Nombre de modelo no valido (segmento '{segment}' de {name!r})."
            )
    return name


def ensure_containment(resolved_path: Any, models_root: Any) -> None:
    """Segunda capa de C-101: containment por realpath.

    El realpath del modelo DEBE tener prefijo el realpath del cache; fail-closed
    entre endpoints que aceptan model (generate/chat/v1/delete/show).
    """
    real_path = str(Path(str(resolved_path)).resolve())
    real_root = str(Path(str(models_root)).resolve())
    inside = real_path == real_root or real_path.startswith(
        real_root.rstrip("\\/") + "\\" if "\\" in real_root else real_root.rstrip("/") + "/"
    )
    if not inside:
        raise ModelNameError(
            f"Path fuera del cache de modelos: {real_path} (C-101 fail-closed)."
        )


def parse_keep_alive(value: object) -> float:
    """Parsea `keep_alive` (shape del contrato, tipo Ollama) a segundos.

    Acepta `0`, `-1` (infinito), enteros/floats (segundos) y strings con
    unidad (`"10s"`, `"5m"`, `"2h"`). Frecuencia default: 5 minutos.
    """
    if value is None:
        return DEFAULT_KEEP_ALIVE_S
    if isinstance(value, bool):
        return DEFAULT_KEEP_ALIVE_S
    if isinstance(value, int):
        if value == -1:
            return float("inf")
        return float(value)
    if isinstance(value, float):
        if value == -1.0:
            return float("inf")
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_KEEP_ALIVE_S
        if text == "-1":
            return float("inf")
        multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(ms|s|m|h)?", text)
        if match:
            amount = float(match.group(1))
            unit = match.group(2) or "s"
            return amount * multipliers[unit]
        logger.warning("keep_alive ilegible %r; usando el default de %ss.", value, DEFAULT_KEEP_ALIVE_S)
        return DEFAULT_KEEP_ALIVE_S
    return DEFAULT_KEEP_ALIVE_S


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normaliza los mensajes al formato de la plantilla de chat."""
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        normalized.append({"role": role, "content": str(content)})
    return normalized


def _engine_load_duration(engine: Any, loaded_now: bool) -> int:
    if not loaded_now:
        return 0
    consume = getattr(engine, "consume_load_duration", None)
    if callable(consume):
        try:
            return int(consume())
        except Exception:  # pragma: no cover - defensivo
            return 0
    return 0


def iter_generation(
    store: ModelStore,
    engine_factory: Callable[[Any, str], Any],
    *,
    model: str,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    options: dict | None = None,
    device: str = "CPU",
    raw: bool = False,
    system: str | None = None,
    keep_alive: Any = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> Iterator[StreamEvent]:
    """Ejecuta una generacion y produce `StreamEvent`s genericos.

    Args:
        store: cache de modelos (resolucion + manifest).
        engine_factory: creador/resolvedor del motor (inyectable para tests;
            en produccion es `get_engine` de `pipeline.py`).
        model: nombre del modelo en el cache (validado C-101).
        prompt: prompt para `/api/generate` (o None en modo chat).
        messages: mensajes para `/api/chat` (o None en modo generate).
        options: dict de `options` del contrato.
        device: dispositivo pedido (ya normalizado por la capa API).
        registry: registro vivo de modelos cargados para `/api/ps` (T-224);
            mutado aqui (loaded_at/expires_at).
    """
    registry = registry if registry is not None else {}
    ka_seconds = parse_keep_alive(keep_alive)

    # ---------------------------------------------------------- validación
    try:
        validate_model_name(model)
    except ModelNameError as exc:
        yield StreamEvent(kind="error", error=str(exc), status=400)
        return

    try:
        options_obj = GenOptions.from_contract_options(options) if options is not None else None
    except Exception as exc:  # OptionCapError (C-103) y errores de cast
        message = str(exc) if isinstance(exc, Exception) else "options invalidas."
        yield StreamEvent(kind="error", error=f"options invalidas: {message}", status=400)
        return

    def _register_expiry() -> None:
        registry[model] = {"loaded_at": _now_iso()}
        if ka_seconds == float("inf"):
            registry[model]["expires_at"] = None
        elif ka_seconds <= 0:
            registry[model]["expires_at"] = _now_iso()
        else:
            registry[model]["expires_at"] = (
                (datetime.now(timezone.utc) + timedelta(seconds=ka_seconds)).isoformat()
            )

    # ------------------------------------------------- resolución y prompt
    try:
        start_ns = time.perf_counter_ns()
        model_path = store.resolve(model)
        ensure_containment(model_path, store.root)
        engine = engine_factory(model_path, device)

        if messages is not None:
            prompt_text = engine.apply_chat_template(_normalize_messages(messages))
        else:
            base_prompt = str(prompt or "")
            if system and not raw:
                base_prompt = f"{system}\n\n{base_prompt}\n"
            prompt_text = base_prompt

        # Contrato §1: prompt/mensajes vacios + sin keep_alive -> LOAD only.
        if not prompt_text.strip():
            loaded_now = bool(engine.load())
            load_ns = _engine_load_duration(engine, loaded_now)
            engine_load_started = time.perf_counter_ns()
            _register_expiry()
            logger.info("Load-only del modelo '%s' (prompt vacio).", model)
            yield StreamEvent(
                kind="end",  # end ('load')
                done_reason="load",
                metrics={
                    "total_duration": time.perf_counter_ns() - start_ns,
                    "load_duration": load_ns,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": time.perf_counter_ns() - engine_load_started,
                    "eval_count": 0,
                    "eval_duration": 0,
                },
                context=[],
                text="",
            )
            return
    except FileNotFoundError as exc:  # builtin y pipeline.ModelNotFoundError
        yield StreamEvent(kind="error", error=str(exc), status=404)
        return
    except EngineNotAvailableError as exc:
        yield StreamEvent(kind="error", error=str(exc), status=503)
        return

    # ------------------------------------------------------- generación
    engine_stream = engine.stream(prompt_text, options_obj)
    try:
        token_count = 0
        first_token_ns: int | None = None
        started_ns = time.perf_counter_ns()
        context: list[int] = []

        if hasattr(engine_stream, "result"):  # LLMEngine.GenStream (motor real)
            try:
                for token in engine_stream:
                    token_count += 1
                    first_token_ns = first_token_ns or time.perf_counter_ns()
                    yield StreamEvent(kind="token", text=token)
                result = engine_stream.result()
                stats = result.stats
                context = list(result.context)
            finally:
                # cancel-on-disconnect (C-106): el consumidor aborta -> STOP
                cancel = getattr(engine_stream, "cancel", None)
                if callable(cancel):
                    cancel()
        else:  # doble de tests: generador plano, stats aproximadas
            for token in engine_stream:
                token_count += 1
                first_token_ns = first_token_ns or time.perf_counter_ns()
                yield StreamEvent(kind="token", text=token)

        # ---------------------------------------------------- keep_alive
        done_reason = "stop"
        if ka_seconds == 0:
            try:
                engine.unload()
            except Exception:  # pragma: no cover - defensivo
                logger.warning("unload de '%s' fallo.", model, exc_info=True)
            done_reason = "unload"
            registry.pop(model, None)
        else:
            _register_expiry()

        if hasattr(engine_stream, "result"):
            metrics = stats.as_contract_metrics()
        else:
            now_ns = time.perf_counter_ns()
            t_first = first_token_ns or now_ns
            metrics = {
                "total_duration": now_ns - started_ns,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": t_first - started_ns,
                "eval_count": token_count,
                "eval_duration": now_ns - t_first,
            }
        yield StreamEvent(
            kind="end",
            metrics=metrics,
            context=context,
            done_reason=done_reason,
            text="",
        )
    except FileNotFoundError as exc:
        yield StreamEvent(kind="error", error=str(exc), status=404)
    except EngineNotAvailableError as exc:
        yield StreamEvent(kind="error", error=str(exc), status=503)
    except TimeoutError as exc:
        # write-timeout del pipeline (C-106): 504 es el mejor code semantico.
        yield StreamEvent(kind="error", error=str(exc), status=504)
    except Exception as exc:  # pragma: no cover - error del motor
        logger.exception("Fallo la generacion")
        yield StreamEvent(kind="error", error=str(exc), status=500)
