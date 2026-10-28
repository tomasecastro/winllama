"""Envoltura de `openvino_genai.LLMPipeline`.

Este es el UNICO modulo del daemon que importa `openvino_genai` (el plan de
spec 001 lo exige: aislar el motor en `pipeline.py` permite cambiar de motor
sin tocar la API). La API usada es la real de OpenVINO GenAI:

    pipe = openvino_genai.LLMPipeline(model_dir, device)
    pipe.generate(prompt, config, streamer)          # config: GenerationConfig
    pipe.get_tokenizer()                             # ov_genai.Tokenizer

Cambio 002 (Hito 2):
  - T-210: mapping completo de options del contrato -> GenerationConfig
    (data-model.md §4) con default greedy `repetition_penalty=1.0` (ADR-0005,
    cierra ISS-01) y caps del control C-103 (`OptionCapError`).
  - T-211: contadores de timing propios en nanosegundos (`perf_counter_ns`):
    `eval_count`, `eval_duration_ns`, `ttft_ns`, `load_duration_ns` (FR-102).
  - T-212 (ISS-02): el streaming YA NO se acumula: los tokens se ceden a medida
    que llegan del streamer (worker-thread + cola), habilitando
    cancel-on-disconnect (control C-106) y first-byte temprano.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Iterator

try:  # numpy viene con openvino; se usa para aplanar los ids del tokenizador.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy siempre presente con openvino
    np = None  # type: ignore

logger = logging.getLogger("winllama_engine.pipeline")

# ------------------------------------------------------------------- caps C-103
MAX_NUM_PREDICT = 4096
MAX_NUM_CTX = 8192
MAX_STOP_ENTRIES = 10
MAX_STOP_CHARS = 256

# Timeout de escritura por chunk (control C-106 / T-262): si el modelo no
# produce un token dentro de esta ventana, se aborta la generacion.
WRITE_TIMEOUT_S = 30.0
# Espera maxima SIN TOKENS desde el inicio (incluye carga fria del modelo,
# que en esta maquina se midio hasta ~35s: bitacora 002).
STARTUP_TIMEOUT_S = 300.0


class OptionCapError(ValueError):
    """Se lanza cuando una option excede los caps del control C-103."""


class EngineNotAvailableError(RuntimeError):
    """Se lanza cuando `openvino_genai` no esta instalado."""


class ModelNotFoundError(FileNotFoundError):
    """Se lanza cuando el directorio del modelo IR no existe o esta incompleto."""


@dataclass
class GenOptions:
    """Opciones de generacion traducidas a `openvino_genai.GenerationConfig`.

    Defaults (ADR-0005): greedy sin penalizacion — `temperature=0.0`,
    `repetition_penalty=1.0`. Los knobs de sampling SOLO se aplican al config
    cuando el cliente muestra intencion de muestrear (`is_sampling`).
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    num_ctx: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    num_thread: int | None = None

    @property
    def is_sampling(self) -> bool:
        """True si hay intencion real de muestrear (ADR-0005 punto 2)."""
        return self.do_sample and self.temperature > 0.0

    @classmethod
    def from_contract_options(cls, options: dict | None) -> "GenOptions":
        """Construye desde el bloque `options` del contrato Winllama.

        El bloque `options` es el contrato de compatibilidad (mismo shape que el
        de Ollama, usado como referencia). Semantica: temperature=0 (o ausente)
        => greedy; temperature>0 => muestreo. `num_predict=-1` (ilimitado en el
        contrato) se clampa a MAX_NUM_PREDICT. Los caps C-103 rechazan con
        `OptionCapError` antes de tocar el motor.
        """
        options = options or {}
        defaults = cls()

        temperature = options.get(
            "temperature", defaults.temperature
        )
        do_sample = bool(options.get("do_sample", temperature != 0))

        num_predict_raw = options.get("num_predict", defaults.max_new_tokens)
        try:
            num_predict = int(num_predict_raw)
        except (TypeError, ValueError) as exc:
            raise OptionCapError(f"num_predict no es un entero: {num_predict_raw!r}") from exc
        if num_predict == -1:
            num_predict = MAX_NUM_PREDICT  # 'ilimitado' del contrato -> cap del daemon
        if num_predict < 1 or num_predict > MAX_NUM_PREDICT:
            raise OptionCapError(
                f"num_predict={num_predict} fuera de rango (1..{MAX_NUM_PREDICT}); "
                "use -1 para el maximo del daemon."
            )

        stop_raw = list(options.get("stop", defaults.stop) or [])
        if len(stop_raw) > MAX_STOP_ENTRIES:
            raise OptionCapError(
                f"stop excede el maximo de {MAX_STOP_ENTRIES} entradas "
                f"({len(stop_raw)} recibidas)."
            )
        for entry in stop_raw:
            if len(str(entry)) > MAX_STOP_CHARS:
                raise OptionCapError(
                    f"stop[{escape_repr(str(entry)[:16])}...] excede "
                    f"{MAX_STOP_CHARS} caracteres por entrada."
                )
        stop = [str(entry) for entry in stop_raw]

        num_ctx = _opt_int(options, "num_ctx", None)
        if num_ctx is not None and (num_ctx < 1 or num_ctx > MAX_NUM_CTX):
            raise OptionCapError(
                f"num_ctx={num_ctx} excede el cap {MAX_NUM_CTX} del daemon "
                "(control C-103: evita OOM de KV-cache)."
            )

        return cls(
            max_new_tokens=num_predict,
            temperature=float(temperature),
            top_p=float(options.get("top_p", defaults.top_p)),
            top_k=int(options.get("top_k", defaults.top_k)),
            do_sample=do_sample,
            repetition_penalty=float(
                options.get("repeat_penalty", defaults.repetition_penalty)
            ),
            stop=stop,
            seed=_opt_int(options, "seed", None),
            num_ctx=num_ctx,
            min_p=_opt_float(options, "min_p", None),
            presence_penalty=_opt_float(options, "presence_penalty", None),
            frequency_penalty=_opt_float(options, "frequency_penalty", None),
            num_thread=_opt_int(options, "num_thread", None),
        )


def escape_repr(x: str) -> str:
    """Repr corto para mensajes de error (mantiene la firma legible)."""
    return repr(x)


def _opt_int(options: dict, key: str, default: int | None) -> int | None:
    raw = options.get(key, default)
    if raw is None:
        return None
    return int(raw)


def _opt_float(options: dict, key: str, default: float | None) -> float | None:
    raw = options.get(key, default)
    if raw is None:
        return None
    return float(raw)


def _require_openvino_genai():  # pragma: no cover - depende del entorno
    try:
        import openvino_genai as ov_genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise EngineNotAvailableError(
            "El paquete 'openvino-genai' no esta instalado. "
            "Instalalo con: pip install openvino-genai"
        ) from exc
    return ov_genai


def is_model_dir(path: Path) -> bool:
    """Un modelo IR de OpenVINO es valido si contiene un `.xml` (grafo)."""
    if not path.is_dir():
        return False
    return any(path.glob("*.xml"))


@dataclass
class GenStats:
    """Contadores de timing en nanosegundos de una generacion (FR-102, P-002).

    TODO-none: ver ISSUES abajo — `prompt_eval_duration` es aproximacion
    (t_primer_token - carga), la separacion exacta prompt/eval no es expuesta
    por la API Python de openvino_genai 2026.4.1.0 (ver bitacora 002/ISS-07).
    """

    total_duration_ns: int = 0
    load_duration_ns: int = 0
    ttft_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    done_reason: str = "stop"

    def as_contract_metrics(self) -> dict[str, int | str]:
        """Nombres EXACTOS del contrato Winllama (contratos/api.md §1).

        Los campos conservan los nombres del contrato de compatibilidad
        (`total_duration`, `prompt_eval_count`, ...): NO se renombran porque los
        clientes existentes dependen de ellos.
        """
        return {
            "total_duration": self.total_duration_ns,
            "load_duration": self.load_duration_ns,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration_ns,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration_ns,
        }


class GenResult(str):
    """Texto completo de la generacion + `GenStats` adjuntos.

    Es un `str` para mantener compatibilidad con los consumidores de Hito 1
    (`"{text}"`) y a la vez expone `result.stats` con los contadores ns.
    """

    def __new__(cls, text: str, stats: GenStats | None = None, context: list[int] | None = None):
        obj = super().__new__(cls, text)
        obj.stats = stats or GenStats()
        obj.context = context if context is not None else []
        return obj

    stats: GenStats
    context: list[int]


_TOKEN = "tok"
_DONE = "done"


class _TokenQueueMessage:
    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload: object = None) -> None:
        self.kind = kind
        self.payload = payload


class GenStream:
    """Stream incremental de tokens con contadores ns y cancelacion.

    La generacion corre en un worker-thread (el `LLMPipeline` no es seguro
    para uso concurrente, y `generate` es bloqueante): el callback-streamer
    empuja cada token a una cola; `__iter__` los consume a medida que
    llegan (flush por token, ISS-02) sin esperar al final.

    Cancel-on-disconnect (C-106): si el consumidor abandona la iteracion
    (GeneratorExit) o llama `cancel()`, el streamer devuelve
    `StreamingStatus.STOP` en el siguiente token y OpenVINO aborta.
    Write-timeout (C-106): un token que tarda > `WRITE_TIMEOUT_S` aborta.
    """

    def __init__(self, engine: "LLMEngine", prompt: str, options: GenOptions | None = None):
        self._engine = engine
        self._prompt = prompt
        self._options = options or GenOptions()
        self._q: "queue.Queue[_TokenQueueMessage]" = queue.Queue()
        self._t_request_start = time.perf_counter_ns()
        self._t_first_token: int | None = None
        self._t_last_token: int | None = None
        self._eval_count = 0
        self._stats: GenStats | None = None
        self._abort = Event()
        self._text_parts: list[str] = []
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._started = False
        self._final_context: list[int] = []
        self._finished = False

    # ------------------------------------------------------------ lifecycle
    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ov-genai-generate")
        self._thread.start()

    def cancel(self) -> None:
        self._abort.set()

    def _callback(self, subword: object):
        now = time.perf_counter_ns()
        if self._t_first_token is None:
            self._t_first_token = now
        self._t_last_token = now
        self._eval_count += 1
        text = str(subword)
        self._text_parts.append(text)
        self._q.put(_TokenQueueMessage(_TOKEN, text))
        if self._abort.is_set() and self._engine._ov_genai is not None:
            # StreamingStatus.STOP e InferenceEnd ambos abortan; STOP es el claro.
            return self._engine._ov_genai.StreamingStatus.STOP
        running = getattr(getattr(self._engine, "_ov_genai", None), "StreamingStatus", None)
        if running is not None:
            return running.RUNNING
        # Doble de tests (FakeStreamingStatus) también expone RUNNING tras el abort check.
        return "RUNNING"

    def _run(self) -> None:
        engine = self._engine
        stats = GenStats()
        done_reason = "stop"
        context: list[int] = []
        try:
            load_started_at = time.perf_counter_ns()
            loaded_now = engine.load()
            load_ns = engine.consume_load_duration() if loaded_now else 0
            stats.load_duration_ns = load_ns

            config = engine._build_config(self._options)
            t_preflight_done = load_started_at  # tiempos relativos al arranque del request
            if loaded_now:
                t_preflight_done = load_started_at + load_ns
            prompt_count, context = engine._prompt_token_info(self._prompt)
            stats.prompt_eval_count = prompt_count

            with engine._lock:
                try:
                    engine._pipe.generate(
                        self._prompt, config, self._callback
                    )
                except Exception as exc:  # error del motor underlying
                    self._error = exc
            if self._t_last_token is not None:
                stats.eval_count = self._eval_count
                stats.eval_duration_ns = self._t_last_token - self._t_first_token  # type: ignore[operator]
            if self._t_first_token is not None:
                # t_primer_token - inicio - carga ≈ prompt eval (+ scheduling).
                stats.ttft_ns = self._t_first_token - self._t_request_start
                stats.prompt_eval_duration_ns = max(
                    0, self._t_first_token - t_preflight_done
                )
            stats.total_duration_ns = time.perf_counter_ns() - self._t_request_start
        except Exception as exc:  # pragma: no cover - propagado al iterador
            self._error = exc
        finally:
            # Los kw de done_reason se derivan en core_generator (keep_alive);
            # aqui solo se reporta el final (stats).
            stats.done_reason = done_reason
            self._stats = stats
            self._q.put(_TokenQueueMessage(_DONE, {"stats": stats, "context": context}))

    # ------------------------------------------------------------ iteration
    def __iter__(self) -> Iterator[str]:
        self._start()
        return self._iterate()

    def _iterate(self) -> Iterator[str]:
        try:
            got_first_token = False
            while True:
                # C-106: el write-timeout es el GAP ENTRE TOKENS (stall), no el
                # arranque: la carga fria puede tardar mucho mas que 30s antes
                # del primer token, y abortar en load aborta requests sanos.
                #   - antes del primer token: STARTUP_TIMEOUT_S
                #   - entre tokens: WRITE_TIMEOUT_S
                window = WRITE_TIMEOUT_S if got_first_token else STARTUP_TIMEOUT_S
                try:
                    msg = self._q.get(timeout=window)
                except queue.Empty as exc:
                    # Write-timeout (C-106): abortar y propagar.
                    self._abort.set()
                    raise TimeoutError(
                        "Write timeout: el modelo no produjo un token en "
                        f"{window:.0f}s "
                        + ("entre tokens" if got_first_token else "antes del primer token")
                        + "; se aborta la generacion."
                    ) from exc
                if msg.kind == _TOKEN:
                    got_first_token = True
                    yield str(msg.payload)
                else:
                    payload = msg.payload if isinstance(msg.payload, dict) else {}
                    self._stats = payload.get("stats") or self._stats
                    self._final_context = list(payload.get("context") or [])
                    self._finished = True
                    return
        finally:
            self._abort.set()  # si el consumidor abandona, abortar el worker

    def result(self) -> GenResult:
        """Espera el final y devuelve `GenResult` (texto completo + stats).

        Si el stream YA se consumió (caso de `core_generator`: se iteró
        explicitamente), se reutiliza el estado interno; en el caso contrario
        se itera hasta la finalización. No re-itera el generator (eso bloquea
        en DONE que nunca llegará: hecho tras el bug del daemon, T-223).
        """
        if not self._finished:
            for _ in self:
                pass
        return GenResult(
            "".join(self._text_parts), self._stats or GenStats(), self._final_context
        )

    def _context_ids(self) -> list[int]:
        # `context` lo envia el worker via cola; recuperado en `_iterate`/result.
        return getattr(self, "_final_context", [])


class LLMEngine:
    """Envoltura con estado de un `LLMPipeline`.

    Mantiene un unico pipeline cargado y serializa el acceso con un lock,
    porque `LLMPipeline` no es seguro para uso concurrente desde hilos Python.
    """

    def __init__(self, model_dir: str | Path, device: str = "CPU") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._pipe = None
        self._ov_genai = None
        self._lock = Lock()
        self._load_lock = Lock()  # coalescing de carga (C-106): un solo load
        self._loaded = False
        self._load_duration_ns = 0  # duracion del ULTIMO load (T-211)

    # ------------------------------------------------------------------ carga
    def load(self) -> bool:
        """Carga el modelo IR en memoria y compila para el dispositivo.

        Returns:
            True si ESTE call ejecuta el load (False si el modelo ya estaba).
        El lock `_load_lock` hace coalescing: requests concurrentes comparten
        la unica compilacion (control C-106).
        """
        with self._load_lock:
            if self._loaded:
                return False
            if not is_model_dir(self.model_dir):
                raise ModelNotFoundError(
                    f"No hay un modelo IR valido en '{self.model_dir}' "
                    "(se esperaba al menos un archivo .xml)."
                )
            ov_genai = _require_openvino_genai()
            logger.info(
                "Cargando modelo '%s' en dispositivo '%s'...", self.model_dir, self.device
            )
            self._ov_genai = ov_genai
            t0 = time.perf_counter_ns()
            # API real: LLMPipeline(model_path, device)
            self._pipe = ov_genai.LLMPipeline(str(self.model_dir), self.device)
            self._load_duration_ns = time.perf_counter_ns() - t0
            self._loaded = True
            # Warmup (ISS-08, T-226): la PRIMERA generate() de un pipeline
            # recien cargado produce una salida distinta (aunque coherente) a
            # la de las siguientes — verificado con openvino_genai 2026.4.1.0
            # en crudo (probe en bitacora 002). Un generate de 1 token en
            # greedy fuerza el path de decoding estable para TODAS las
            # peticiones del daemon; el coste se paga dentro de load_duration.
            try:
                warm_cfg = ov_genai.GenerationConfig()
                warm_cfg.max_new_tokens = 1
                warm_cfg.temperature = 0
                self._pipe.generate("warmup", warm_cfg)
            except Exception as exc:  # pragma: no cover - no bloqueante
                logger.debug("Warmup de 1 token fallo (no bloquea): %s", exc)
            logger.info("Modelo cargado correctamente.")
            return True

    def consume_load_duration(self) -> int:
        """Devuelve y resetea la duracion del ultimo load (uso por request)."""
        value = self._load_duration_ns
        return value

    def unload(self) -> None:
        """Libera el pipeline (permite recargar con otro modelo)."""
        with self._lock:
            self._pipe = None
            self._loaded = False
            self._ov_genai = None
        logger.info("Modelo descargado del cache de motores.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ----------------------------------------------------------- generacion
    def _prompt_token_info(self, prompt: str) -> tuple[int, list[int]]:
        """Cuenta tokens del prompt y devuelve el context (ids) si es posible.

        Si el tokenizador no esta disponible en el modelo, devuelve (0, []).
        La API es `tokenizer.encode(prompt) -> TokenizedInputs` cuyo
        `.input_ids` es un ov.Tensor (`.data` = numpy /*mismo acceso en 2026.4*/).
        """
        try:
            tokenizer = self._pipe.get_tokenizer()
            encoding = tokenizer.encode(prompt)
            ids = encoding.input_ids
            data = getattr(ids, "data", None)
            if data is None and np is not None:
                data = np.asarray(ids, dtype="int64")  # fallback defensivo
            flat = [int(x) for x in np.asarray(data).ravel().tolist()]
            return len(flat), flat
        except Exception as exc:  # pragma: no cover - depende del modelo
            logger.debug("Tokenizador no disponible para prompt-eval-count: %s", exc)
            return 0, []

    def _build_config(self, options: GenOptions):
        ov_genai = self._ov_genai
        if ov_genai is None:  # pragma: no cover - defensive
            raise EngineNotAvailableError("El engine no esta cargado.")
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = options.max_new_tokens

        # ADR-0005: en greedy NO se aplican knobs de sampling.
        if options.is_sampling:
            config.do_sample = True
            config.temperature = options.temperature
            config.top_p = options.top_p
            config.top_k = options.top_k
            if options.min_p is not None:
                config.min_p = options.min_p
            if options.presence_penalty is not None:
                config.presence_penalty = options.presence_penalty
            if options.frequency_penalty is not None:
                config.frequency_penalty = options.frequency_penalty
        else:
            config.do_sample = False

        config.repetition_penalty = options.repetition_penalty  # 1.0 neutral default
        if options.stop:
            config.stop_strings = set(options.stop)
        if options.seed is not None:
            config.rng_seed = options.seed
        if options.num_ctx is not None:
            # num_ctx (ventana de contexto) -> max_length (total prompt+gen).
            config.max_length = options.num_ctx
        try:
            validate = getattr(config, "validate", None)
            if callable(validate):
                validate()
        except Exception as exc:
            raise ValueError(f"GenerationConfig invalido: {exc}") from exc
        return config

    def generate(self, prompt: str, options: GenOptions | None = None) -> GenResult:
        """Genera texto completo para `prompt` y devuelve `GenResult`.

        Mantiene compatibilidad `str` (los consumidores de Hito 1 usan el
        texto): `text_assert in engine.generate(...)` sigue funcionando.
        """
        return self.stream_full(prompt, options)

    def stream_full(self, prompt: str, options: GenOptions | None = None) -> GenResult:
        """Generacion completa con `GenResult` (metrics ns incluidos)."""
        stream = self.stream(prompt, options)
        return stream.result()

    def stream(self, prompt: str, options: GenOptions | None = None) -> GenStream:
        """Genera texto emitiendo cada fragmento de token a medida que se produce.

        Usa el `streamer` de OpenVINO GenAI (API real): una funcion que recibe
        cada subword y devuelve `ov_genai.StreamingStatus.RUNNING` para seguir.
        El stream NO bufferea (ISS-02 cerrado): tokens en directo + estats ns.
        """
        return GenStream(self, prompt, options)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Construye un prompt con la plantilla de chat del modelo (si existe).

        Si el modelo no expone `apply_chat_template`, se degrada a un formato
        de transcripcion explicito para no fallar la peticion.
        """
        self.load()
        try:
            tokenizer = self._pipe.get_tokenizer()
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception as exc:  # pragma: no cover - depende del modelo
            logger.debug("apply_chat_template no disponible (%s); usando fallback.", exc)
            return _fallback_chat_prompt(messages)


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Formato de transcripcion simple cuando el modelo no trae plantilla."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ------------------------------------------------------------------ cache
_ENGINES: dict[tuple[str, str], LLMEngine] = {}
_CACHE_LOCK = Lock()


def get_engine(model_dir: str | Path, device: str = "CPU") -> LLMEngine:
    """Devuelve un `LLMEngine` cacheado por (modelo, dispositivo)."""
    key = (str(Path(model_dir).resolve()), device)
    with _CACHE_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = LLMEngine(key[0], device)
            _ENGINES[key] = engine
        return engine


def clear_cache() -> None:
    """Libera todos los pipelines cacheados (usado en tests y al cambiar modelo)."""
    with _CACHE_LOCK:
        for engine in _ENGINES.values():
            engine.unload()
        _ENGINES.clear()


def drop_engines_for(model_dir: str | Path) -> int:
    """Descarta (y descarga) todos los engines de UN modelo concreto.

    Usado por DELETE /api/delete: seno, el manifest re-creado con el mismo path
    reutilizaria un engine obsoleto. Devuelve cuantos se soltaron.
    """
    prefix = str(Path(model_dir).resolve())
    dropped = 0
    with _CACHE_LOCK:
        for key in list(_ENGINES):
            if key[0] == prefix:
                _ENGINES.pop(key).unload()
                dropped += 1
    return dropped


def generate_text(
    model_dir: str | Path,
    prompt: str,
    device: str = "CPU",
    options: GenOptions | None = None,
) -> str:
    """Helper de conveniencia: carga (o reutiliza) un engine y genera texto."""
    return str(get_engine(model_dir, device).generate(prompt, options))


def stream_text(
    model_dir: str | Path,
    prompt: str,
    device: str = "CPU",
    options: GenOptions | None = None,
) -> Iterator[str]:
    """Helper de conveniencia: streaming de tokens sobre un engine cacheado."""
    yield from get_engine(model_dir, device).stream(prompt, options)


StreamCallback = Callable[[str], object]
