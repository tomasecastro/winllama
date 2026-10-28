"""Fixtures compartidas de la suite del engine (T-202).

Reemplaza la duplicacion de fakes de Hito 1 por un modelo falso robusto que NO
requiere GPU ni red. Los tests viejos (`test_api.py`, `test_pipeline.py`)
conservan sus fakes inline; este conftest es la base de los tests de contrato
de la 002 (`test_contracts_api.py`, `test_contracts_v1.py`) y unitarios
nuevos.

El modelo falso (`FakeGenEngine`) reproduce la interfaz que el
`core_generator` consume: `apply_chat_template`, `generate`, `stream`, y permite
controlar tokens por request, KEEP-ALIVE observado y bloqueo del stream para
afirmar el first-byte (flush) sin esperar al final de la generacion.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from winllama_engine import api

# Host que TestClient envia por defecto ("testserver"); se acepta solo en tests.
TEST_HOST = "testserver"


class FakeGenEngine:
    """Doble del `LLMEngine` consumido por la API.

    Interfaz replicada de `winllama_engine.pipeline.LLMEngine`:
      - apply_chat_template(messages) -> prompt
      - generate(prompt, options) -> texto completo
      - stream(prompt, options) -> iterador de tokens
      - loaded / load() / unload() para el estado del modelo
      - cancel() para abortar la generacion en curso (cancel-on-disconnect)
    """

    def __init__(self, text: str = "hola mundo", tokens: list[str] | None = None) -> None:
        self._chunks = tokens if tokens is not None else text.split()
        self._chunks = [c + " " for c in self._chunks]
        self._gate: threading.Event | None = None
        self._cancelled = False
        self.requests: list[dict[str, Any]] = []
        self.unloaded = False
        self.stream_finished_count = 0

    # --------------------------------------------------------- comportamiento
    def block_after_first_token(self, gate: threading.Event) -> None:
        """El proximo stream SE DETIENE tras emitir el primer token hasta `gate`."""
        self._gate = gate

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def loaded(self) -> bool:
        return not self.unloaded

    def load(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return " | ".join(m.get("content", "") for m in messages)

    def generate(self, prompt: str, options: object | None = None) -> str:
        self.requests.append({"mode": "generate", "prompt": prompt, "options": options})
        return "".join(self._chunks)

    def stream(self, prompt: str, options: object | None = None) -> Iterator[str]:
        self.requests.append({"mode": "stream", "prompt": prompt, "options": options})
        self._cancelled = False
        for i, chunk in enumerate(self._chunks):
            yield chunk
            if self._gate is not None and i == 0:
                # Bloquea DESPUES de emitir el primer token (first-byte tests).
                gate = self._gate
                self._gate = None
                gate.wait(timeout=10)
        self.stream_finished_count += 1


def make_client(
    monkeypatch: pytest.MonkeyPatch,
    engine: FakeGenEngine | None = None,
    model_name: str = "qwen2-0_5b",
    **app_kwargs: Any,
) -> tuple[TestClient, FakeGenEngine, Any]:
    """TestClient con un modelo IR falso en el cache y motor falso parcheado.

    Returns:
        (client, engine_falso, store) para afirmar contra el motor y el manifest.
    """
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="winllama-tests-"))
    model_dir = tmp / model_name
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    engine = engine or FakeGenEngine()

    def _fake_get_engine(model_dir_arg: object, device: str) -> FakeGenEngine:
        return engine

    monkeypatch.setattr(api, "get_engine", _fake_get_engine)
    app = api.create_app(
        models_dir=str(tmp), extra_allowed_hosts={TEST_HOST}, **app_kwargs
    )
    return TestClient(app), engine, app.state.model_store


def fake_metrics_factory() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Metricas ns sinteticas coherentes para afirmar shapes del contratos."""
    base = {
        "total_duration": 8493852375,
        "load_duration": 6589624375,
        "prompt_eval_count": 14,
        "prompt_eval_duration": 119039000,
        "eval_count": 110,
        "eval_duration": 1779061000,
    }

    def _merged(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return {**base, **(overrides or {})}

    return _merged
