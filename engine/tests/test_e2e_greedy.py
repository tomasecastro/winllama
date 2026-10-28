"""T-226: e2e de coherencia greedy (FR-110 / ADR-0005 / ISS-01).

3/3 requests con `temperature=0` y SIN `repeat_penalty` deben producir salida
identica y coherente (determinista). Corre SOLO si el modelo de referencia de
la E2E esta en el cache local (`%USERPROFILE%\\.winllama\\models/Llama-3.2-1B-
Instruct-int4-ov`); sin modelo la suite NO falla (skip documentado).

La evidencia real (curl + timing) vive en la bitacora 002 (T-228/T-252).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from winllama_engine.pipeline import GenOptions, LLMEngine

MODEL_NAME = "Llama-3.2-1B-Instruct-int4-ov"
MODEL_DIR = Path.home() / ".winllama" / "models" / MODEL_NAME

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.is_dir(), reason=f"Modelo {MODEL_NAME} no esta en el cache local"
)


def _greedy_options() -> GenOptions:
    return GenOptions.from_contract_options(
        {"temperature": 0, "num_predict": 48}
    )


def test_greedy_is_deterministic_3_of_3() -> None:
    """3/3: misma salida EXACTA (greedy sin penalizacion y sin semilla)."""
    engine = LLMEngine(MODEL_DIR, "CPU")
    outputs = [
        str(engine.generate("Why is the sky blue?", _greedy_options()))
        for _ in range(3)
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0].strip()) > 10


def test_greedy_output_is_coherent_not_degraded() -> None:
    """ISS-01: sin repeat_penalty la salida greedy no degrada a eco/repeticion.

    Criterio objetivo: la respuesta contiene la palabra clave de la pregunta
    (scattering/atmosphere/rayleigh/light) y NO es una repeticion degenerada
    del prompt ('Blue?' x n).
    """
    engine = LLMEngine(MODEL_DIR, "CPU")
    text = str(engine.generate("Why is the sky blue?", _greedy_options())).lower()
    keywords = ("scatter", "atmosphere", "rayleigh", "light", "wavelength", "sun")
    assert any(k in text for k in keywords), f"Salida greedy degradada: {text!r}"
    # No puede empezar por la literal pregunta repetida (eco degenerado).
    assert not text.startswith("why is the sky blue?why")
