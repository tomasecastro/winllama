"""Winllama engine: daemon de inferencia OpenVINO GenAI (ADR-0004).

Este paquete expone un daemon HTTP local (127.0.0.1:11434) con un contrato
compatible con la API de Ollama (usada como referencia de compatibilidad; el
producto es Winllama). El motor real (`openvino_genai`) solo se importa desde
`winllama_engine.pipeline`, de modo que el resto del daemon (API, deteccion de
dispositivos, cache de modelos) puede importarse y testearse sin tener OpenVINO
instalado.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
