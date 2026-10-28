# winllama

**Inferencia local de LLM en Windows con aceleración Intel OpenVINO.**

**Status**: Investigación / Acceso temprano (spec 001 – ver `specs/001-investigacion-winllama/spec.md`)

---

## Arquitectura (ADR-0004)

Winllama se ejecuta como **dos procesos separados** (`ADR-0004`): un CLI rápido en Rust que solo habla HTTP, y un daemon en Python que sirve la inferencia a través de Intel OpenVINO GenAI.

| Componente | Lenguaje | Rol | Dependencias principales |
|---|---|---|---|
| **`winllama`** (CLI) | Rust (edición 2021) | Cliente: `run`, `serve`, `pull`, `list`, `rm`, `ps`, `doctor` | `clap`, `reqwest`, `tokio`, `serde` |
| **`winllama-engine`** (daemon) | Python 3.11+ | Sirve la inferencia local en `127.0.0.1:11434` con API compatible Ollama/OpenAI | `openvino-genai`, `fastapi`, `uvicorn` |

Por qué dos procesos:

- Intel soporta oficialmente OpenVINO GenAI en **Python**: `pip install openvino-genai` entrega wheels precompilados — **sin MSVC, sin Visual C++ Build Tools**.
- El CLI Rust es un **cliente HTTP puro** (~10 ms de arranque); nunca enlaza OpenVINO.
- **El rendimiento de inferencia es idéntico**: ambas rutas llaman al mismo motor C++ (`ov::genai`), así que el lenguaje del daemon no cambia los tokens/s.

> **Nota para el usuario final**: no necesita Visual C++ Build Tools, ni instalar Python manualmente — el instalador futuro embebe un runtime de Python junto con el daemon.

---

## Tabla de Contenidos
1. [Arquitectura (ADR-0004)](#arquitectura-adr-0004)
2.  [Instalación](#instalación)
3.  [Uso](#uso)
4.  [Auto-detección de puerto](#auto-detección-de-puerto)
5.  [Glosario](#glosario)

---

## Instalación
**Desde el instalador (planeado)**: descargue el último `WinllamaSetup.exe` desde la página de lanzamientos. El instalador empaqueta el binario `winllama.exe`, el daemon Python y sus wheels precompilados — sin Visual C++ Build Tools y sin instalar Python manualmente.

**Compilar desde el código fuente**:

1. CLI Rust:
   ```bash
   cd src/winllama/cli && cargo build --release
   ```
2. Engine Python:
   ```bash
   cd src/winllama/engine && pip install -e .
   ```
   Requiere **Python 3.10+** (probado con 3.14; existen wheels precompilados `cp314` de `openvino-genai`).

---

## Uso
Todos los comandos son idénticos a Ollama para una curva de aprendizaje plana.

```bash
winllama pull llama3.2
winllama run llama3.2
winllama serve
winllama list
winllama rm <model>
winllama ps
winllama doctor
```

---

## Auto-detección de puerto
- Al iniciar, `winllama serve` verifica si `127.0.0.1:11434` está ocupado.
- **Si está libre**: el daemon se une a `11434` y registra: `Listening on http://127.0.0.1:11434`.
- **Si está ocupado**: el daemon asigna automáticamente el siguiente puerto disponible (11435, 11436, …), registra el cambio y escribe el puerto final en `~/.winllama/config.json` para sesiones futuras.
- El puerto asignado siempre se muestra en la consola al inicio.

```bash
# Ejemplo: forzar un puerto específico
winllama serve --port 11444
```

---

## Glosario (Términos técnicos, neutro)

| Término | Definición |
|--------|------------|
| **Daemon** | Proceso en segundo plano que provee la API local. |
| **Puerto** | Identificador de red (`127.0.0.1:11434`). |
| **Ocupado** | Puerto ya usado por otro proceso. |
| **Transparencia** | Auto‑asignación sin configuración del usuario. |
| **Modelo** | IA modelo en formato OpenVINO IR (`.xml` + `.bin`). |
| **CLI** | Interfaz de línea de comandos. |

---

## Comandos Soportados
| Comando | Descripción |
|---------|-------------|
| `winllama pull <model>` | Descargar un modelo desde Hugging Face (formato OpenVINO IR). |
| `winllama run <model>` | Iniciar chat interactivo. |
| `winllama serve` | Iniciar el daemon HTTP. |
| `winllama list` | Listar modelos locales. |
| `winllama rm <model>` | Remover un modelo. |
| `winllama ps` | Mostrar modelos cargados. |
| `winllama doctor` | Diagnosticar entorno, drivers y dispositivos disponibles. |

---

## Notas
- El proyecto está en desarrollo activo bajo `specs/001-investigacion-winllama/`.
- Para especificaciones completas, consulte el `spec.md` y `research.md` en `specs/001-investigacion-winllama/`.
- Nativo Windows 11 x64: binario CLI Rust más daemon Python. No se requiere WSL ni Docker.

---
*Este proyecto está bajo desarrollo activo. Consulte `specs/001-investigacion-winllama/` para decisiones arquitectónicas y el veredicto actual.*
