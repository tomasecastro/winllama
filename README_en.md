# winllama

**Local LLM inference on Windows with Intel OpenVINO acceleration.**

**Status**: Investigation / Early Access (spec 001 – see `specs/001-investigacion-winllama/spec.md`)

---

## Architecture (ADR-0004)

Winllama runs as **two separate processes** (`ADR-0004`): a fast Rust CLI that only speaks HTTP, and a Python daemon that serves inference through Intel OpenVINO GenAI.

| Component | Language | Role | Key dependencies |
|---|---|---|---|
| **`winllama`** (CLI) | Rust (edition 2021) | Client: `run`, `serve`, `pull`, `list`, `rm`, `ps`, `doctor` | `clap`, `reqwest`, `tokio`, `serde` |
| **`winllama-engine`** (daemon) | Python 3.11+ | Serves local inference on `127.0.0.1:11434` with an Ollama/OpenAI-compatible API | `openvino-genai`, `fastapi`, `uvicorn` |

Why two processes:

- Intel officially supports OpenVINO GenAI in **Python**: `pip install openvino-genai` ships prebuilt wheels — **no MSVC, no Visual C++ Build Tools** required.
- The Rust CLI is a **pure HTTP client** (~10 ms startup); it never links OpenVINO.
- **Inference performance is identical**: both paths call the same C++ engine (`ov::genai`), so the language of the daemon does not change tokens/s.

> **End-user note**: you do not need Visual C++ Build Tools, and you do not need to install Python manually — the future installer embeds a Python runtime beside the daemon.

---

## Table of Contents
1. [Architecture (ADR-0004)](#architecture-adr-0004)
2.  [Installation](#installation)
3.  [Usage](#usage)
4.  [Port auto-detection](#port-auto-detection)
5.  [Glossary](#glossary)

---

## Installation
**From installer (planned)**: download the latest `WinllamaSetup.exe` from the releases page. The installer bundles the `winllama.exe` CLI, the Python daemon and its prebuilt wheels — no Visual C++ Build Tools and no manual Python installation required.

**Build from source**:

1. Rust CLI:
   ```bash
   cd src/winllama/cli && cargo build --release
   ```
2. Python engine:
   ```bash
   cd src/winllama/engine && pip install -e .
   ```
   Requires **Python 3.10+** (tested with 3.14; prebuilt `cp314` wheels exist for `openvino-genai`).

---

## Usage
All commands are identical to Ollama for a flat learning curve.

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

## Port auto-detection
- On startup, `winllama serve` checks if `127.0.0.1:11434` is occupied.
- **If free**: daemon binds to `11434` and logs: `Listening on http://127.0.0.1:11434`.
- **If occupied**: daemon automatically assigns the next available port (11435, 11436, …), logs the assignment, and writes the final port to `~/.winllama/config.json` for future sessions.
- The assigned port is always displayed in the console at launch.

```bash
# Example: force a specific port
winllama serve --port 11444
```

---

## Glossary (Technical terms, neutral)

| Term | Definition |
|------|------------|
| **Daemon** | Background process providing the local inference API. |
| **Port** | Network endpoint identifier (`127.0.0.1:11434`). |
| **Occupied** | Port already in use by another process. |
| **Transparency** | Auto‑assignment without user configuration. |
| **Model** | IA model in OpenVINO IR format (`.xml` + `.bin`). |
| **CLI** | Command‑line interface. |

---

## Supported Commands
| Command | Description |
|---------|-------------|
| `winllama pull <model>` | Download a model from Hugging Face (OpenVINO IR). |
| `winllama run <model>` | Start interactive chat. |
| `winllama serve` | Start the HTTP API daemon. |
| `winllama list` | List locally stored models. |
| `winllama rm <model>` | Remove a model. |
| `winllama ps` | Show loaded models. |
| `winllama doctor` | Diagnose environment, drivers, and device availability. |

---

## Notes
- The project is in active development under `specs/001-investigacion-winllama/`.
- For full specifications, see the `spec.md` and `research.md` in `specs/001-investigacion-winllama/`.
- Native Windows 11 x64: a Rust CLI binary plus a Python daemon. No WSL or Docker required.

---
*This project is under active development. See `specs/001-investigacion-winllama/` for architectural decisions and the current verdict.*
