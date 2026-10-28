"""Soporte `python -m winllama_engine` (el CLI Rust lanza este proceso)."""
from .main import cli

raise SystemExit(cli())
