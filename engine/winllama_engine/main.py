"""Arranque del daemon: `python -m winllama_engine`.

Resuelve puerto (con auto-deteccion si 11434 esta ocupado), construye la app
FastAPI y levanta uvicorn. El CLI Rust (`winllama serve`) lanza exactamente este
modulo como subproceso y hace handshake contra `/api/version`.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

from .api import create_app
from .models import ModelStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("winllama_engine.main")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
PORT_SCAN_LIMIT = 64


def port_is_free(host: str, port: int) -> bool:
    """True si `host:port` puede ser reservado (nadie lo esta escuchando)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str, start: int = DEFAULT_PORT, limit: int = PORT_SCAN_LIMIT) -> int:
    """Devuelve el primer puerto libre en [start, start+limit).

    Raises:
        RuntimeError: si no encuentra puerto libre en el rango.
    """
    for candidate in range(start, start + limit):
        if port_is_free(host, candidate):
            return candidate
    raise RuntimeError(
        f"No se encontro puerto libre en {host} entre {start} y {start + limit - 1}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="winllama-engine",
        description="Daemon de inferencia Winllama (OpenVINO GenAI) — ADR-0004.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host de escucha (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Puerto (default 11434).")
    parser.add_argument(
        "--device", default="CPU", help="Dispositivo de inferencia: CPU/GPU/NPU/AUTO."
    )
    parser.add_argument("--model", default=None, help="Modelo por defecto en el cache.")
    parser.add_argument("--models-dir", default=None, help="Directorio de cache de modelos.")
    parser.add_argument(
        "--no-port-scan",
        action="store_true",
        help="Falla en vez de auto-asignar otro puerto si el pedido esta ocupado.",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI: parsea argumentos y arranca uvicorn."""
    args = build_parser().parse_args(argv)

    if not port_is_free(args.host, args.port):
        if args.no_port_scan:
            logger.error(
                "El puerto %s:%s esta ocupado y --no-port-scan esta activo.", args.host, args.port
            )
            return 1
        chosen = find_free_port(args.host, args.port)
        logger.warning(
            "Puerto %s:%s ocupado; se auto-asigna %s.", args.host, args.port, chosen
        )
        args.port = chosen

    app = create_app(
        models_dir=args.models_dir,
        default_model=args.model,
        default_device=args.device,
    )

    import uvicorn

    logger.info("Listening on http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main() -> None:
    """Entry point para consola (usado por el script `winllama-engine`)."""
    raise SystemExit(cli())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
