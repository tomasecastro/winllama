"""Tests de `winllama_engine.main`: resolucion de puerto."""
from __future__ import annotations

import socket

from winllama_engine import main


def test_find_free_port_skips_occupied() -> None:
    # Reserva el 127.0.0.1:0 para obtener un puerto ocupado real.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        assert main.port_is_free("127.0.0.1", busy) is False
        chosen = main.find_free_port("127.0.0.1", start=busy)
        assert chosen != busy
        assert chosen >= busy


def test_find_free_port_returns_start_when_free() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        free = sock.getsockname()[1]
    # Tras cerrar el socket, el puerto deberia estar libre.
    assert main.find_free_port("127.0.0.1", start=free) == free


def test_build_parser_defaults() -> None:
    args = main.build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 11434
    assert args.device == "CPU"
