"""Deteccion y seleccion de dispositivos Intel (CPU / GPU / NPU).

La deteccion usa la API real de OpenVINO Runtime expuesta por `openvino_genai`:
`Core().available_devices`. Si OpenVINO no esta instalado, se degrada a CPU
con un aviso visible (constitucion Art. VIII-W: degradacion con aviso).

Prioridad por defecto (requisito "usar todo el hardware al maximo"):
    GPU (Arc dedicada / iGPU) > NPU > CPU
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("winllama_engine.device")

# Orden de preferencia de mayor a menor rendimiento esperado.
DEVICE_PRIORITY: tuple[str, ...] = ("GPU", "NPU", "CPU")

# Etiqueta legible para cada familia de dispositivo de Intel.
DEVICE_LABELS: dict[str, str] = {
    "CPU": "CPU (Intel x86-64)",
    "GPU": "GPU Intel (Arc dedicada o iGPU)",
    "NPU": "NPU Intel (AI Boost)",
    "AUTO": "AUTO (seleccion automatica de OpenVINO)",
    "MULTI": "MULTI (reparto entre varios dispositivos)",
}


@dataclass(frozen=True)
class DeviceInfo:
    """Descripcion de un dispositivo disponible para inferencia."""

    name: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "label": self.label}


@dataclass
class DeviceReport:
    """Resultado de la deteccion de dispositivos."""

    available: list[DeviceInfo] = field(default_factory=list)
    selected: str = "CPU"
    degraded: bool = False
    warning: str | None = None
    openvino_available: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": [d.as_dict() for d in self.available],
            "selected": self.selected,
            "degraded": self.degraded,
            "warning": self.warning,
            "openvino_available": self.openvino_available,
            "error": self.error,
        }


def _normalize_device(raw: str) -> str:
    """Reduce nombres como 'GPU.0' o 'GPU.1' a su familia ('GPU')."""
    return raw.split(".", 1)[0].strip().upper()


def detect_devices() -> DeviceReport:
    """Detecta los dispositivos OpenVINO disponibles en la maquina.

    Nunca lanza excepcion: si OpenVINO no esta instalado o falla, devuelve un
    reporte degradado a CPU con `warning` y `error` poblados.
    """
    try:
        import openvino as ov  # import local: OpenVINO es dependencia del motor
    except Exception as exc:  # pragma: no cover - depende del entorno
        warning = (
            "OpenVINO no esta instalado o no se pudo importar; se degrada a CPU. "
            f"Detalle: {exc}"
        )
        logger.warning(warning)
        return DeviceReport(
            available=[DeviceInfo("CPU", DEVICE_LABELS["CPU"])],
            selected="CPU",
            degraded=True,
            warning=warning,
            openvino_available=False,
            error=str(exc),
        )

    try:
        core = ov.Core()
        raw_devices = list(core.available_devices)
    except Exception as exc:  # pragma: no cover - depende del entorno
        warning = f"No se pudieron enumerar los dispositivos OpenVINO; se degrada a CPU. Detalle: {exc}"
        logger.warning(warning)
        return DeviceReport(
            available=[DeviceInfo("CPU", DEVICE_LABELS["CPU"])],
            selected="CPU",
            degraded=True,
            warning=warning,
            openvino_available=True,
            error=str(exc),
        )

    # Deduplicar por familia (GPU.0 y GPU.1 -> GPU), conservando el orden reportado.
    seen: set[str] = set()
    available: list[DeviceInfo] = []
    for raw in raw_devices:
        family = _normalize_device(raw)
        if family in seen:
            continue
        seen.add(family)
        available.append(DeviceInfo(raw, DEVICE_LABELS.get(family, raw)))

    if not available:
        available = [DeviceInfo("CPU", DEVICE_LABELS["CPU"])]

    return DeviceReport(available=available, selected=pick_device(available))


def pick_device(available: list[DeviceInfo], preferred: str | None = None) -> str:
    """Elige el dispositivo de mayor prioridad disponible.

    Args:
        available: dispositivos detectados.
        preferred: dispositivo solicitado explicitamente por el usuario.
            Si esta disponible, se usa tal cual.

    Returns:
        El nombre del dispositivo elegido (familia, sin sufijo `.N`).
    """
    families = {_normalize_device(d.name) for d in available}

    if preferred:
        wanted = _normalize_device(preferred)
        if wanted in families or wanted in {"AUTO", "MULTI"}:
            return wanted
        # Dispositivo pedido no disponible: degradar con aviso.
        logger.warning(
            "Dispositivo '%s' solicitado pero no disponible %s; se degrada.",
            preferred,
            sorted(families),
        )

    for candidate in DEVICE_PRIORITY:
        if candidate in families:
            return candidate
    return "CPU"


def resolve_device(preferred: str | None = None) -> DeviceReport:
    """Detecta dispositivos y aplica la preferencia del usuario.

    Marca `degraded=True` y llena `warning` cuando el dispositivo preferido no
    esta disponible, para que la API lo muestre al cliente.
    """
    report = detect_devices()

    if not preferred:
        report.selected = pick_device(report.available)
        return report

    families = {_normalize_device(d.name) for d in report.available}
    wanted = _normalize_device(preferred)
    if wanted in families or wanted in {"AUTO", "MULTI"}:
        report.selected = wanted
        return report

    report.selected = pick_device(report.available)
    report.degraded = True
    report.warning = (
        f"Dispositivo '{preferred}' solicitado pero no disponible "
        f"(detectados: {sorted(families)}); usando '{report.selected}'."
    )
    logger.warning(report.warning)
    return report
