"""Tests de `winllama_engine.device`: deteccion y seleccion de dispositivos."""

from __future__ import annotations

from winllama_engine.device import (
    DeviceInfo,
    detect_devices,
    pick_device,
    resolve_device,
)


def test_pick_device_prefers_gpu_over_cpu() -> None:
    available = [DeviceInfo("CPU", "cpu"), DeviceInfo("GPU.0", "gpu")]
    assert pick_device(available) == "GPU"


def test_pick_device_prefers_gpu_over_npu() -> None:
    available = [DeviceInfo("CPU", "cpu"), DeviceInfo("NPU", "npu"), DeviceInfo("GPU", "gpu")]
    assert pick_device(available) == "GPU"


def test_pick_device_falls_back_to_cpu() -> None:
    assert pick_device([DeviceInfo("CPU", "cpu")]) == "CPU"


def test_pick_device_honors_explicit_preference() -> None:
    available = [DeviceInfo("CPU", "cpu"), DeviceInfo("GPU", "gpu")]
    assert pick_device(available, preferred="CPU") == "CPU"


def test_pick_device_degrades_when_preference_missing() -> None:
    available = [DeviceInfo("CPU", "cpu")]
    assert pick_device(available, preferred="NPU") == "CPU"


def test_detect_devices_never_raises_and_reports_cpu() -> None:
    report = detect_devices()
    assert report.selected in {"CPU", "GPU", "NPU"}
    assert any(d.name.upper().startswith("CPU") for d in report.available)
    assert isinstance(report.as_dict(), dict)


def test_resolve_device_marks_degraded_for_missing_preference() -> None:
    report = resolve_device(preferred="GPU")
    # Sin hardware Intel/OpenVINO, GPU no esta; debe degradar con aviso.
    if report.selected != "GPU":
        assert report.degraded is True
        assert report.warning
