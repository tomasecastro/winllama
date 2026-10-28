"""Tests de `winllama_engine.models`: cache local y manifest."""

from __future__ import annotations

from pathlib import Path

from winllama_engine.models import ModelEntry, ModelStore


def _make_ir_model(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "model.xml").write_text("<net/>", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"0" * 16)
    return model_dir


def test_register_and_list_models(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "qwen2-0_5b")
    store.register(
        ModelEntry(name="qwen2-0_5b", path=str(tmp_path / "qwen2-0_5b"), source="Qwen/Qwen2")
    )
    names = [e.name for e in store.list_models()]
    assert names == ["qwen2-0_5b"]
    assert (tmp_path / "manifest.json").is_file()


def test_resolve_raises_for_missing_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    try:
        store.resolve("nope")
    except FileNotFoundError:
        return
    raise AssertionError("Se esperaba FileNotFoundError")


def test_list_models_discovers_untracked_dirs(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "llama3-8b")
    entries = store.list_models()
    assert [e.name for e in entries] == ["llama3-8b"]
    assert entries[0].size_bytes > 0


def test_remove_deletes_model_and_manifest_entry(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "phi3")
    store.register(ModelEntry(name="phi3", path=str(tmp_path / "phi3")))
    assert store.remove("phi3") is True
    assert not (tmp_path / "phi3").exists()
    assert store.list_models() == []


def test_show_computes_size_and_metadata(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "tiny")
    store.register(
        ModelEntry(name="tiny", path=str(tmp_path / "tiny"), quantization="int4", parameters="0.5B")
    )
    entry = store.show("tiny")
    assert entry.quantization == "int4"
    assert entry.parameters == "0.5B"
    assert entry.size_bytes > 0


# ------------------------------------------------- pruebas T-213 (002 / US1)
# FR-117/FR-118 base: files[{path, sha256, bytes}], verified_at y mirrors
# (data-model.md §1). SHA256 hex de 64 chars; bytes > 0.
def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def test_verify_hashes_files_into_manifest(tmp_path: Path) -> None:
    """ModelStore.verify() calcula SHA256 por archivo y marca verified_at."""
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "tiny")
    store.register(ModelEntry(name="tiny", path=str(tmp_path / "tiny")))
    entry = store.verify("tiny")
    assert entry.verified_at, "verified_at debe quedar seteado tras verify"
    files = entry.files or []
    assert files, "files[] no puede quedar vacio tras verify (FR-117)"
    by_name = {f["path"]: f for f in files}
    expected = _sha256_file(tmp_path / "tiny" / "model.xml")
    assert by_name["model.xml"]["sha256"] == expected
    assert by_name["model.xml"]["sha256"] != expected[:63]  # 64 hex chars forma
    assert len(by_name["model.xml"]["sha256"]) == 64
    assert by_name["model.xml"]["bytes"] == 6
    assert by_name["model.bin"]["bytes"] == 16
    # Persistido en el manifest:
    raw = store._read_manifest()["tiny"]
    assert raw["verified_at"] == entry.verified_at
    assert raw["files"] == files


def test_verify_fail_closed_when_file_changes(tmp_path: Path) -> None:
    """FR-117: a corrobora con hash CERO tolerancia (fail-closed comparado)."""
    import hashlib

    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "tiny")
    store.register(ModelEntry(name="tiny", path=str(tmp_path / "tiny")))
    entry = store.verify("tiny")
    old_hash = entry.files[0]["sha256"]
    # Corrupcion del archivo: el hash RE-CALCULADO difiere (mismo Candido).
    (tmp_path / "tiny" / "model.xml").write_text("<net corrupted/>", encoding="utf-8")
    recomputed = hashlib.sha256(
        (tmp_path / "tiny" / "model.xml").read_bytes()
    ).hexdigest()
    assert recomputed != old_hash


def test_mirrors_persisted_in_manifest(tmp_path: Path) -> None:
    """ISS-05: la lista de espejos vive en el manifest (orden de intento)."""
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "tiny")
    entry = store.register(
        ModelEntry(
            name="tiny",
            path=str(tmp_path / "tiny"),
            source="EmbeddedLLM/Llama-3.2-1B-Instruct-int4-sym-ov",
            mirrors=["EmbeddedLLM/mirror-1", "second/mirror-2"],
        )
    )
    raw = store._read_manifest()["tiny"]
    assert raw["mirrors"] == ["EmbeddedLLM/mirror-1", "second/mirror-2"]
    assert entry.mirrors == ["EmbeddedLLM/mirror-1", "second/mirror-2"]


def test_verify_without_registry_uses_disk_entry(tmp_path: Path) -> None:
    """Un modelo en disco sin (o antes de) pasar por register: verify viene bien."""
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "sinreg")
    entry = store.verify("sinreg")
    assert entry.verified_at
    assert {f["path"] for f in entry.files or []} == {"model.xml", "model.bin"}


def test_list_models_backwards_compat_without_files(tmp_path: Path) -> None:
    """Manifest viejo (Hito 1) sin files/verified_at: sigue listando."""
    store = ModelStore(tmp_path)
    _make_ir_model(tmp_path, "legado")
    store.register(ModelEntry(name="legado", path=str(tmp_path / "legado")))
    entries = store.list_models()
    assert [e.name for e in entries] == ["legado"]
    assert entries[0].files == []
    assert entries[0].verified_at is None
