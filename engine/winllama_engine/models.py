"""Cache y manifest local de modelos OpenVINO IR.

El daemon guarda los modelos IR (`.xml` + `.bin`) en
`~/.winllama/models/<nombre>/` y mantiene un manifest JSON con metadatos
(origen Hugging Face, parametros, cuantizacion, fecha). Este modulo NO descarga
de la red: la descarga real (`pull`) corresponde al Hito 4; aqui se resuelve el
cache local, se listan modelos y se registran entradas.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("winllama_engine.models")

MANIFEST_NAME = "manifest.json"


def default_models_dir() -> Path:
    """Directorio de cache de modelos: `~/.winllama/models`."""
    return Path.home() / ".winllama" / "models"


@dataclass
class ModelEntry:
    """Metadatos de un modelo presente en el cache local.

    Campos 002 (T-213, FR-117/FR-118 base; data-model.md §1):
      - `files`: [{path, sha256, bytes}] — SHA256 hex de 64 chars (Hito 1: []).
      - `verified_at`: ISO8601 de la verificacion de hash (None = sin verificar).
      - `mirrors`: repos espejo en orden de intento (ISS-05; default []).
    """

    name: str
    path: str
    source: str | None = None
    quantization: str | None = None
    parameters: str | None = None
    size_bytes: int = 0
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    files: list[dict[str, object]] = field(default_factory=list)
    verified_at: str | None = None
    mirrors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    """SHA256 (hex 64 chars) de un archivo en bloques de 64 KB."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


class ModelStore:
    """Gestiona el cache local de modelos y su manifest."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_models_dir()
        self.manifest_path = self.root / MANIFEST_NAME

    # --------------------------------------------------------------- cache
    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_ir_dir(self, name: str) -> bool:
        """True si el modelo `name` contiene un grafo IR (`.xml`)."""
        model_dir = self.root / name
        return model_dir.is_dir() and any(model_dir.glob("*.xml"))

    def model_dir(self, name: str) -> Path:
        """Ruta absoluta al directorio del modelo `name`."""
        return (self.root / name).resolve()

    def resolve(self, name: str) -> Path:
        """Resuelve un nombre de modelo a su directorio IR existente.

        Raises:
            FileNotFoundError: si el modelo no esta en el cache o no tiene IR.
        """
        model_dir = self.root / name
        if not self.is_ir_dir(name):
            raise FileNotFoundError(
                f"El modelo '{name}' no esta en el cache '{self.root}' "
                "o no contiene un grafo IR (.xml)."
            )
        return model_dir.resolve()

    # ------------------------------------------------------------ manifest
    def _load_entry(self, name: str, model_dir: Path) -> ModelEntry | None:
        """Construye la entrada desde el manifest SIN exigir IR (o None)."""
        models = self._read_manifest()
        raw = models.get(name)
        if raw is None:
            return None
        return ModelEntry(
            name=name,
            path=str(model_dir),
            source=raw.get("source"),
            quantization=raw.get("quantization"),
            parameters=raw.get("parameters"),
            size_bytes=int(raw.get("size_bytes") or 0),
            added_at=raw.get("added_at", ModelEntry(name=name, path="").added_at),
            files=list(raw.get("files") or []),
            verified_at=raw.get("verified_at"),
            mirrors=list(raw.get("mirrors") or []),
        )

    def verify(self, name: str) -> ModelEntry:
        """Calcula SHA256 por archivo del modelo y persiste en el manifest
        con `verified_at` (FR-117). Corresponde a la base del pin de C-104.

        Raises:
            FileNotFoundError: si el modelo no esta en el cache o no tiene IR.
        """
        model_dir = self.resolve(name)
        files: list[dict[str, object]] = []
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                files.append(
                    {
                        "path": item.name,
                        "sha256": _sha256_file(item),
                        "bytes": item.stat().st_size,
                    }
                )
        merged = self._load_entry(name, model_dir)
        entry = ModelEntry(
            name=name,
            path=str(model_dir),
            files=files,
            verified_at=datetime.now(timezone.utc).isoformat(),
        )
        if merged is not None:
            # Preservar metadatos existentes si el modelo ya estaba registrado.
            entry.source = entry.source or merged.source
            entry.quantization = merged.quantization
            entry.parameters = merged.parameters
            entry.size_bytes = merged.size_bytes
            entry.added_at = merged.added_at
            entry.mirrors = list(merged.mirrors)
        self.register(entry)
        return entry

    def _read_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.is_file():
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Manifest corrupto en '%s': %s", self.manifest_path, exc)
            return {}
        return data.get("models", {}) if isinstance(data, dict) else {}

    def _write_manifest(self, models: dict[str, dict]) -> None:
        self.ensure_root()
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": models,
        }
        with self.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def register(self, entry: ModelEntry) -> ModelEntry:
        """Registra o actualiza un modelo en el manifest."""
        models = self._read_manifest()
        models[entry.name] = entry.as_dict()
        self._write_manifest(models)
        return entry

    def list_models(self) -> list[ModelEntry]:
        """Lista los modelos del cache (manifest + directorios con IR)."""
        models = self._read_manifest()
        entries: list[ModelEntry] = []
        seen: set[str] = set()

        for name, raw in models.items():
            model_dir = self.root / name
            if not self.is_ir_dir(name):
                continue
            seen.add(name)
            entries.append(
                ModelEntry(
                    name=name,
                    path=str(model_dir.resolve()),
                    source=raw.get("source"),
                    quantization=raw.get("quantization"),
                    parameters=raw.get("parameters"),
                    size_bytes=int(raw.get("size_bytes") or _dir_size_bytes(model_dir)),
                    added_at=raw.get("added_at", ModelEntry(name=name, path="").added_at),
                    files=list(raw.get("files") or []),
                    verified_at=raw.get("verified_at"),
                    mirrors=list(raw.get("mirrors") or []),
                )
            )

        # Modelos presentes en disco pero ausentes del manifest.
        if self.root.is_dir():
            for child in sorted(self.root.iterdir()):
                if child.is_dir() and child.name not in seen and self.is_ir_dir(child.name):
                    entries.append(
                        ModelEntry(
                            name=child.name,
                            path=str(child.resolve()),
                            size_bytes=_dir_size_bytes(child),
                        )
                    )
        return entries

    def show(self, name: str) -> ModelEntry:
        """Devuelve los metadatos de un modelo, calculando tamano si falta."""
        model_dir = self.resolve(name)
        entry = self._load_entry(name, model_dir)
        if entry is not None:
            if not entry.size_bytes:
                entry.size_bytes = _dir_size_bytes(model_dir)
            return entry
        return ModelEntry(
            name=name, path=str(model_dir), size_bytes=_dir_size_bytes(model_dir)
        )

    def remove(self, name: str) -> bool:
        """Elimina el modelo del disco y del manifest. True si existia."""
        model_dir = self.root / name
        existed = model_dir.is_dir()
        if existed:
            shutil.rmtree(model_dir)
        models = self._read_manifest()
        if name in models:
            models.pop(name)
            self._write_manifest(models)
            existed = True
        return existed
