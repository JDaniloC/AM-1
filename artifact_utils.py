"""Utilities to persist and load model artifacts shared across notebooks.

The goal is to avoid serializing heavy model objects. Instead, each notebook
stores self-contained metadata and prediction tensors under
``artifacts/<model_key>/<dataset_slug>/`` so that downstream analysis can be
performed without re-importing the training dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

ARTIFACT_ROOT = Path("artifacts")
ARTIFACT_VERSION = 1


@dataclass
class ArtifactBundle:
    """Container used when reading artifacts back from disk."""

    model_key: str
    dataset_name: str
    dataset_slug: str
    metadata: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    path: Path


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "dataset"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_serializable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def get_dataset_dir(model_key: str, dataset_name: str) -> Path:
    slug = _slugify(dataset_name)
    return ARTIFACT_ROOT / model_key.lower() / slug


def write_artifact_bundle(
    *,
    model_key: str,
    dataset_name: str,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_prob: Optional[np.ndarray],
    class_labels: Iterable[Any],
    metrics: Mapping[str, Any],
    hyperparams: Optional[Mapping[str, Any]] = None,
    runtime_seconds: Optional[float] = None,
    sample_ids: Optional[Iterable[Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write metadata + prediction arrays for a single dataset/model combo."""

    dataset_dir = get_dataset_dir(model_key, dataset_name)
    _ensure_dir(dataset_dir)

    metadata: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "model_key": model_key,
        "dataset_name": dataset_name,
        "dataset_slug": dataset_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "class_labels": list(class_labels),
        "metrics": {k: _as_serializable(v) for k, v in metrics.items()},
    }

    if hyperparams is not None:
        metadata["hyperparams"] = {
            k: _as_serializable(v) for k, v in hyperparams.items()
        }
    if runtime_seconds is not None:
        metadata["runtime_seconds"] = float(runtime_seconds)
    if extra_metadata:
        metadata.update({k: _as_serializable(v) for k, v in extra_metadata.items()})

    metadata_path = dataset_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    arrays: Dict[str, np.ndarray] = {
        "y_true": np.asarray(y_true),
    }
    if y_pred is not None:
        arrays["y_pred"] = np.asarray(y_pred)
    if y_prob is not None:
        arrays["y_prob"] = np.asarray(y_prob)
    if sample_ids is not None:
        arrays["sample_ids"] = np.asarray(list(sample_ids))

    np.savez_compressed(dataset_dir / "predictions.npz", **arrays)
    return dataset_dir


def load_artifact_bundle(model_key: str, dataset_name: Optional[str] = None) -> List[ArtifactBundle]:
    """Load all stored artifacts for ``model_key`` (optionally filtered)."""

    base_dir = ARTIFACT_ROOT / model_key.lower()
    if not base_dir.exists():
        return []

    bundles: List[ArtifactBundle] = []
    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_name and _slugify(dataset_name) != dataset_dir.name:
            continue
        metadata_path = dataset_dir / "metadata.json"
        predictions_path = dataset_dir / "predictions.npz"
        if not metadata_path.exists() or not predictions_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        arrays = dict(np.load(predictions_path, allow_pickle=False))
        bundles.append(
            ArtifactBundle(
                model_key=model_key,
                dataset_name=metadata.get("dataset_name", dataset_dir.name),
                dataset_slug=dataset_dir.name,
                metadata=metadata,
                arrays=arrays,
                path=dataset_dir,
            )
        )
    return bundles


def load_all_artifacts() -> Dict[str, List[ArtifactBundle]]:
    """Return every stored artifact grouped by model key."""

    artifacts: Dict[str, List[ArtifactBundle]] = {}
    if not ARTIFACT_ROOT.exists():
        return artifacts
    for model_dir in ARTIFACT_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        artifacts[model_key] = load_artifact_bundle(model_key)
    return artifacts


def clear_artifacts(model_key: Optional[str] = None, dataset_name: Optional[str] = None) -> None:
    """Helper to remove stored artifacts during development."""

    if model_key is None:
        if ARTIFACT_ROOT.exists():
            for child in ARTIFACT_ROOT.iterdir():
                if child.is_dir():
                    clear_artifacts(child.name, dataset_name)
        return

    base_dir = ARTIFACT_ROOT / model_key.lower()
    if not base_dir.exists():
        return
    for dataset_dir in base_dir.iterdir():
        if dataset_name and _slugify(dataset_name) != dataset_dir.name:
            continue
        for file in dataset_dir.glob("*"):
            file.unlink(missing_ok=True)
        dataset_dir.rmdir()
    if not any(base_dir.iterdir()):
        base_dir.rmdir()
