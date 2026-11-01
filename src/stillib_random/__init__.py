"""
A Python package for creating, snapshotting, and restoring PCG64DXSM random number generators with fingerprinting for consistency verification.
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .core import (
    BehaviorOnMismatch,
    PathArg,
    Snapshot,
    create_rng,
    load_snapshot,
    restore_rng,
    save_snapshot,
    snapshot_rng,
)


def save_rng(
    rng: np.random.Generator, path: PathArg, *, force_replace: bool = False
) -> Path:
    """
    Export the state of a PCG64DXSM random number generator to a snapshot file, including its fingerprint for consistency verification.
    The generator can later be restored to that state from the snapshot file.

    Args:
        rng (np.random.Generator): The random number generator to snapshot.
        path (PathArg): The file path where the snapshot will be saved.
        force_replace (bool): If True, overwrite the snapshot file if it already exists. Defaults to False.

    Returns:
        Path: The resolved speficied path.

    Raises:
        ValueError: If the provided generator is not a PCG64DXSM instance.
        FileExistsError: If the snapshot file already exists and force_replace is False.

    """
    snapshot = snapshot_rng(rng)
    return save_snapshot(snapshot, path, force_replace=force_replace)


def load_rng(
    path: PathArg, *, on_mismatch: BehaviorOnMismatch = "raise"
) -> np.random.Generator:
    """
    Load a PCG64DXSM random number generator from a snapshot file, restoring its state and verifying its fingerprint for consistency.

    Args:
        path (PathArg): The file path of the snapshot to load.
        on_mismatch (BehaviorOnMismatch): The behavior to adopt when a fingerprint mismatch is detected; choose from "raise" or "warn". Defaults to "raise".

    Returns:
        np.random.Generator: The restored random number generator.

    Raises:
        ValueError: If the snapshot file is invalid or the generator is not a PCG64DXSM instance.
        warnings.RuntimeWarning: If a fingerprint mismatch is detected and on_mismatch is set to "warn".
        FileNotFoundError: If the snapshot file does not exist.

    """
    snapshot = load_snapshot(path)
    return restore_rng(snapshot, on_mismatch=on_mismatch)


__version__ = version("stillib_random")

__all__ = [
    "Snapshot",
    "create_rng",
    "restore_rng",
    "save_snapshot",
    "load_snapshot",
    "snapshot_rng",
    "save_rng",
    "load_rng",
]
