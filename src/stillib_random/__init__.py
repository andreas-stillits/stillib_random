from importlib.metadata import version

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
) -> None:
    snapshot = snapshot_rng(rng)
    save_snapshot(snapshot, path, force_replace=force_replace)


def load_rng(
    path: PathArg, *, on_mismatch: BehaviorOnMismatch = "raise"
) -> np.random.Generator:
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
