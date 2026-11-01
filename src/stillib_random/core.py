from __future__ import annotations

import hashlib
import json
import platform
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np

PathArg = str | PathLike[str]


BehaviorOnMismatch = Literal["raise", "warn"]


# value: a specific default algorithm (default_rng may change in numpy versions)
#        and spawn keys for hieararchical RNGs - handy for reproducibility and parallelization with job ids
def create_rng(seed: int, *ids: int) -> np.random.Generator:
    """
    Spawn a new PCG64DXSM random number generator from a given seed and optional spawn keys.
    Fixing the seed, spawn key and BitGenerator is a best effort attempt to ensure reproducibility across numpy versions.
    NumPy gurantees that PCG64DXSM will produce the same sequence of raw bits, but the mapping of those bits to floating point numbers may change across versions.

    Args:
        seed (int): The seed for the random number generator.
        *ids (int): Optional spawn keys for hierarchical RNGs.

    Returns:
        np.random.Generator: A new random number generator instance.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=ids)
    return np.random.Generator(np.random.PCG64DXSM(sequence))


class _Fingerprint(TypedDict):
    """
    Fingerprint schema for RNG identification.
    Contains SHA256 hashes of the generator state, 8 raw bits, and 8 normal-distributed floating point numbers.
    While the raw bit stream is guaranteed to be the same across numpy versions, the mapping of those bits to floating point numbers may change across versions.
    """

    state_sha256: str
    raw_sha256: str
    normal_sha256: str
    choice_sha256: str


@dataclass(frozen=True)
class Snapshot:
    """
    Schema for RNG snapshotting.
    """

    schema_version: int
    bit_generator: str
    state: Any
    fingerprint: _Fingerprint
    numpy_version: str
    python_version: str
    stillib_random_version: str


def _digest(value: Any) -> str:
    """Compute the SHA256 digest of a JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_probe() -> np.random.PCG64DXSM:
    """Get a probe instance of the PCG64DXSM bit generator for fingerprinting."""
    return np.random.PCG64DXSM(0)


def _fingerprint(state: Any) -> _Fingerprint:
    """
    Compute the fingerprint of a given RNG state.
    Hash:
        - The generator state
        - 8 raw bits from the generator
        - 8 normal-distributed floating point numbers
        - 8 random choices from a range of 10
    """

    probe = _get_probe()
    probe.state = state

    raw = [int(x) for x in probe.random_raw(8)]

    probe.state = state
    normal = [float(x).hex() for x in np.random.Generator(probe).normal(size=8)]

    probe.state = state
    choice = [int(x) for x in np.random.Generator(probe).choice(10, size=8)]

    return {
        "state_sha256": _digest(state),
        "raw_sha256": _digest(raw),
        "normal_sha256": _digest(normal),
        "choice_sha256": _digest(choice),
    }


def snapshot_rng(rng: np.random.Generator) -> Snapshot:
    """
    Create a snapshot of the current state of a PCG64DXSM random number generator, including its state and a fingerprint for verification.

    Args:
        rng (np.random.Generator): The random number generator to snapshot.

    Returns:
        Snapshot: The snapshot of the random number generator.

    Raises:
        ValueError: If the provided generator is not a PCG64DXSM instance.
    """
    if not isinstance(rng.bit_generator, np.random.PCG64DXSM):
        raise ValueError("Only PCG64DXSM bit generator is supported for snapshotting.")

    state = rng.bit_generator.state

    return Snapshot(
        schema_version=1,
        bit_generator="PCG64DXSM",
        state=state,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        stillib_random_version=version("stillib_random"),
        fingerprint=_fingerprint(state),
    )


def restore_rng(
    snapshot: Snapshot, *, on_mismatch: BehaviorOnMismatch = "raise"
) -> np.random.Generator:
    """
    Restore a PCG64DXSM random number generator state from a given snapshot, verifying its fingerprint for consistency.

    Args:
        snapshot (Snapshot): The snapshot containing the state and fingerprint of the random number generator.
        on_mismatch (BehaviorOnMismatch): Behavior when a fingerprint mismatch is detected; choose from "raise" or "warn".

    Returns:
        np.random.Generator: The restored random number generator.

    Raises:
        ValueError: If the snapshot schema version or bit generator is unsupported, or if a fingerprint mismatch is detected and on_mismatch is set to "raise".
        warnings.RuntimeWarning: If a fingerprint mismatch is detected and on_mismatch is set to "warn".
    """
    if on_mismatch not in (
        "raise",
        "warn",
    ):
        raise ValueError("on_mismatch must be one of 'raise' or 'warn'.")

    if snapshot.schema_version != 1:
        raise ValueError("Unsupported snapshot schema")

    if snapshot.bit_generator != "PCG64DXSM":
        raise ValueError("Unsupported BitGenerator")

    state = snapshot.state

    if state.get("bit_generator") != "PCG64DXSM":
        raise ValueError("Incompatible generator state")

    rng = np.random.Generator(_get_probe())
    rng.bit_generator.state = state

    expected = snapshot.fingerprint
    actual = _fingerprint(rng.bit_generator.state)

    differences = [name for name in actual if actual[name] != expected.get(name)]

    if differences:
        message = "RNG fingerprint mismatch: " + ", ".join(differences)

        if on_mismatch == "raise":
            raise ValueError(message)

        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return rng


def save_snapshot(
    snapshot: Snapshot, path: PathArg | None = None, *, force_replace: bool = False
) -> Path:
    """
    Save a Snapshot to a specified file path in JSON format.

    Args:
        snapshot (Snapshot): The snapshot to save.
        path (PathArg | None): The file path to save the snapshot. If None, defaults to "rng_snapshot.json" in the current working directory.
        force_replace (bool): If True, overwrite the file if it already exists. Defaults to False.

    Returns:
        Path: The resolved specified path.

    Raises:
        FileExistsError: If the file already exists and force_replace is False.
    """
    if path is None:
        path = Path.cwd() / "rng_snapshot.json"
    else:
        path = Path(path).expanduser().resolve()

    if path.exists() and not force_replace:
        raise FileExistsError(
            f"File {path} already exists. Use force_replace=True to overwrite."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(snapshot), file, indent=4)

    return path


def load_snapshot(path: PathArg) -> Snapshot:
    """
    Load a Snapshot from a specified file path in JSON format.

    Args:
        path (PathArg): The file path to load the snapshot from.

    Returns:
        Snapshot: The loaded snapshot.

    Raises:
        FileNotFoundError: If the file does not exist or isn't a file.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists() and not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist.")

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return Snapshot(**data)
