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
    sequence = np.random.SeedSequence(seed, spawn_key=ids)
    return np.random.Generator(np.random.PCG64DXSM(sequence))


class _Fingerprint(TypedDict):
    state_sha256: str
    raw_sha256: str
    normal_sha256: str


@dataclass(frozen=True)
class Snapshot:
    schema_version: int
    bit_generator: str
    state: Any
    numpy_version: str
    python_version: str
    stillib_random_version: str
    fingerprint: _Fingerprint


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_probe() -> np.random.PCG64DXSM:
    return np.random.PCG64DXSM(0)


def _fingerprint(state: Any) -> _Fingerprint:
    probe = _get_probe()
    probe.state = state

    raw = [int(x) for x in probe.random_raw(8)]

    probe.state = state
    normal = [float(x).hex() for x in np.random.Generator(probe).normal(size=8)]

    return {
        "state_sha256": _digest(state),
        "raw_sha256": _digest(raw),
        "normal_sha256": _digest(normal),
    }


def snapshot_rng(rng: np.random.Generator) -> Snapshot:
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
    snapshot: Snapshot, path: PathArg, *, force_replace: bool = False
) -> Path:
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
    path = Path(path).expanduser().resolve()
    if not path.exists() and not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist.")

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return Snapshot(**data)
