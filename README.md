# stillib_random - a python utility for reproducible random number generation built on NumPy

## Why this exists
The purpose of this utility is to standardize how random number generators (RNGs) are constructed with a Snapshot abstraction for documentation, restore-state, and pause/resume funcitonality. The basic object is still `nump.random.Generator` without any wrapping. Reproducibility is enhanced by enforcing:
- Users must supply a `seed` to create an RNG
- RNGs can spawn deterministically from that seed using a spawn-key/integer-sequence. This is handy for multiprocessing where an RNG can be constructed from the base seed and a `job_id` rather than passing aroud RNGs or tying identity to the worker process.
- Rather than relying on `numpy.random.default_rng`, the utility forces the underlying `BitGenerator` to be `PCG64DXSM` specifically. NumPy guarantees that the same seed and spawn-key will produce the same sequence of raw random bits across versions.
- NumPy does not guarantee that the transformation from raw random bits to e.g. normal or uniform floating point numbers is stable across versions. If an RNG is reconstructed later in a different NumPy build, this can lead to diverging random samples even if the `BitGenerator` is identical. To catch such silent drift, the utility stores a fingerprint of the snapshotted RNG that can later validate the reconstructed state.


## Installation 
### Using PIP
Clone this repository and navigate to the designated location. Activate the virtual environment where you wish to install the utility, then run:
```bash
pip install -e .
```
**_Alternatively:_**

### Add `stillib_random` as a dependency of your project
If your project contains a `pyproject.toml` file, you can add the utility under the project dependencies field:
```TOML
# your pyproject.toml

[project]
# other settings ...
dependencies = [
    "stillib_random @ git+https://github.com/andreas-stillits/stillib_random.git@v0.1.0",
]
```
Different releases can be referenced by modifying the trailing tag.

## Quick Start 

### Everyday usage
Instantiate an RNG using a seed and draw random numbers:
```python
from stillib_random import create_rng

rng = create_rng(176)
sample = rng.normal(size=1_000)
```

### Snapshotting
Create a Snapshot object and use it to restore the state later on:
```python
from stillib_random import create_rng, snapshot_rng, restore_rng

rng = create_rng(176)
snapshot = snapshot_rng(rng)
sample1 = rng.random(10)

rng = restore_rng(snapshot) # validates the saved fingerprint against that of the retored RNG
sample2 = rng.random(10)

assert np.all(sample1 == sample2)
```

Save the Snapshot to JSON and reproduce later on - even if the seed and spawn-key is lost:
```python
from stillib_random import create_rng, save_rng, load_rng

rng = create_rng(176)
save_rng(rng, "snapshots/example.json")
sample1 = rng.random(10)

# restore from file later
rng = load_rng("snapshots/example.json", on_mismatch="raise") # raises ValueError if the fingerprints do not match
sample2 = rng.random(10)

assert np.all(sample1 == sample2)
```
The JSON contains additional provenance that can aid replication if the fingerprints have diverged.

### Example JSON

```JSON
{
    "schema_version": 1,
    "bit_generator": "PCG64DXSM",
    "state": {
        "bit_generator": "PCG64DXSM",
        "state": {
            "state": 290829991650476195847285471627591589367,
            "inc": 304044327811940190220232591581763596903
        },
        "has_uint32": 0,
        "uinteger": 0
    },
    "fingerprint": {
        "state_sha256": "7e3d2ade13008d7d60c340f947f075bb154cc9754b622ce2b23800f9ecb8124f",
        "raw_sha256": "f6673f3ab755a5df523497cfa3d32466c18aad85c85010a980271ef8d117c8db",
        "normal_sha256": "20e029bc484a8f4bb67ec2b42da6db2277d59e0d9b219357924f804fe47130ad",
        "choice_sha256": "f768267177769ee3a91f83a0649e50e558b3ce73bdb861dc02ccdaf1eae0f787",
        "uniform_sha256": "96a97c612dc25f956bf0cbe9951eed2a126983f883626d919b1d195ddbdd565c"
    },
    "numpy_version": "2.0.0",
    "python_version": "3.10.22",
    "stillib_random_version": "0.1.0"
}
```
The fingerprint consists of 5 SHA256 hashes:
- state_sha256: the hash of the `numpy.random.Generator.BitGenerator.state` contents
- raw_sha256: the hash of 8 raw random bits
- normal_sha256: the hash of 8 normal floating points in hex representation
- choice_sha256: the hash of 8 choices from `range(10)`
- uniform_sha256: the hash of 8 uniform floating points in hex representation

NumPy should guarantee that `raw_sha256` is stable across versions because the utility fixes the `BitGenerator` to `PCG64DXSM`. The last 3 are transformations of the raw bits that may vary across NumPy builds. **The fingerprinting process does not advance the state of the passed RNG object** - neither on save or load.

## Multiprocessing Example

RNGs can easily be associated with logical jobs one may wish to split across multiple processes. The following example showcases parallel random computation independent of the job order or worker identity during scheduling:
```python
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from stillib_random import create_rng

SEED = 176

@dataclass
class Job:
    data: float
    id: int

def add_noise(job: Job) -> Job:
    rng = create_rng(SEED, job.id)
    noise = rng.normal()
    new_data = job.data + noise
    return Job(new_data, job.id)

jobs = [Job(i, i) for i in range(5)]

sequential = list(map(add_noise, jobs))

with ProcessPoolExecutor(max_workers=2) as pool:
    parallel = list(pool.map(add_noise, jobs))
    #pool.map preserves input order but what worker sees what job is not fixed across runs.

assert sequential == parallel
```
`create_rng` can take any number of integer keys and that may reflect sub-tasks and sub-sub-tasks, etc. This avoids an arbitrary rule like `job_seed = seed + job_id` that could produce unintentionally identical streams across runs - e.g. using `seed-1` in another run would produce `n_jobs-1` sequences also present in the previous `seed` run. Every seed and spawn-key combination is instead a unique stream identity by virtue of `numpy.random.SeedSequence`.