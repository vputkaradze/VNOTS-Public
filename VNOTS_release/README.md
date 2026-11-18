# V-NOTS: piston and dissipative rigid-body experiments

Code for generating trajectory pairs, learning thermodynamic Lagrangians and
friction forces, and reconstructing long trajectories with variational integrators.
The constitutive laws, network architectures, discrete residuals, and anchor
losses are extracted from the original experiment programs. See
[SOURCE_NOTES.md](SOURCE_NOTES.md) for the source mapping and release changes.

## Installation

Use Python 3.10 or 3.11 and TensorFlow/Keras 2. The original HDF5 checkpoints use
the Keras 2 format; Keras 3 is not supported by this release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

For a Linux GPU cluster, activate an environment with a compatible TensorFlow 2
and CUDA installation. Data generation and SciPy reconstruction run on CPU;
TensorFlow training can use an available GPU. Commands below run from this directory.

## Default experiments

| Setting | Piston | SO(3) rigid body |
|---|---:|---:|
| Trajectories | 5,000 | 500 |
| Retained points per trajectory | 21 | 21 |
| Consecutive data pairs | 100,000 | 10,000 |
| Pair time interval | 0.01 | 0.01 |
| Optimizer updates per learned case | 100,002 | 50,002 |
| Batch size, sampled with replacement | 1,024 | 1,024 |
| Initial learning rate | 0.001 | 0.001 |
| Final learning rate | 0.00001 | 0.0001 |
| Model arithmetic | float32 | float64 |
| Reconstruction time steps | 20,000 | 20,000 |
| Reconstruction samples, including t=0 | 20,001 | 20,001 |

The data sizes match the checked cluster files. The schedules and model precision
follow the supplied original training scripts. These are distinct sources of
information: present files and script defaults do not establish the settings of
every historical training run. New runs record their actual dataset hash and settings.

## Generate and inspect data

```bash
python run.py piston generate
python run.py so3 generate
python run.py piston check-data
python run.py so3 check-data
```

Outputs go to `runs/piston/` and `runs/so3/`. Use `--run-dir DIRECTORY` to choose
another location; use `--data FILE` to read an existing dataset in training or
inspection (or to choose the output dataset filename during generation).

Generation uses the original NumPy uniform draws and SciPy RK45 with
`rtol=1e-10`, `atol=1e-12`. Piston initial x and p are in [-1,1] and both entropies
in [0,1]. SO(3) initial momentum components and entropy are in [-1,1]. The original
extra integration sample is retained internally and discarded before constructing
the pairs. `--points 21` therefore means 20 pairs per trajectory.

`--seed 0` is the reproducible default for new runs. Newly generated data are not
claimed to be byte-identical to the historical unseeded datasets.

Original on-disk formats are preserved:

- Piston: `Double_piston_G_vars.npy`, object array `(2, pairs, 8)`; columns are
  x, v, T1, T2, p, S1, S2, time. Loading requires `allow_pickle=True`; use trusted files.
- SO(3): `Dissipative_Rigid_Body_Data_SO3_generalized.npz`, keys `Begin_pts` and
  `End_pts`, each `(9, pairs)`; rows are mu1, mu2, mu3, S, Omega1, Omega2, Omega3, T, time.

## Train

```bash
python run.py piston train --case 3
python run.py so3 train --case 3
```

Each command trains one case. Repeat for the desired cases:

| Case | Piston | SO(3) |
|---|---|---|
| 0 | Learn G; both forces known | Learn G; force known |
| 1 | G known; learn both forces | G known; learn force |
| 2 | All known; no training | All known; no training |
| 3 | Learn G and both forces | Learn G and force |
| 4 | G and F1 known; learn F2 | — |
| 5 | G and F2 known; learn F1 | — |

Training uses only observable coordinates from trajectory pairs. Constitutive
anchor targets are computed from the known model, as in the original experiments.
`--updates` is an exact number of optimizer updates, not epochs or full-data passes.
The defaults reproduce the original inclusive loops (0 through 100001 or 50001).
Architectures and anchor weights are fixed to the original experiment values.

Files include `weights_<label>.h5` (piston), `weights_SO3_<label>.h5` (SO3),
`loss_<label>.csv`, and `train_<label>.json`. Metadata records data and weight
SHA256 hashes, package versions, seed, update count, precision, and source hashes.
A training record changes from `running` to `complete` only after weights are saved.
Reuse of the same run directory and case overwrites that case's outputs.

## Reconstruct and plot

```bash
python run.py piston reconstruct --cases 3
python run.py piston plot --cases 3
python run.py so3 reconstruct --cases 3
python run.py so3 plot --cases 3
```

To run the fully known baseline without data or weights:

```bash
python run.py piston reconstruct --cases 2
python run.py so3 reconstruct --cases 2
```

For all cases, train each learned case, then omit `--cases` from reconstruction
and plotting. Original weight and result filenames are accepted; copy them into
an appropriate run directory. SO3 plotting recomputes the Legendre energy from
the weights, because the historical `h_pred` field can contain G rather than H.

`--steps 20000` means 20,000 advances of h=0.01, reaching t=200. Reference tolerances
match the original reconstructors: piston 1e-10/1e-12, SO3 1e-8/1e-10.
Implicit updates use SciPy `root(method='hybr', tol=1e-10)` and the original residuals.
Every accepted step must be finite, satisfy the residual threshold, and have
positive temperature (and valid chamber volumes for the piston). The default
max-norm residual thresholds are 1e-4 for float32 piston arithmetic and 1e-8 for
float64 SO3. Piston solves that stall may retry with a finite-difference scale
appropriate for float32. Solver flags, residuals, and retries are recorded.
`--residual-tol` changes only the acceptance threshold; it does not change the equations.

Result arrays retain the original layouts:

- Piston `y_exact`, `y_pred`: x, v, T1, T2, p, S1, S2.
- SO3 `y_exact`, `y_pred`: mu1, mu2, mu3, S, Omega1, Omega2, Omega3, T.

Plotting creates the publication-named PDF panels and `metrics.csv`:

- Piston: S1 is aligned at t=0 for the six-component MAE; both entropies are
  aligned at t=0 for evaluating the true Hamiltonian. Momentum is never shifted.
- SO3: MAE averages all eight saved components without an entropy shift. Energy
  is mu·Omega + T S − G using the learned or known G for that case.

Errors are raw component errors without scale normalization. Colors follow the
publication's variable panels and are used consistently in the energy panels.

## Small checks

```bash
python -m unittest discover -s tests -v
python run.py piston generate --run-dir runs/smoke-piston --trajectories 2
python run.py piston train --run-dir runs/smoke-piston --case 3 --updates 2 --batch-size 4
python run.py piston reconstruct --run-dir runs/smoke-piston --cases 2 --steps 10
python run.py piston plot --run-dir runs/smoke-piston --cases 2
```

Repeat with `so3` and a separate run directory. Two updates are a software check,
not a trained model suitable for long-time reconstruction. See `VALIDATION.md`
for the checks performed on this release.

## Slurm

A portable array-job entry point is provided in `scripts/train_array.sh`:

```bash
mkdir -p logs
sbatch --array=0-5 --output=logs/piston_%A_%a.out scripts/train_array.sh piston
sbatch --array=0-3 --output=logs/so3_%A_%a.out scripts/train_array.sh so3
```

Generate the dataset before submitting training. Activate the intended Python
environment before submission, and add the partition, account, GPU, memory, and
time options appropriate to your cluster. The script uses the submission directory
and `SLURM_ARRAY_TASK_ID`; it contains no personal paths or credentials.
