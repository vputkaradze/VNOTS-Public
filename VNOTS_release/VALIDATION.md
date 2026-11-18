# Validation

Checked on 2026-10-14 with Python 3.10.9, TensorFlow 2.10.0, and SciPy 1.10.1
on the available Mac CPU environment. The requirements allow TensorFlow 2.10–2.15;
other versions and cluster GPU execution have not been exercised in this release check.

## Checks completed

- Python compilation and shell syntax checks passed.
- Five automated tests passed: cluster pair-count defaults, invalid time-step
  rejection, network parameter counts, piston constitutive/energy identities,
  and SO3 constitutive/energy identities and dissipativity.
- AST comparison confirmed that all extracted piston model functions and all
  SO3 architecture, residual, and loss functions are identical to their original
  definitions. The SO3 model wrapper differs only in the documented energy return.
- Seeded two-trajectory generation produced exactly equal array values to the
  original constitutive functions, sampling sequence, and integration/slicing path.
- Full default generation completed: piston 5000 trajectories / 100000 pairs;
  SO3 500 trajectories / 10000 pairs. Both datasets passed shape, finiteness,
  positive-temperature, and h=0.01 checks.
- Every learned case completed a short training run and saved weights: two updates
  for the fully learned cases and one update for each other learned case.
- All original trained checkpoints loaded successfully. All six piston cases and
  all four SO3 cases completed 20-step reconstructions, including the known baselines.
- The two piston cases with learned G also completed 500-step reconstructions
  under the default residual checks.
- Both systems generated their two figure panels and metrics CSV files.
- Replaying the original 20001-sample publication trajectories through the new
  plotting calculations reproduced the previously independently checked component
  MAE, mean energy error, and maximum energy error for all ten cases. The largest
  absolute metric difference was 4.45e-16.
- Code was checked for personal filesystem paths and inappropriate/internal review
  comments; none are included in the release scripts.

## Scope

The short training checks exercise software paths; they are not scientifically
converged training runs. No complete 100002/50002-update retraining was performed.
The original 20000-step trajectories were replayed for plotting checks; a new full
20000-step reconstruction of every model was not performed. New SO3 reconstruction
uses the documented consistent first-row initialization, which differs from the
historical saved files.

Generated test datasets, copied original checkpoints, test trajectories, caches,
and test plots are outside the release directory and are excluded from the archive.
The original experiment folders and existing release scripts are unchanged.
