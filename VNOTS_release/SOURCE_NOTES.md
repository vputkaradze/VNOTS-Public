# Source and numerical conventions

The release was prepared from the original experiment directories, not from
an alternative formulation of the paper equations. `SOURCE_MANIFEST.json` gives
relative source names and SHA256 hashes of files used to extract executable code.

| Release component | Original source |
|---|---|
| Piston parameters and constitutive functions | `Double chamber piston/parameters.py`, `useful_programs.py` |
| Piston models, residuals, loss | `Double chamber piston/dp_train_one_case.py` |
| Piston generation/sampling | `Double chamber piston/dp_generate_data.py`; supplied cluster generator |
| Piston reconstruction and figure definitions | `Reconstruction_double_piston.py`, `Plot_all_variables_V6.py` |
| SO3 parameters and constitutive functions | `SO3_computations/parameters.py`, `useful_programs.py` |
| SO3 models, residuals, loss | `SO3_computations/so3_case.py` |
| SO3 generation/sampling | `Generate_data_SO3_diss_observable.py`; supplied cluster generator |
| SO3 reconstruction and figure definitions | `so3_reconstruction_only.py`, `plot_all_variable_V5.py` |

## Defaults and provenance

The checked cluster datasets contained 100,000 piston pairs and 10,000 SO3 pairs.
The supplied SO3 cluster generator had a 5000-trajectory default, which would make
100,000 pairs if run unchanged. This release uses 500 SO3 trajectories to match the
measured file size. Piston uses 5000. Neither choice proves the origin of an old
checkpoint. Original datasets, trained weights, and result binaries are not included.

Training defaults preserve the supplied scripts' inclusive-loop update counts,
learning-rate endpoints, batch sizes, model precision, and network initializers.
The release sets seeds for new experiments. Different TensorFlow versions, device
execution, and the new seed setup can produce different trained weights.

Piston anchors: 15×15 temperature grid on [0.5,2]² at x=v=0 for mass and thermal
curvature; 200 random stationary points for zero momentum. SO3 anchors: 20³
angular velocities on [-0.5,0.5]³ at T=2, plus 100 zero-velocity points with T in
[0.1,2]. Targets are G, its four first derivatives, and G_TT (temperature difference
1e-3). The original 64-point zero-momentum penalty and 64-point friction-trace
penalty (target 0.075, weight 0.01) are preserved.

## Release changes

- One CLI replaces machine-specific paths and separate overlapping runners.
- Unused experimental functions, commented-out code, and informal internal comments
  are omitted. Core function bodies are extracted from the originals.
- Generation preallocates pair arrays instead of repeatedly concatenating them.
  Sampling order, integration interval, retained slices, and formats are preserved.
- Piston energy uses elementwise kinetic energy for batched inputs. Scalar results
  are unchanged; the historical publication plotter evaluated one state at a time.
- SO3 energy computes its quadratic form in linear memory instead of allocating
  a trajectory-length squared matrix and taking its diagonal.
- The SO3 model returns H = y·grad(G) − G for unknown G, rather than returning G
  in its energy field. The publication plotter already used the Legendre transform.
- SO3 initial reconstructed momentum and entropy are derived from the model, like
  all later rows. This removes the inconsistent first saved row of the original
  reconstructor. Initial observables and all variational update equations are unchanged.
- Reconstruction checks residuals, finite states, positive temperatures, and piston
  geometry. Float32 piston roots can retry with an appropriate difference scale.
  Diagnostics make these acceptance decisions visible.
- Plotting follows the original publication's entropy and MAE definitions. SO3
  energy-panel colors are made consistent with its variable-panel colors.
- Per-run JSON records and CSV losses identify the actual dataset, code, and weights.

The learned dynamics are not claimed to be an exact reproduction of any historical
training run. The small documented output corrections can change the initial
SO3 MAE/energy point, and root safeguards can change a reconstruction that previously
accepted an unconverged step. The paper figure assets themselves are not modified.
