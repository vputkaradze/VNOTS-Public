"""Publication panels with the original component and entropy conventions."""
import csv
import importlib
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import EXPERIMENTS, case_flags


def so3_energy(states, run, case):
    # Recompute the Legendre transform; historical h_pred files may contain G.
    import tensorflow as tf
    from .models import setup, build

    setup("so3")
    cfg = EXPERIMENTS["so3"]
    model, _ = build("so3", case)
    if not all(case_flags("so3", case)):
        model.load_weights(str(run / f"{cfg.weight_prefix}{cfg.labels[case]}.h5"))
    result = []
    for start in range(0, len(states), 1024):
        chunk = states[start : start + 1024]
        x = chunk[:, 4:8]
        if len(x) == 1:
            x = np.repeat(x, 2, axis=0)
        g = model(tf.constant(x, tf.float64))[2].numpy().reshape(-1)[: len(chunk)]
        result.append(np.sum(chunk[:, :4] * chunk[:, 4:8], axis=1) - g)
    return np.concatenate(result)


def plot(args):
    cfg = EXPERIMENTS[args.system]
    run = Path(args.run_dir)
    selected = (
        range(len(cfg.labels))
        if args.cases == "all"
        else [int(x) for x in args.cases.split(",")]
    )
    loaded = {}
    for case in selected:
        case_flags(args.system, case)
        path = run / f"{cfg.result_prefix}{cfg.labels[case]}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; reconstruct it or select available cases with --cases."
            )
        with np.load(path) as raw:
            loaded[case] = {key: raw[key] for key in ("t_out", "y_exact", "y_pred")}
    if not loaded:
        raise ValueError("Select at least one case.")
    first = next(iter(loaded.values()))
    t, exact = first["t_out"], first["y_exact"]
    physics = importlib.import_module(f"vnots.{args.system}.physics")
    piston = args.system == "piston"
    cols = list(range(6 if piston else 8))
    names = (
        ["x", "v", "T1", "T2", "p", "S1"]
        if piston
        else ["mu1", "mu2", "mu3", "S", "Omega1", "Omega2", "Omega3", "T"]
    )
    ref_energy = physics.compute_energy(
        exact[:, [0, 4, 5, 6]].T if piston else exact[:, :4].T
    )
    variables, axes = plt.subplots(
        2, 3 if piston else 4, figsize=(15 if piston else 17, 7)
    )
    errors, error_axes = plt.subplots(1, 2, figsize=(12, 4))
    ref_color = "black" if piston else "blue"
    for index, ax in enumerate(axes.flat):
        ax.plot(t, exact[:, cols[index]], color=ref_color, label="Ground truth")
        ax.set_title(names[index])
        ax.set_xlabel("t")
    error_axes[0].plot(t, ref_energy, color=ref_color, label="Ground truth")
    metrics = []
    for case, raw in loaded.items():
        if (
            raw["t_out"].shape != t.shape
            or raw["y_exact"].shape != exact.shape
            or not np.allclose(raw["t_out"], t)
            or not np.allclose(raw["y_exact"], exact, rtol=1e-8, atol=1e-10)
        ):
            raise ValueError(
                "All plotted cases must share the same reference trajectory and time grid."
            )
        pred = raw["y_pred"]
        if pred.shape != exact.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid reconstructed array for case {case}")
        displayed = pred.copy()
        if piston:
            displayed[:, 5] += exact[0, 5] - pred[0, 5]
            aligned = pred.copy()
            aligned[:, 5:7] += exact[0, 5:7] - pred[0, 5:7]
            energy = physics.compute_energy(aligned[:, [0, 4, 5, 6]].T)
        else:
            energy = so3_energy(pred, run, case)
        mae = np.mean(np.abs(displayed[:, cols] - exact[:, cols]), axis=1)
        label, color = cfg.labels[case], cfg.colors[case]
        for index, ax in enumerate(axes.flat):
            ax.plot(t, displayed[:, cols[index]], color=color, label=label)
        error_axes[0].plot(t, energy, color=color, label=label)
        error_axes[1].semilogy(
            t, np.maximum(mae, np.finfo(float).tiny), color=color, label=label
        )
        metrics.append(
            [
                case,
                label,
                float(mae.mean()),
                float(np.mean(abs(energy - ref_energy))),
                float(np.max(abs(energy - ref_energy))),
                float(np.max(abs(energy - energy[0]))),
            ]
        )
    for ax in axes.flat:
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    error_axes[0].set_title("Energy")
    error_axes[1].set_title("MAE over (" + ", ".join(names) + ")")
    for ax in error_axes:
        ax.set_xlabel("t")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    variables.tight_layout()
    errors.tight_layout()
    if piston:
        filenames = (
            "Double_piston_all_cases_variables.pdf",
            "Double_piston_all_cases_energy_error.pdf",
        )
    else:
        filenames = (
            "SO3_observables_results_generalized.pdf",
            "SO3_observables_energy.pdf",
        )
    for fig, filename in zip((variables, errors), filenames):
        fig.savefig(run / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {run / filename}")
    with (run / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "label",
                "time_mean_component_mae",
                "time_mean_energy_error",
                "max_energy_error",
                "max_energy_change_from_initial",
            ]
        )
        writer.writerows(metrics)
