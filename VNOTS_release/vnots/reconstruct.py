"""Long-time reconstruction with the original variational equations."""
import importlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from scipy.optimize import root
from .config import EXPERIMENTS, case_flags
from .models import setup, build
from .util import sha256, write_metadata


def reference(system, t):
    physics = importlib.import_module(f"vnots.{system}.physics")
    if system == "piston":
        initial, rhs, rtol, atol = (
            [0.0, -0.5, 0.5, 0.5],
            physics.double_piston,
            1e-10,
            1e-12,
        )
    else:
        initial, rhs, rtol, atol = (
            [-0.25, 0.5, 0.5, 0.5],
            physics.Diss_Euler_eqs_generalized,
            1e-8,
            1e-10,
        )
    sol = solve_ivp(
        rhs, (0, t[-1]), initial, t_eval=t, method="RK45", rtol=rtol, atol=atol
    )
    if not sol.success or sol.y.shape[1] != len(t) or not np.isfinite(sol.y).all():
        raise RuntimeError(f"Reference ODE failed: {sol.message}")
    if system == "piston":
        from .piston.parameters import M

        x, p, s1, s2 = sol.y
        states = np.column_stack(
            (
                x,
                p / M,
                physics.Ufunc(s1, x, 0)[2],
                physics.Ufunc(s2, x, 1)[2],
                p,
                s1,
                s2,
            )
        )
    else:
        states = np.column_stack((sol.y.T, physics.dhdy(sol.y).T))
    return states, np.asarray(physics.compute_energy(sol.y)), rtol, atol


def reconstruct(args):
    cfg = EXPERIMENTS[args.system]
    dtype = setup(args.system, args.seed)
    if args.steps < 1:
        raise ValueError("steps must be at least 1.")
    case_ids = (
        list(range(len(cfg.labels)))
        if args.cases == "all"
        else [int(x) for x in args.cases.split(",")]
    )
    for case in case_ids:
        case_flags(args.system, case)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f"vnots.{args.system}.model")
    physics = importlib.import_module(f"vnots.{args.system}.physics")
    t = np.arange(args.steps + 1) * 0.01
    exact, h_exact, rtol, atol = reference(args.system, t)
    obs = slice(0, 4) if args.system == "piston" else slice(4, 8)
    residual_tol = (
        args.residual_tol
        if args.residual_tol is not None
        else (1e-4 if args.system == "piston" else 1e-8)
    )
    if residual_tol <= 0:
        raise ValueError("residual-tol must be positive.")
    for case in case_ids:
        model, _ = build(args.system, case)
        flags = case_flags(args.system, case)
        label = cfg.labels[case]
        weight_path = run / f"{cfg.weight_prefix}{label}.h5"
        weight_hash = None
        if not all(flags):
            model.load_weights(str(weight_path))
            weight_hash = sha256(weight_path)

        @tf.function
        def predict(x):
            if args.system == "piston":
                grad = model([x, x])[1]
                return grad
            return model(x, training=False)

        @tf.function
        def equations(xf, x0):
            if args.system == "piston":
                return tf.stack(module.compute_eqs(xf, x0, model), axis=1)
            rmu, rs = module.compute_eqs(
                xf, x0, model, tf.constant(0.01, dtype), tf.constant(1.0, dtype)
            )
            return tf.concat([rmu, rs[:, None]], axis=1)

        def tensor(x):
            # The original SO3 helpers expect doubled single-state inputs.
            return tf.constant(np.stack([x, x]), dtype=dtype)

        def full_state(x):
            if args.system == "piston":
                grad = predict(tensor(x)).numpy()[0]
                return np.r_[x, grad[1:4]], None
            energy, grad, _, _ = predict(tensor(x))
            return np.r_[grad.numpy()[0], x], float(energy.numpy()[0])

        pred = np.zeros_like(exact)
        h_pred = np.zeros(len(t))
        pred[0], energy = full_state(exact[0, obs])
        if energy is not None:
            h_pred[0] = energy
        norms = np.zeros(args.steps)
        status = np.zeros(args.steps, dtype=bool)
        retries = 0
        for j in range(1, len(t)):
            previous = pred[j - 1, obs]
            previous_tf = tensor(previous)

            def residual(x):
                return equations(tensor(x), previous_tf).numpy()[0].astype(float)

            result = root(residual, previous, method="hybr", tol=1e-10)
            norm = float(np.max(np.abs(residual(result.x))))
            if args.system == "piston" and (
                not np.isfinite(norm) or norm > residual_tol
            ):
                # Finite differences below float32 precision can stall MINPACK.
                alternative = root(
                    residual,
                    previous,
                    method="hybr",
                    tol=1e-10,
                    options={"eps": np.finfo(np.float32).eps},
                )
                alternative_norm = float(np.max(np.abs(residual(alternative.x))))
                retries += 1
                if np.isfinite(alternative_norm) and (
                    not np.isfinite(norm) or alternative_norm < norm
                ):
                    result, norm = alternative, alternative_norm
            x = result.x
            valid = np.isfinite(x).all() and np.isfinite(norm) and norm <= residual_tol
            valid = valid and (
                np.all(x[2:4] > 0) and abs(x[0]) < 2
                if args.system == "piston"
                else x[3] > 0
            )
            if not valid:
                raise RuntimeError(
                    f"{label}: root failed at step {j}; residual={norm:.3e}; {result.message}"
                )
            pred[j], energy = full_state(x)
            if not np.isfinite(pred[j]).all() or (
                energy is not None and not np.isfinite(energy)
            ):
                raise RuntimeError(f"{label}: nonfinite reconstruction at step {j}")
            if energy is not None:
                h_pred[j] = energy
            norms[j - 1], status[j - 1] = norm, result.success
            if j % args.print_every == 0 or j == args.steps:
                print(
                    f"{label}: step {j}/{args.steps} | residual={norm:.3e}", flush=True
                )
        if args.system == "piston":
            aligned = pred.copy()
            aligned[:, 5:7] += exact[0, 5:7] - pred[0, 5:7]
            h_pred = physics.compute_energy(aligned[:, [0, 4, 5, 6]].T)
        result_path = run / f"{cfg.result_prefix}{label}.npz"
        np.savez(
            result_path,
            t_out=t,
            y_exact=exact,
            y_pred=pred,
            h_exact=h_exact,
            h_pred=h_pred,
            root_residual=norms,
            root_success=status,
            known_G=flags[0],
            label=label,
            case_id=case,
            **(
                {"known_F": flags[1]}
                if args.system == "so3"
                else {"known_F1": flags[1], "known_F2": flags[2]}
            ),
        )
        write_metadata(
            result_path.with_suffix(".json"),
            operation="reconstruct",
            system=args.system,
            case=case,
            steps=args.steps,
            h=0.01,
            dtype=cfg.dtype,
            weights_sha256=weight_hash,
            result_sha256=sha256(result_path),
            reference_rtol=rtol,
            reference_atol=atol,
            root_tol=1e-10,
            residual_tol=residual_tol,
            max_residual=float(norms.max()),
            float32_retries=retries,
            accepted_without_solver_success=int(np.count_nonzero(~status)),
            initialization="hidden coordinates from derivatives of G at initial observables",
            energy="true H with initial entropy alignment"
            if args.system == "piston"
            else "Legendre transform of learned or exact G",
        )
        print(f"Saved: {result_path}")
