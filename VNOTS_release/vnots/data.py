"""Generate adjacent pairs using the original sampling and RK45 tolerances."""
import importlib
import time
import numpy as np
from scipy.integrate import solve_ivp
from .config import EXPERIMENTS
from .util import data_path, sha256, write_metadata


def generate(args):
    cfg = EXPERIMENTS[args.system]
    physics = importlib.import_module(f"vnots.{args.system}.physics")
    par = importlib.import_module(f"vnots.{args.system}.parameters")
    trajectories = cfg.trajectories if args.trajectories is None else args.trajectories
    if trajectories < 1 or args.points < 3:
        raise ValueError("Use at least one trajectory and three retained points.")
    np.random.seed(args.seed)
    path = data_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = trajectories * (args.points - 1)
    features = 8 if args.system == "piston" else 9
    begin = np.empty((count, features))
    end = np.empty_like(begin)
    # The original code integrates one extra sample, then retains the first npts.
    t_out = par.h * np.arange(args.points + 1)
    started = time.perf_counter()
    for j in range(trajectories):
        if args.system == "piston":
            initial = np.r_[
                np.random.uniform(-1, 1, 1),
                np.random.uniform(-1, 1, 1),
                np.random.uniform(0, 1, 2),
            ]
            rhs = physics.double_piston
        else:
            initial = np.r_[np.random.uniform(-1, 1, 3), np.random.uniform(-1, 1, 1)]
            rhs = physics.Diss_Euler_eqs_generalized
        sol = solve_ivp(
            rhs,
            (0, t_out[-1]),
            initial,
            t_eval=t_out,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        if (
            not sol.success
            or sol.y.shape[1] != len(t_out)
            or not np.isfinite(sol.y).all()
        ):
            raise RuntimeError(f"ODE failed for trajectory {j}: {sol.message}")
        states = sol.y[:, : args.points]
        if args.system == "piston":
            x, p, s1, s2 = states
            rows = np.column_stack(
                (
                    x,
                    p / par.M,
                    physics.Ufunc(s1, x, 0)[2],
                    physics.Ufunc(s2, x, 1)[2],
                    p,
                    s1,
                    s2,
                    t_out[: args.points],
                )
            )
        else:
            rows = np.vstack((states, physics.dhdy(states), t_out[: args.points])).T
        sl = slice(j * (args.points - 1), (j + 1) * (args.points - 1))
        begin[sl], end[sl] = rows[:-1], rows[1:]
        if j % args.print_every == 0 or j == trajectories - 1:
            print(f"Trajectory {j+1}/{trajectories}", flush=True)
    with path.open("wb") as stream:
        if args.system == "piston":
            np.save(stream, np.asarray([begin, end], dtype=object), allow_pickle=True)
        else:
            np.savez(stream, Begin_pts=begin.T, End_pts=end.T)
    write_metadata(
        path.with_suffix(".json"),
        operation="generate",
        system=args.system,
        trajectories=trajectories,
        points=args.points,
        pairs=count,
        seed=args.seed,
        h=par.h,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        data_sha256=sha256(path),
        elapsed_seconds=time.perf_counter() - started,
    )
    print(f"Saved {count:,} pairs: {path}")
