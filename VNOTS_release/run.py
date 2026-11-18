#!/usr/bin/env python3
"""Generate, train, reconstruct, and plot the V-NOTS experiments."""
import argparse
import os
from pathlib import Path
from vnots.config import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=EXPERIMENTS)
    parser.add_argument(
        "command", choices=("generate", "check-data", "train", "reconstruct", "plot")
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Data, weights, logs, and results directory (default runs/SYSTEM).",
    )
    parser.add_argument("--data", type=Path, help="Optional existing dataset path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trajectories", type=int, help="Generation: 5000 piston / 500 SO3 by default."
    )
    parser.add_argument(
        "--points",
        type=int,
        default=21,
        help="Generation: retained points per trajectory.",
    )
    parser.add_argument(
        "--case",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Training case; defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        help="Training: exact update count (default 100002 piston / 50002 SO3).",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Reconstruction: time steps; saves steps+1 samples.",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Reconstruction/plotting: comma-separated case ids, or all.",
    )
    parser.add_argument(
        "--residual-tol",
        type=float,
        help="Reconstruction: max accepted residual (1e-4 piston / 1e-8 SO3).",
    )
    parser.add_argument("--print-every", type=int, default=1000)
    args = parser.parse_args()
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    args.run_dir = (args.run_dir or Path("runs") / args.system).resolve()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.command == "check-data":
        from vnots.util import inspect_data

        inspect_data(args)
    elif args.command == "generate":
        from vnots.data import generate

        generate(args)
    elif args.command == "train":
        from vnots.train import train

        train(args)
    elif args.command == "reconstruct":
        from vnots.reconstruct import reconstruct

        reconstruct(args)
    else:
        from vnots.plot import plot

        plot(args)


if __name__ == "__main__":
    main()
