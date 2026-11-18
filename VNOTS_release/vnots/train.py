"""Train one case with the original loss, architecture, and learning-rate schedule."""
import csv
import importlib
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from .config import EXPERIMENTS, case_flags
from .models import setup, build, anchors
from .util import data_path, load_data, sha256, write_metadata


def train(args):
    cfg = EXPERIMENTS[args.system]
    flags = case_flags(args.system, args.case)
    if all(flags):
        print(
            "Case 2 is fully known: no training or weights are needed. Run reconstruct."
        )
        return
    updates = cfg.updates if args.updates is None else args.updates
    if updates < 1 or args.batch_size < 2:
        raise ValueError("Use at least one update and batch size >= 2.")
    dtype = setup(args.system, args.seed)
    module = importlib.import_module(f"vnots.{args.system}.model")
    begin, end = load_data(args.system, data_path(args))
    columns = slice(0, 4) if args.system == "piston" else slice(4, 8)
    begin, end = tf.constant(begin[:, columns], dtype), tf.constant(
        end[:, columns], dtype
    )
    targets = anchors(args.system)
    model, variables = build(args.system, args.case)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    label = cfg.labels[args.case]
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    weight_path = run / f"{cfg.weight_prefix}{label}.h5"
    metadata_path = run / f"train_{label}.json"
    metadata = dict(
        operation="train",
        system=args.system,
        case=args.case,
        label=label,
        seed=args.seed,
        dtype=cfg.dtype,
        updates=updates,
        batch_size=args.batch_size,
        lr_initial=1e-3,
        lr_final=1e-3 * 10 ** (-cfg.decades) if updates > 1 else 1e-3,
        pairs=int(len(begin)),
        data_file=data_path(args).name,
        data_sha256=sha256(data_path(args)),
        h=0.01,
    )
    write_metadata(metadata_path, status="running", **metadata)

    @tf.function
    def step(y0, yf):
        with tf.GradientTape() as tape:
            if args.system == "piston":
                total, dyn, _ = module.compute_loss_detailed(
                    yf, y0, model, 1.0, *targets
                )
            else:
                total, dyn = module.loss_total(
                    y0,
                    yf,
                    model,
                    tf.constant(0.01, dtype),
                    tf.constant(1.0, dtype),
                    tf.constant(1.0, dtype),
                    *targets,
                    tf.constant(1.0, dtype),
                    tf.constant(1.0, dtype),
                    tf.constant(0.01, dtype),
                )
            tf.debugging.assert_all_finite(total, "Nonfinite training loss")
        gradients = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        for grad, _ in pairs:
            tf.debugging.assert_all_finite(grad, "Nonfinite training gradient")
        optimizer.apply_gradients(pairs)
        return total, dyn

    started = time.perf_counter()
    with (run / f"loss_{label}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["update", "loss", "dynamical_loss", "learning_rate"])
        for i in range(updates):
            lr = 1e-3 * 10 ** (-cfg.decades * i / max(1, updates - 1))
            optimizer.learning_rate.assign(lr)
            idx = tf.random.uniform([args.batch_size], 0, len(begin), dtype=tf.int32)
            total, dyn = step(tf.gather(begin, idx), tf.gather(end, idx))
            values = float(total.numpy()), float(dyn.numpy())
            writer.writerow([i + 1, *values, lr])
            if i % args.print_every == 0 or i == updates - 1:
                stream.flush()
                print(
                    f"Update {i+1}/{updates} | loss={values[0]:.6e} | dynamics={values[1]:.6e} | lr={lr:.3e}",
                    flush=True,
                )
    model.save_weights(str(weight_path))
    write_metadata(
        metadata_path,
        status="complete",
        **metadata,
        elapsed_seconds=time.perf_counter() - started,
        weights_sha256=sha256(weight_path),
        final_loss=values[0],
    )
    print(f"Saved: {weight_path}")
