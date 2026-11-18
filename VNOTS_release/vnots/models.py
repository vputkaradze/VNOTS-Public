"""Construct original model wrappers and their constitutive anchors."""
import importlib
import numpy as np
import tensorflow as tf
from .config import EXPERIMENTS, case_flags


def setup(system, seed=0):
    cfg = EXPERIMENTS[system]
    tf.keras.backend.set_floatx(cfg.dtype)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf.as_dtype(cfg.dtype)


def build(system, case):
    module = importlib.import_module(f"vnots.{system}.model")
    flags = case_flags(system, case)
    dtype = tf.as_dtype(EXPERIMENTS[system].dtype)
    g = module.build_model_G()
    f = module.build_model_F()
    sample = tf.constant(
        [[0.0, 0.0, 1.0, 1.0]] * 2
        if system == "piston"
        else [[0.0, 0.0, 0.0, 1.0]] * 2,
        dtype=dtype,
    )
    if system == "piston":
        f2 = module.build_model_F()
        model = module.G_and_F_model(
            g, f, f2, known_G=flags[0], known_F1=flags[1], known_F2=flags[2]
        )
        model([sample, sample])
        networks = (g, f, f2)
    else:
        model = module.GAndF(g, f, known_G=flags[0], known_F=flags[1], dtype=dtype)
        model(sample)
        networks = (g, f)
    variables = [
        v
        for known, net in zip(flags, networks)
        if not known
        for v in net.trainable_variables
    ]
    return model, variables


def anchors(system):
    """Match the original scripts' grids, targets, and random stationary anchors."""
    physics = importlib.import_module(f"vnots.{system}.physics")
    dtype = tf.as_dtype(EXPERIMENTS[system].dtype)
    if system == "piston":
        q = tf.random.uniform([200, 1], -1.0, 1.0)
        t1 = tf.random.uniform([200, 1], 0.5, 2.0)
        t2 = tf.random.uniform([200, 1], 0.5, 2.0)
        stationary = tf.concat([q, tf.zeros([200, 1]), t1, t2], axis=1)
        t1, t2 = np.meshgrid(np.linspace(0.5, 2.0, 15), np.linspace(0.5, 2.0, 15))
        z = np.zeros(t1.size)
        grid = tf.constant(np.stack([z, z, t1.ravel(), t2.ravel()], axis=1), dtype)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(grid)
            _, grad = physics.compute_G(tf.transpose(grid))
            grad = tf.transpose(grad)
            s1, s2 = grad[:, 2], grad[:, 3]
        c1, c2 = tape.gradient(s1, grid)[:, 2], tape.gradient(s2, grid)[:, 3]
        del tape
        return grid, stationary, c1, c2
    thermal = tf.concat(
        [
            tf.zeros((100, 3), dtype),
            tf.reshape(
                tf.linspace(tf.constant(0.1, dtype), tf.constant(2.0, dtype), 100),
                (-1, 1),
            ),
        ],
        axis=1,
    )
    axis = tf.linspace(tf.constant(-0.5, dtype), tf.constant(0.5, dtype), 20)
    vx, vy, vz = tf.meshgrid(axis, axis, axis)
    om = tf.reshape(tf.stack([vx, vy, vz], axis=-1), (-1, 3))
    kinetic = tf.concat([om, tf.ones((8000, 1), dtype) * 2], axis=1)
    grid = tf.concat([kinetic, thermal], axis=0)
    _, grad, g = physics.compute_G(grid)
    eps = tf.constant(1e-3, dtype)
    delta = tf.constant([0.0, 0.0, 0.0, 1.0], dtype) * eps
    plus, minus = physics.compute_G(grid + delta)[1], physics.compute_G(grid - delta)[1]
    return (
        grid,
        tf.reshape(g, (-1,)),
        grad[:, :3],
        grad[:, 3],
        (plus[:, 3] - minus[:, 3]) / (2 * eps),
    )
