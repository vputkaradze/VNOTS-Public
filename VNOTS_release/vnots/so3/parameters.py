"""Physical parameters from the original experiment."""
import numpy as np
import tensorflow as tf

h = 0.01
U0 = 1
Inertia = np.diag([1.0, 2.0, 3.0])
Inertia_tf = tf.Variable(Inertia, dtype=tf.float64)
InvI = np.linalg.inv(Inertia)
nu = 0.01
nonlinear_nu = 0.01
gamma = 1.0
A = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
eps_tensor = tf.Variable(
    [
        [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
        [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
    ],
    dtype=tf.float64,
)
