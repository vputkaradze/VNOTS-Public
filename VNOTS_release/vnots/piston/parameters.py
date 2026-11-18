"""Physical parameters from the original experiment."""
import numpy as np
import tensorflow as tf

h = 0.01
A = np.array([2, 1])
L = 2
M = 1
nu = 0.02 * np.array([1.0, 2.0])
nu_nonlinear = 1.0 * np.array([2.0, 1.0])
N_times_K_B = 1
chat = 1
