"""Constitutive laws and ODE from the original experiment."""
import numpy as np
import tensorflow as tf
from .parameters import A, L, M, nu, nu_nonlinear, N_times_K_B, chat


def Ufunc(S, X, ind):
    Area = A[ind]
    if ind == 0:
        V = Area * (L - X)
        dVdX = -Area
    else:
        V = Area * (L + X)
        dVdX = Area
    U = (chat * V) ** (-2.0 / 3) * np.exp(S / N_times_K_B)
    dUdV = -2.0 / 3 * chat ** (-2.0 / 3) * V ** (-5.0 / 3) * np.exp(S / N_times_K_B)
    dUdX = dUdV * dVdX
    dUdS = U / N_times_K_B
    return [U, dUdX, dUdS]


def compute_energy(y):
    x = y[0]
    p = y[1]
    S1 = y[2]
    S2 = y[3]
    KE = 1 / 2 * p**2 / M
    [U1, dU1, dUdS1] = Ufunc(S1, x, 0)
    [U2, dU2, dUdS2] = Ufunc(S2, x, 1)
    return KE + U1 + U2


def compute_G(y):
    x = y[0]
    v = y[1]
    T1 = y[2]
    T2 = y[3]
    dGdx = 0
    G = 0.5 * M * v**2
    dGdv = M * v
    for ind in range(2):
        Area = A[ind]
        if ind == 0:
            V = Area * (L - x)
            T = T1
            dVdx = -Area
            if isinstance(T1, np.ndarray):
                S1 = N_times_K_B * np.log(N_times_K_B * (chat * V) ** (2.0 / 3) * T1)
                dGdT1 = S1
            else:
                S1 = N_times_K_B * tf.math.log(
                    N_times_K_B * (chat * V) ** (2.0 / 3) * T1
                )
                dGdT1 = S1
        else:
            V = Area * (L + x)
            T = T2
            dVdx = Area
            if isinstance(T2, np.ndarray):
                S2 = N_times_K_B * np.log(N_times_K_B * (chat * V) ** (2.0 / 3) * T2)
                dGdT2 = S2
            else:
                S2 = N_times_K_B * tf.math.log(
                    N_times_K_B * (chat * V) ** (2.0 / 3) * T2
                )
                dGdT2 = S2
        dGdV = 2.0 / 3 * N_times_K_B * T / V
        dGdx += dGdV * dVdx
        if isinstance(T2, np.ndarray):
            G += (
                N_times_K_B * T * np.log(N_times_K_B * (chat * V) ** (2.0 / 3) * T)
                - N_times_K_B * T
            )
        else:
            G += (
                N_times_K_B * T * tf.math.log(N_times_K_B * (chat * V) ** (2.0 / 3) * T)
                - N_times_K_B * T
            )
    if isinstance(T2, np.ndarray):
        dGdy = np.vstack([dGdx, dGdv, dGdT1, dGdT2])
    else:
        dGdy = tf.stack([dGdx, dGdv, dGdT1, dGdT2])
    return [G, dGdy]


def lam_friction_func(T, x, v, ind):
    lam = nu[ind] + nu_nonlinear[ind] * v**2 * (1 + 0.1 * x**2 * T**2)
    return lam


def double_piston(t, y):
    x = y[0]
    p = y[1]
    S1 = y[2]
    S2 = y[3]
    v = p / M
    [U1, dUdx1, T1] = Ufunc(S1, x, 0)
    [U2, dUdx2, T2] = Ufunc(S2, x, 1)
    lam1 = lam_friction_func(T1, x, v, 0)
    lam2 = lam_friction_func(T2, x, v, 1)
    Fint1 = -dUdx1
    Fint2 = -dUdx2
    dxdt = v
    dpdt = Fint1 + Fint2 - (lam1 + lam2) * v
    dS1dt = lam1 / T1 * v**2
    dS2dt = lam2 / T2 * v**2
    dy = np.hstack([dxdt, dpdt, dS1dt, dS2dt])
    return dy
