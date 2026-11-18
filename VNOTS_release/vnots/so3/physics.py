"""Constitutive laws and ODE from the original experiment."""
import numpy as np
import tensorflow as tf
from .parameters import (
    Inertia,
    Inertia_tf,
    InvI,
    nu,
    nonlinear_nu,
    gamma,
    A,
    eps_tensor,
    U0,
)


def compute_num_pts(X):
    if len(tf.squeeze(X).shape) > 1:
        y = X.shape
        nt = y[0]
    else:
        y = X
        nt = 1
    return nt


def hatmap_tf(v):
    nt = compute_num_pts(v)
    if nt > 1:
        vhat = tf.tensordot(v, eps_tensor, axes=[1, 0])
    else:
        vhat = tf.tensordot(v, eps_tensor, axes=[0, 0])
    return tf.squeeze(vhat)


def compute_num_dim(X):
    if isinstance(X, np.ndarray):
        num_dim = len(np.squeeze(X).shape)
    else:
        num_dim = len(tf.squeeze(X).shape)
    return num_dim


def Friction_function(dhdx):
    num_dim = compute_num_dim(dhdx)
    if num_dim > 1:
        Omega = dhdx[:, 0:3]
        T = dhdx[:, 3]
    else:
        Omega = dhdx[0:3]
        T = dhdx[3]
    if isinstance(dhdx, np.ndarray):
        friction_values = nu * Omega + nonlinear_nu * np.matmul(
            A.T, np.tanh(np.matmul(A, Omega))
        )
    else:
        friction_values = nu * Omega + nonlinear_nu * tf.linalg.matvec(
            A.T, tf.math.tanh(tf.linalg.matvec(A, Omega))
        )
    return friction_values


def Diss_Euler_eqs_generalized(t, y):
    mu = y[0:3]
    S = y[3]
    dh_dmu_dS = dhdy(y)
    Omega = dh_dmu_dS[0:3]
    T = dh_dmu_dS[3]
    F = -Friction_function(dh_dmu_dS)
    dmudt = -np.cross(Omega, mu) + F
    dSdt = -np.dot(F, Omega) / T
    return np.hstack([dmudt, dSdt])


def compute_energy(y):
    mu = y[0:3]
    S = y[3]
    num_dim = compute_num_dim(y)
    if num_dim > 1:
        T = 1 / 2 * np.exp(gamma * S) * np.sum(np.matmul(InvI, mu) * mu, axis=0)
    else:
        T = 1 / 2 * np.exp(gamma * S) * np.dot(np.matmul(InvI, mu), mu)
    U = U0 * np.exp(gamma * S)
    return T + U


def compute_G(X):
    is_tf = not isinstance(X, np.ndarray)
    if is_tf:
        if len(X.shape) == 1:
            X = tf.expand_dims(X, 0)
        Omega = X[:, 0:3]
        T = X[:, 3]
    elif X.ndim == 1:
        Omega = X[0:3]
        T = X[3]
        single_point = True
    else:
        Omega = X[:, 0:3]
        T = X[:, 3]
        single_point = False

    def get_physics_scalars(Omega_in, T_in):
        s = 1.0
        if is_tf:
            mu_mod = tf.linalg.matvec(Inertia_tf, Omega_in)
            mumod_times_omega = tf.reduce_sum(mu_mod * Omega_in, axis=1)
            disc_raw = tf.math.square(T_in / gamma) - 2.0 * U0 * mumod_times_omega
            disc = tf.maximum(disc_raw, 0.0)
            exp_gamma_S = (T_in / gamma + s * tf.math.sqrt(disc)) / (2.0 * U0)
            S_val = 1.0 / gamma * tf.math.log(exp_gamma_S)
            term_kinetic = 0.5 / exp_gamma_S * mumod_times_omega
            H_val = term_kinetic + exp_gamma_S * U0
            G_val = 1.0 / exp_gamma_S * mumod_times_omega + T_in * S_val - H_val
            return (H_val, G_val, S_val, exp_gamma_S)
        else:
            if single_point:
                mu_mod = np.dot(Inertia, Omega_in)
                mumod_times_omega = np.dot(mu_mod, Omega_in)
            else:
                mu_mod = np.dot(Inertia, Omega_in.T).T
                mumod_times_omega = np.sum(mu_mod * Omega_in, axis=1)
            disc_raw = (T_in / gamma) ** 2 - 2.0 * U0 * mumod_times_omega
            disc = np.maximum(disc_raw, 0.0)
            exp_gamma_S = (T_in / gamma + s * np.sqrt(disc)) / (2.0 * U0)
            S_val = 1.0 / gamma * np.log(exp_gamma_S)
            term_kinetic = 0.5 / exp_gamma_S * mumod_times_omega
            H_val = term_kinetic + exp_gamma_S * U0
            G_val = 1.0 / exp_gamma_S * mumod_times_omega + T_in * S_val - H_val
            return (H_val, G_val, S_val, exp_gamma_S)

    if is_tf:
        with tf.GradientTape() as tape:
            tape.watch(X)
            O_tf = X[:, 0:3]
            T_tf = X[:, 3]
            (H_val, G_val, S_val, exp_S) = get_physics_scalars(O_tf, T_tf)
        grad_G = tape.gradient(G_val, X)
        if X.shape[0] == 1:
            grad_G = tf.concat([grad_G, grad_G], axis=0)
        return [H_val, grad_G, G_val]
    else:
        (H_val, G_val, S_val, exp_S) = get_physics_scalars(Omega, T)
        if single_point:
            mu = 1.0 / exp_S * np.dot(Inertia, Omega)
            grad_G = np.append(mu, S_val)
        else:
            mu = 1.0 / exp_S[:, None] * np.dot(Inertia, Omega.T).T
            grad_G = np.hstack([mu, S_val[:, None]])
        return [H_val, grad_G, G_val]


def dhdy(y):
    num_dim = compute_num_dim(y)
    if num_dim > 1:
        mu = y[0:3, :]
        S = y[3, :]
        dhdmu = np.matmul(InvI, mu) * np.exp(gamma * S)
        dhdS = (
            gamma
            * np.exp(gamma * S)
            * (0.5 * np.sum(np.matmul(InvI, mu) * mu, axis=0) + U0)
        )
        out = np.vstack([dhdmu, dhdS])
    else:
        mu = y[0:3]
        S = y[3]
        dhdmu = np.matmul(InvI, mu) * np.exp(gamma * S)
        dhdS = gamma * np.exp(gamma * S) * (0.5 * np.dot(np.matmul(InvI, mu), mu) + U0)
        out = np.hstack([dhdmu, dhdS])
    return out
