"""Original network architecture, variational residuals, and anchor losses."""
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from .parameters import h, M
from .physics import compute_G, lam_friction_func


def build_model_G(num_layers: int = 4, size_layer: int = 128) -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.Input((4,)))
    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(
                size_layer, activation="tanh", kernel_initializer="glorot_normal"
            )
        )
    model.add(keras.layers.Dense(1))
    return model


def build_model_F(num_layers: int = 3, size_layer: int = 128) -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.Input((4,)))
    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(
                size_layer, activation="tanh", kernel_initializer="glorot_normal"
            )
        )
    model.add(keras.layers.Dense(1))
    return model


def G_gradients_nn(model_G: keras.Model, x: tf.Tensor) -> tf.Tensor:
    """Returns dG/d(q,v,T1,T2) from the NN scalar output G(x)."""
    with tf.GradientTape() as tape:
        tape.watch(x)
        G = model_G(x)
    return tape.gradient(G, x)


def read_variables(yf: tf.Tensor, y0: tf.Tensor, h_val: float):
    """
    y0, yf contain at least 4 columns: [q, v, T1, T2, ...]
    """
    (q0, v0, T1_0, T2_0) = (y0[:, 0], y0[:, 1], y0[:, 2], y0[:, 3])
    (qf, vf, T1_f, T2_f) = (yf[:, 0], yf[:, 1], yf[:, 2], yf[:, 3])
    v_act = (qf - q0) / h_val
    return (q0, qf, v0, vf, v_act, T1_0, T2_0, T1_f, T2_f)


class G_and_F_model(tf.keras.Model):
    def __init__(
        self,
        model_G: keras.Model,
        model_F1: keras.Model,
        model_F2: keras.Model,
        known_G: bool = False,
        known_F1: bool = False,
        known_F2: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.model_G = model_G
        self.model_F1 = model_F1
        self.model_F2 = model_F2
        self.known_G = known_G
        self.known_F1 = known_F1
        self.known_F2 = known_F2

    def call(self, inputs, training=None):
        (yf, y0) = inputs
        yf = tf.convert_to_tensor(yf, dtype=tf.float32)
        y0 = tf.convert_to_tensor(y0, dtype=tf.float32)
        (q0, qf, v0, vf, v_act, T1_0, T2_0, T1_f, T2_f) = read_variables(yf, y0, h)
        v_k_eff = v0
        y0_in = tf.stack([q0, v_k_eff, T1_0, T2_0], axis=1)
        yf_in = yf[:, 0:4]
        if self.known_G:
            (_, grad0_raw) = compute_G(tf.transpose(y0_in))
            (_, gradf_raw) = compute_G(tf.transpose(yf_in))
            grad_G_0 = tf.transpose(grad0_raw)
            grad_G_f = tf.transpose(gradf_raw)
        else:
            grad_G_0 = G_gradients_nn(self.model_G, y0_in)
            grad_G_f = G_gradients_nn(self.model_G, yf_in)
        if self.known_F1:
            Ffr1_0 = -lam_friction_func(T1_0, q0, v_k_eff, 0) * v_k_eff
            Ffr1_f = -lam_friction_func(T1_f, qf, vf, 0) * vf
        else:
            Ffr1_0 = -tf.reshape(self.model_F1(y0_in), [-1]) ** 2 * v_k_eff
            Ffr1_f = -tf.reshape(self.model_F1(yf_in), [-1]) ** 2 * vf
        if self.known_F2:
            Ffr2_0 = -lam_friction_func(T2_0, q0, v_k_eff, 1) * v_k_eff
            Ffr2_f = -lam_friction_func(T2_f, qf, vf, 1) * vf
        else:
            Ffr2_0 = -tf.reshape(self.model_F2(y0_in), [-1]) ** 2 * v_k_eff
            Ffr2_f = -tf.reshape(self.model_F2(yf_in), [-1]) ** 2 * vf
        return (grad_G_0, grad_G_f, Ffr1_0, Ffr2_0, Ffr1_f, Ffr2_f, v_k_eff)


def compute_eqs(yf, y0, G_F_model: G_and_F_model):
    (q0, qf, v0, vf, v_act, T1_0, T2_0, T1_f, T2_f) = read_variables(yf, y0, h)
    (grad_G_0, grad_G_f, Ffr1_0, Ffr2_0, Ffr1_f, Ffr2_f, v_k_eff) = G_F_model([yf, y0])
    (p0, S1_0, S2_0) = (grad_G_0[:, 1], grad_G_0[:, 2], grad_G_0[:, 3])
    (pf, S1_f, S2_f) = (grad_G_f[:, 1], grad_G_f[:, 2], grad_G_f[:, 3])
    eq_q = v_act - v0
    eq_p = (pf - p0) / h - (grad_G_f[:, 0] + Ffr1_f + Ffr2_f)
    eq_S1 = T1_0 * (S1_f - S1_0) / h + Ffr1_0 * v_k_eff
    eq_S2 = T2_0 * (S2_f - S2_0) / h + Ffr2_0 * v_k_eff
    return (eq_q, eq_p, eq_S1, eq_S2)


def compute_loss_detailed(
    yf,
    y0,
    G_F_model: G_and_F_model,
    weight_anchor: float,
    IC_anchor: tf.Tensor,
    Global_V0_anchor: tf.Tensor,
    G_TT1_exact: tf.Tensor,
    G_TT2_exact: tf.Tensor,
):
    (eq_q, eq_p, eq_S1, eq_S2) = compute_eqs(yf, y0, G_F_model)
    phys_loss = (
        tf.reduce_mean(eq_q**2)
        + tf.reduce_mean(eq_p**2)
        + (tf.reduce_mean(eq_S1**2) + tf.reduce_mean(eq_S2**2))
    )
    anc_loss = 0.0
    if not G_F_model.known_G and weight_anchor > 0:
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(IC_anchor)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(IC_anchor)
                t1.watch(Global_V0_anchor)
                G_pred = G_F_model.model_G(IC_anchor)
                G_global = G_F_model.model_G(Global_V0_anchor)
            grad_G = t1.gradient(G_pred, IC_anchor)
            grad_G_global = t1.gradient(G_global, Global_V0_anchor)
            p_global_pred = grad_G_global[:, 1]
            p_pred = grad_G[:, 1]
            S1_p = grad_G[:, 2]
            S2_p = grad_G[:, 3]
        loss_p_global = tf.reduce_mean(tf.abs(p_global_pred) ** 2)
        dp_dv = t2.gradient(p_pred, IC_anchor)[:, 1]
        G_TT1_pred = t2.gradient(S1_p, IC_anchor)[:, 2]
        G_TT2_pred = t2.gradient(S2_p, IC_anchor)[:, 3]
        anc_loss_T = tf.reduce_mean(
            tf.abs(G_TT1_pred - G_TT1_exact) ** 2
            + tf.abs(G_TT2_pred - G_TT2_exact) ** 2
        )
        v_anc_loss = tf.reduce_mean(tf.abs(dp_dv - M) ** 2)
        anc_loss = anc_loss_T + v_anc_loss + loss_p_global
    total_loss = phys_loss + weight_anchor * anc_loss
    return (total_loss, phys_loss, weight_anchor * anc_loss)
