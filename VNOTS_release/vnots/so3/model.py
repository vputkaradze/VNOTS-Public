"""Original network architecture, variational residuals, and anchor losses."""
import tensorflow as tf
from tensorflow import keras
from .physics import compute_G, Friction_function, hatmap_tf


def build_model_G(num_layers: int = 4, size_layer: int = 128) -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.Input((4,)))
    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(
                size_layer,
                activation="tanh",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
            )
        )
    model.add(keras.layers.Dense(1))
    return model


def build_model_F(num_layers: int = 4, size_layer: int = 64) -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.Input((4,)))
    for _ in range(num_layers):
        model.add(
            keras.layers.Dense(
                size_layer, activation="tanh", kernel_initializer="glorot_normal"
            )
        )
    model.add(keras.layers.Dense(6))
    return model


def G_gradients_nn(model_G: keras.Model, x: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        G = model_G(x)
    return tape.gradient(G, x)


def compute_S_matrix(params: tf.Tensor, dtype) -> tf.Tensor:
    Nvec = params[:, 0:3]
    f = params[:, 3:6]
    Nhat = hatmap_tf(Nvec)
    Q = tf.linalg.expm(Nhat)
    QT = tf.transpose(Q, perm=[0, 2, 1])
    D = tf.linalg.diag(f**2 + tf.constant(1e-06, dtype))
    return QT @ (D @ Q)


class GAndF(tf.keras.Model):
    def __init__(
        self, model_G, model_F, *, known_G: bool, known_F: bool, dtype, name=None
    ):
        super().__init__(name=name)
        self.model_G = model_G
        self.model_F = model_F
        self.known_G = bool(known_G)
        self.known_F = bool(known_F)
        self.dtype_ = dtype

    def call(self, y, training=None):
        y = tf.convert_to_tensor(y, dtype=self.dtype_)
        if self.known_G:
            (H, grad_G, G) = compute_G(y)
            G = tf.reshape(G, (-1, 1))
        else:
            G = self.model_G(y)
            grad_G = G_gradients_nn(self.model_G, y)
            H = tf.reduce_sum(y * grad_G, axis=1) - tf.reshape(G, (-1,))
        if self.known_F:
            F = -Friction_function(y)
        else:
            Omega = y[:, 0:3]
            out = self.model_F(y)
            S = compute_S_matrix(out, self.dtype_)
            F = -tf.linalg.matvec(S, Omega)
        return (H, grad_G, G, F)


def three_products(mu: tf.Tensor, Omega: tf.Tensor) -> tf.Tensor:
    term1 = tf.linalg.cross(Omega, tf.linalg.cross(mu, Omega))
    norm2 = tf.reduce_sum(Omega**2, axis=1, keepdims=True)
    term2 = mu * norm2
    return term1 - term2


@tf.function
def compute_eqs(yf, y0, model: GAndF, h, include_quadratic):
    (_, dG0, _, F0) = model(y0, training=False)
    (_, dGf, _, Ff) = model(yf, training=False)
    Omega0 = y0[:, 0:3]
    Omegaf = yf[:, 0:3]
    T0 = y0[:, 3]
    mu0 = dG0[:, 0:3]
    muf = dGf[:, 0:3]
    S0 = dG0[:, 3]
    Sf = dGf[:, 3]
    torque0 = tf.linalg.cross(Omega0, mu0)
    torquef = tf.linalg.cross(Omegaf, muf)
    quad_term = (
        h
        / tf.constant(4.0, model.dtype_)
        * (three_products(muf, Omegaf) - three_products(mu0, Omega0))
    )
    mu_eq = (
        (muf - mu0) / h + 0.5 * (torquef + torque0) - include_quadratic * quad_term - Ff
    )
    production0 = tf.reduce_sum(Omega0 * F0, axis=1)
    S_eq = (Sf - S0) / h + production0 / T0
    return (mu_eq, S_eq)


def gauge_mu_zero_at_Omega0(model_G: keras.Model, dtype, n: int = 64) -> tf.Tensor:
    T_vals = tf.random.uniform((n, 1), minval=0.1, maxval=2.0, dtype=dtype)
    Omega0 = tf.zeros((n, 3), dtype=dtype)
    y = tf.concat([Omega0, T_vals], axis=1)
    with tf.GradientTape() as tape:
        tape.watch(y)
        G = model_G(y)
    dG = tape.gradient(G, y)
    mu0 = dG[:, 0:3]
    return tf.reduce_mean(mu0**2)


def friction_scale_gauge(
    model_F: keras.Model, dtype, target_trace: float = 0.075, n: int = 64
) -> tf.Tensor:
    T_vals = tf.random.uniform((n, 1), minval=0.1, maxval=2.0, dtype=dtype)
    Omega0 = tf.zeros((n, 3), dtype=dtype)
    y = tf.concat([Omega0, T_vals], axis=1)
    out = model_F(y)
    S = compute_S_matrix(out, dtype)
    tr = tf.linalg.trace(S)
    return tf.reduce_mean((tr - tf.constant(target_trace, dtype)) ** 2)


@tf.function
def loss_total(
    y0b,
    yfb,
    model: GAndF,
    h,
    include_quadratic,
    scale_entropy_error,
    IC_anchor,
    G_exact,
    mu_exact,
    S_exact,
    d2G_dT2_exact,
    weight_anchor,
    weight_mu0,
    weight_F_gauge,
):
    (mu_eq, S_eq) = compute_eqs(yfb, y0b, model, h, include_quadratic)
    loss_phys = tf.reduce_mean(mu_eq**2) + scale_entropy_error * tf.reduce_mean(
        S_eq**2
    )
    loss = loss_phys
    if not model.known_G:
        with tf.GradientTape() as t2:
            t2.watch(IC_anchor)
            with tf.GradientTape() as t1:
                t1.watch(IC_anchor)
                G_pred = model.model_G(IC_anchor)
            dG_pred = t1.gradient(G_pred, IC_anchor)
            mu_pred = dG_pred[:, 0:3]
            S_pred = dG_pred[:, 3]
        d2 = t2.gradient(S_pred, IC_anchor)
        d2G_dT2_pred = d2[:, 3]
        G_pred_flat = tf.squeeze(G_pred)
        loss_G = tf.reduce_mean((G_pred_flat - G_exact) ** 2)
        loss_mu = tf.reduce_mean((mu_pred - mu_exact) ** 2)
        loss_C = tf.reduce_mean((d2G_dT2_pred - d2G_dT2_exact) ** 2)
        loss_S = tf.reduce_mean((S_pred - S_exact) ** 2)
        loss += weight_anchor * (loss_G + loss_mu + loss_C + loss_S)
        loss += weight_mu0 * gauge_mu_zero_at_Omega0(model.model_G, model.dtype_, n=64)
    if not model.known_F:
        loss += weight_F_gauge * friction_scale_gauge(
            model.model_F, model.dtype_, target_trace=0.075, n=64
        )
    return (loss, loss_phys)
