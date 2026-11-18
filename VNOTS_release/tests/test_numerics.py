"""Checks of constitutive identities, energy balance, and checkpoint architectures."""
import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from vnots.config import EXPERIMENTS
from vnots.models import setup, build
from vnots.util import load_data


def finite_gradient(fun, x, eps=1e-6):
    eye = np.eye(len(x)) * eps
    return np.array([(fun(x + d) - fun(x - d)) / (2 * eps) for d in eye])


class ConstitutiveTests(unittest.TestCase):
    def test_cluster_pair_defaults(self):
        self.assertEqual(EXPERIMENTS["piston"].trajectories * 20, 100000)
        self.assertEqual(EXPERIMENTS["so3"].trajectories * 20, 10000)

    def test_piston_energy_and_legendre_transform(self):
        from vnots.piston import physics as p

        y = np.array([0.2, -0.3, 0.4, 0.6])
        gradient = finite_gradient(p.compute_energy, y)
        self.assertAlmostEqual(float(gradient @ p.double_piston(0, y)), 0.0, places=8)
        x = np.array([y[0], y[1], p.Ufunc(y[2], y[0], 0)[2], p.Ufunc(y[3], y[0], 1)[2]])
        g, grad = p.compute_G(x[:, None])
        np.testing.assert_allclose(grad[1:, 0], y[1:], atol=1e-12)
        self.assertAlmostEqual(
            float(x[1:] @ grad[1:, 0] - g[0]), float(p.compute_energy(y)), places=12
        )
        states = np.column_stack([y, y + np.array([0.1, 0.2, 0.1, -0.1])])
        np.testing.assert_allclose(
            p.compute_energy(states), [p.compute_energy(v) for v in states.T]
        )

    def test_so3_energy_and_legendre_transform(self):
        from vnots.so3 import physics as p

        y = np.array([-0.25, 0.5, 0.5, 0.5])
        gradient = finite_gradient(p.compute_energy, y)
        np.testing.assert_allclose(gradient, p.dhdy(y), atol=1e-9)
        self.assertAlmostEqual(
            float(gradient @ p.Diss_Euler_eqs_generalized(0, y)), 0.0, places=8
        )
        h, grad, g = p.compute_G(p.dhdy(y))
        np.testing.assert_allclose(grad, y, atol=1e-12)
        self.assertAlmostEqual(float(p.dhdy(y) @ grad - g), float(h), places=12)
        states = np.column_stack([y, y + np.array([0.1, -0.2, 0.1, 0.2])])
        np.testing.assert_allclose(
            p.compute_energy(states), [p.compute_energy(v) for v in states.T]
        )
        omega = p.dhdy(y)[:3]
        self.assertLess(float(-p.Friction_function(p.dhdy(y)) @ omega), 0.0)

    def test_original_network_counts(self):
        for system in ("piston", "so3"):
            setup(system)
            model, _ = build(system, 3)
            self.assertEqual(model.model_G.count_params(), 50305)
            if system == "piston":
                self.assertEqual(model.model_F1.count_params(), 33793)
                self.assertEqual(model.model_F2.count_params(), 33793)
            else:
                self.assertEqual(model.model_F.count_params(), 13190)
                x = tf.constant([[0.1, 0.2, 0.1, 2.0]] * 2, tf.float64)
                h, grad, g, force = model(x)
                np.testing.assert_allclose(
                    h.numpy(),
                    np.sum(x.numpy() * grad.numpy(), axis=1) - g.numpy().ravel(),
                )
                self.assertTrue(
                    np.all(np.sum(force.numpy() * x.numpy()[:, :3], axis=1) <= 0)
                )

    def test_dataset_validation(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            begin = np.zeros((9, 3))
            end = begin.copy()
            begin[7] = end[7] = 1
            end[-1] = 0.1
            np.savez(path, Begin_pts=begin, End_pts=end)
            with self.assertRaisesRegex(ValueError, "h=0.01"):
                load_data("so3", path)


if __name__ == "__main__":
    unittest.main()
