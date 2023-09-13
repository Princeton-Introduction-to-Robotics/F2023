from unittest import TestCase
from mae345.quadrotor import Quadrotor
import numpy as np
from scipy.spatial.transform import Rotation as R

# This is a second derivation of the dynamics I found that I am verifying against
def other_quad_dynamics(state: np.ndarray, input: np.ndarray, mass: float) -> np.ndarray:
    s = np.sin
    c = np.cos
    t = np.tan

    g = 9.8
    Ix = 2.3951e-5
    Iy = 2.3951e-5
    Iz = 3.2347e-5

    (phi, theta, psi) = state[3:6]
    (x_dot, y_dot, z_dot) = state[6:9]
    (p, q, r) = state[9:]


    return np.array([x_dot,
                     y_dot,
                     z_dot,
                     q * (s(phi) / c(theta)) + r * (c(phi) / c(theta)),
                     q * c(phi) - r * s(phi),
                     p + q * (s(phi) * t(theta)) + r * (c(phi) * t(theta)),
                     -input[0] * (s(phi) * s(psi) + c(phi) * c(psi) * s(theta)) / mass,
                     -input[0] * (s(phi) * c(psi) - c(phi) * s(psi) * s(theta)) / mass,
                     -g + input[0] * c(phi) * c(theta) / mass,
                     (Iy - Iz) * q * r / Ix + input[1] / Ix,
                     (Iz - Ix) * p * r / Iy + input[2] / Iy,
                     (Ix - Iy) * p * q / Iz + input[3] / Iz])



class TestQuadrotor(TestCase):
    def test_dynamics(self):
        np.random.seed(0)

        state = np.random.rand(12)
        input = np.random.rand(4)
        q = Quadrotor(1)

        falling = np.zeros(12)
        falling[8] = -9.8

        self.assertTrue(np.allclose(q.dynamics(np.zeros(12), np.zeros(4)), falling))


        # Just fail the test. The above model is only approximate.
        self.fail()

        self.assertTrue(np.allclose(q.dynamics(state, input),
                                    other_quad_dynamics(state, input, 1)))
