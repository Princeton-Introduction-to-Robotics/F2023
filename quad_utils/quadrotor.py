import numpy as np

import sympy as syp
from sympy.physics.vector import dynamicsymbols

from abc import ABC, abstractmethod

from scipy.spatial.transform import Rotation as Rot
from scipy.integrate import solve_ivp

from typing import Union


# I have found sources quoting different numbers for the Crazyflie inertia parameters. I will list them here for
# reference.

# MIT Parameters:
# Ixx = 2.3951e−5
# Iyy = 2.3951e−5
# Izz = 3.2347e−5
#
# Reference: "Planning and Control for Quadrotor Flight through Cluttered Environments" by Landry pg. 39
#
# UPenn Parameters:
#
# Ixx = 1.43e-5
# Iyy = 1.43e-5
# Izz = 2.89e-5
# m   = 0.03 kg
#
# Reference: MEAM620 Project 1.
#
# Out Parameters:
#
# Ixx = 1.4194e-05
# Iyy = 1.4089e-05
# Izz = 2.9741e-05

class Quadrotor(ABC):
    """
    Abstract class describing the quadrotor dynamical model.

    A class derived from this one must specify the following mechanical properties of the system by overriding the
    corresponding properties: quadrotor mass (``mass``), gravitational constant (``gravity``) and
    inertia tensor (``I``). In addition, a control policy must be specified by overriding the function ``controller``.
    """

    def __init__(self):
        self._invI = np.linalg.inv(self.I)
        self._computed_dynamics_jacobian = False
        self._A = None
        self._B = None

    @property
    @abstractmethod
    def mass(self) -> float:
        """
        The mass of the quadrotor.

        :return: Returns the mass of the quadrotor as a float.
        """
        pass

    @property
    @abstractmethod
    def gravity(self) -> float:
        """
        The gravitational constant used for quadrotor simulations.

        :return: gravitational constant used for quadrotor simulations (typically 9.8 m / s^2).
        """
        pass

    @property
    @abstractmethod
    def I(self) -> np.ndarray:
        """
        The inertia tensor for the quadrotor.

        :return: A 3-by-3 matrix representing the inertia tensor of the quadrotor.
                 This matrix is commonly diagonal.
        """
        pass

    @property
    def invI(self) -> np.ndarray:
        """
        This property returns the inverse of the inertia tensor matrix. It is precomputed
        to speed simulations.

        :return: The inertia tensor inverse.
        """
        return self._invI

    @property
    @abstractmethod
    def max_thrust(self) -> float:
        """
        This property returns the maximum thrust the quadrotor is able to
        command instantaneously (all four rotors combined).
        """
        pass

    @property
    @abstractmethod
    def lift_coeff(self) -> float:
        """
        This property returns the lift coefficient of the quadrotor (commonly k_f).
        """
        pass

    @property
    @abstractmethod
    def drag_coeff(self):
        """
        This property returns the lift coefficient of the quadrotor (commonly k_m).
        """
        pass

    @property
    @abstractmethod
    def arm_length(self):
        """
        This property returns the arm length of the quadrotor (commonly L).
        :return:
        """
        pass

    def dynamics(self, state: np.ndarray, input: np.ndarray) -> np.ndarray:
        """
        This method describes the dynamics of the quadrotor for use in simulations. We use adopt the same notational
        conventions of the listed references.

        References: "Minimum Snap Trajectory Generation and Control for Quadrotors" by Mellinger and Kumar
                    "Trajectory Generation and Control for Quadrotors" by Mellinger


        :param state: An 12-dimensional vector representing the state of the quadrotor in the order
                      [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot p, q, r].
        :param input: A 4-dimensional vector detailing the input to the robot at the current timestep. The variables
                      are, in order, the net body force, followed by the X, Y, and Z body moments.
        :return: An 12-dimensional vector representing the instantaneous change in state of the robot.
        """
        phi = state[3]
        theta = state[4]

        wRb = Rot.from_euler('xyz', -state[[3, 4, 5]]).as_matrix()

        z_b = wRb[2, :]
        z_w = np.array([0, 0, 1])
        w_BW = state[9:]

        # Here we compute the change in Euler angles. See slide 9 here:
        # https://alliance.seas.upenn.edu/~meam620/wiki/index.php?n=Main.Schedule?action=download&upname=04_2018_travelocities.pdf
        euler_dot = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                          [0, np.cos(phi), -np.sin(phi)],
                          [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]]) @ w_BW

        r_ddot = np.array([0, 0, -self.gravity]) + wRb @ np.array([0, 0, input[0] / self.mass])
        w_BW_dot = self.invI @ (input[1:] - np.cross(w_BW, self.I @ w_BW))

        return np.concatenate((state[6:9], euler_dot, r_ddot, w_BW_dot))

    def linearize(self, linearize_state: np.ndarray, linearize_input: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Linearizes the quadrotor dynamics about arbitrary states / inputs.

        The first time this function is called, the relevant Jacobians of the
        dynamics are computed symbolically using SymPy, therefore it is slower.
        The results are stored internally using SymPy's Lambdify functionality,
        so future calls will be sped up significantly.

        :param linearize_state: The state to linearize about as a 12-dimensional
                                vector.

        :param linearize_input: The input to linearize about as a 4-dimensional
                                vector.

        :return: A pair of matrices (A, B), where A is the state matrix and B
        is the input matrix.
        """
        if not self._computed_dynamics_jacobian:
            x, y, z = dynamicsymbols('x y z')
            phi, theta, psi = dynamicsymbols('phi theta psi')
            p, q, r = dynamicsymbols('p q r')

            u1, u2, u3, u4 = syp.symbols('u1 u2 u3 u4')
            t = syp.symbols('t')

            x_dot = syp.diff(x, t)
            y_dot = syp.diff(y, t)
            z_dot = syp.diff(z, t)

            state_list = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
            input_list = [u1, u2, u3, u4]
            state = syp.Matrix(state_list)
            input = syp.Matrix(input_list)

            d = syp.Matrix([x, y, z])
            d_dot = syp.diff(d, t)
            w_BW = syp.Matrix([p, q, r])

            eTb = syp.Matrix([[1, syp.sin(phi) * syp.tan(theta), syp.cos(phi) * syp.tan(theta)],
                              [0, syp.cos(phi), -syp.sin(phi)],
                              [0, syp.sin(phi) / syp.cos(theta), syp.cos(phi) / syp.cos(theta)]])

            R = syp.rot_axis3(psi) * syp.rot_axis2(theta) * syp.rot_axis1(phi)
            I = syp.Matrix(self.I)

            d_ddot = syp.Matrix([0, 0, -self.gravity]) + R * syp.Matrix([0, 0, u1 / self.mass])
            euler_dot = eTb * syp.Matrix([p, q, r])
            w_BW_dot = syp.Matrix(self.invI) * (-w_BW.cross(I * w_BW) + syp.Matrix([u2, u3, u4]))

            dynamics = syp.Matrix([d_dot, euler_dot, d_ddot, w_BW_dot])

            A_helper = syp.lambdify(state_list + input_list,
                                    syp.simplify(dynamics.jacobian(state)))

            B_helper = syp.lambdify(state_list + input_list,
                                    syp.simplify(dynamics.jacobian(input)))

            self._A = lambda state_vec, input_vec: \
                A_helper(*([x for x in state_vec] + [x for x in input_vec]))

            self._B = lambda state_vec, input_vec: \
                B_helper(*([x for x in state_vec] + [x for x in input_vec]))

            self._computed_dynamics_jacobian = True

        return (self._A(linearize_state, linearize_input),
                self._B(linearize_state, linearize_input))

    def hover_state_linearization(self) -> (np.ndarray, np.ndarray):
        """
        Linearizes the quadrotor model about the hover state.

        Linearization is done using Quadrotor.linearize, so it will be slow if
        no previous linearizations have been done.

        :return: A pair of matrices (A, B), where A is the state matrix and B
        is the input matrix.
        """

        return self.linearize(np.zeros(12),
                              np.array([self.gravity * self.mass, 0, 0, 0]))

    @abstractmethod
    def controller(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        The feedback control policy used to steer the quadrotor. State and input variables are the same as in
        ``dynamics``.

        :param state: A 12-dimensional vector specifying the current state of the robot.
        :param t: The current time of the simulation.
        :return: A 4-dimensional vector specifying the inputs to the robot.
        """
        pass

    @abstractmethod
    def clipped_controller(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        The feedback control policy used to steer the quadrotor. State and input variables are the same as in
        ``dynamics``. Includes simulation of clipping due to physical limits of the motors / firmware.

        :param state: A 12-dimensional vector specifying the current state of the robot.
        :param t: The current time of the simulation.
        :return: A 4-dimensional vector specifying the inputs to the robot.
        """
        pass

    def simulate(self, ic: np.ndarray, duration: float, timestep: Union[None, float] = None,
                 clip_input: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Simulate the quadrotor using numerical integration. State and input variables are the same as in
        ``dynamics``.

        :param ic: A 12-dimensional vector specifying the initial condition of the system.
        :param duration: The duration of the simulated trajectory.
        :param timestep: The timestep to use for the simulation. If None, solve_ivp determines the steps.
        :param clip_input: If True, the input is clipped using ``clipped_controller``. Otherwise unclipped
                           signal is used.
        :return: A tuple (t, x, u), where t is a vector containing the times at which the trajectory was evaluated,
                 x is a matrix containing the corresponding states as columns, and u is a matrix containing the
                corresponding inputs as columns.
        """

        control_func = self.clipped_controller if clip_input else self.controller

        sol = solve_ivp(lambda t, state: self.dynamics(state, control_func(state, t)), (0, duration), ic, method='BDF',
                        t_eval=(np.arange(0, duration, timestep) if timestep is not None else None))

        return sol.t, sol.y, np.array([control_func(sol.y[:, i], sol.t[i]) for i in range(len(sol.t))]).T


class Crazyflie(Quadrotor):
    """
    Abstract class describing the Bitcraze Crazyflie 2.0 dynamical model.

    All that needs to be implemented in a derived class is the control policy (``controller``).
    """

    def __init__(self):
        rot45 = Rot.from_euler('z', -np.pi / 4).as_matrix()
        self._I = np.diag([1.4194e-05, 1.4089e-05,  2.9741e-05])
        super().__init__()

    @property
    def mass(self) -> float:
        return 0.03

    @property
    def gravity(self) -> float:
        return 9.8

    @property
    def I(self) -> np.ndarray:
        return self._I

    @property
    def max_thrust(self) -> float:
        return 60.0 * 9.8

    @property
    def lift_coeff(self) -> float:
        """
        This property returns the lift coefficient of the quadrotor (commonly k_f).
        """
        return 1.938952e-6

    @property
    def drag_coeff(self):
        """
        This property returns the lift coefficient of the quadrotor (commonly k_m).
        """
        return 4.760115E-08

    @property
    def arm_length(self):
        """
        This property returns the arm length of the quadrotor (commonly L).
        :return:
        """
        return 0.046

    @abstractmethod
    def controller(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        The feedback control policy used to steer the quadrotor. State and input variables are the same as in
        ``dynamics``.

        :param state: A 12-dimensional vector specifying the current state of the robot.
        :param t: The current time of the simulation.
        :return: A 4-dimensional vector specifying the inputs to the robot.
        """
        pass

    def clipped_controller(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        The feedback control policy used to steer the quadrotor. State and input variables are the same as in
        ``dynamics``. Includes simulation of clipping due to physical limits of the motors / firmware.

        :param state: A 12-dimensional vector specifying the current state of the robot.
        :param t: The current time of the simulation.
        :return: A 4-dimensional vector specifying the inputs to the robot.
        """

        kf = self.lift_coeff
        km = self.drag_coeff
        L = self.arm_length

        pwm_to_input = kf * np.array([
                   [                    1,                     1,                     1,                     1],
                   [-(2 ** (1/2) * L) / 2, -(2 ** (1/2) * L) / 2,  (2 ** (1/2) * L) / 2,  (2 ** (1/2) * L) / 2],
                   [ (2 ** (1/2) * L) / 2, -(2 ** (1/2) * L) / 2, -(2 ** (1/2) * L) / 2,  (2 ** (1/2) * L) / 2],
                   [              km / kf,              -km / kf,               km / kf,              -km / kf]])

        u = self.controller(state, t)

        u_pwm = np.zeros(4)
        u_pwm[0] = u[0] / (4 * kf)
        u_pwm[1] = 2 * np.sqrt(2) * u[1] / (4 * L * kf);
        u_pwm[2] = 2 * np.sqrt(2) * u[2] / (4 * L * kf);
        u_pwm[3] = u[3] / (4 * km);

        u_pwm_int16 = np.zeros(4)
        u_pwm_int16[0] = u_pwm[0]
        u_pwm_int16[1:] = np.clip(u_pwm[1:], -32768, 32767)
        u_pwm_int16[1:3] = u_pwm_int16[1:3] / 2

        motor_pwms = np.clip(np.array([[1, -1,  1,  1],
                                       [1, -1, -1, -1],
                                       [1,  1, -1,  1],
                                       [1,  1,  1, -1]]) @ u_pwm_int16,
                              0, 2 ** 16 - 1)

        u_clipped = pwm_to_input @ motor_pwms

        return u_clipped


class Dummyflie(Crazyflie):
    """
    A simple implementation of the Crazyflie class with control law u = 0.
    """

    def __init__(self):
        super().__init__()

    def controller(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        The feedback control policy used to steer the quadrotor. State and input variables are the same as in
        ``dynamics``. This is a dumb implementation of a feedback law, i.e. u = 0.

        :param state: A 12-dimensional vector specifying the current state of the robot.
        :param t: The current time of the simulation.
        :return: A 4-dimensional vector specifying the inputs to the robot.
        """
        return np.zeros(4)
