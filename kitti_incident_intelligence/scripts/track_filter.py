"""
Range-Aware Kalman Filter for 3D Track Filtering
Dynamically scales measurement covariance with range according to stereo physics:
    sigma_Z(Z) = (Z^2 * delta_d) / (f * B) + sigma_0
Rejects implausible measurement jumps and provides filtered positions and velocities.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RangeAwareKalmanFilter:
    """
    Constant-Velocity Kalman Filter in 3D Camera/Ground plane coordinates:
    State: x = [X, Z, Vx, Vz]^T
    where X is lateral position, Z is longitudinal range,
          Vx is lateral velocity, Vz is longitudinal velocity (relative).
    """

    def __init__(
        self,
        init_X: float,
        init_Z: float,
        dt: float = 0.1,
        fB: float = 384.3815,
        fx: float = 721.5377,
        disp_noise_px: float = 0.35,
        min_sigma_z: float = 0.05,
        proc_acc_std: float = 1.5,
    ):
        self.dt = dt
        self.fB = fB
        self.fx = fx
        self.disp_noise_px = disp_noise_px
        self.min_sigma_z = min_sigma_z
        self.proc_acc_std = proc_acc_std

        # State vector: [X, Z, Vx, Vz]
        self.x = np.array([init_X, init_Z, 0.0, 0.0], dtype=np.float64)

        # Initial Covariance Matrix P
        init_sigma_z = self._compute_sigma_z(init_Z, 1.0)
        init_sigma_x = self._compute_sigma_x(init_Z)
        self.P = np.diag([init_sigma_x**2, init_sigma_z**2, 4.0, 4.0]).astype(np.float64)

        # State Transition Matrix F
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Measurement Matrix H: observes [X, Z]
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Continuous white noise acceleration model for Q
        q_var = proc_acc_std**2
        dt2 = 0.5 * (dt**2)
        G = np.array(
            [
                [dt2, 0.0],
                [0.0, dt2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        self.Q = (G @ G.T) * q_var

        # Quality metrics
        self.age = 0
        self.hits = 1
        self.last_mahalanobis = 0.0
        self.last_measurement_sigma_z = init_sigma_z

    def _compute_sigma_z(self, Z: float, confidence: float) -> float:
        """Range-dependent depth uncertainty from empirical stereo physics."""
        safe_z = max(1.0, min(80.0, Z))
        safe_conf = max(0.2, min(1.0, confidence))
        sigma = ((safe_z**2) * self.disp_noise_px) / (self.fB * safe_conf) + self.min_sigma_z
        return float(sigma)

    def _compute_sigma_x(self, Z: float) -> float:
        """Lateral uncertainty scaled with range."""
        safe_z = max(1.0, min(80.0, Z))
        sigma = (safe_z * 1.5) / self.fx + 0.05
        return float(sigma)

    def predict(self) -> np.ndarray:
        """Time update step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.x

    def update(
        self, meas_X: float, meas_Z: float, confidence: float = 1.0
    ) -> Tuple[np.ndarray, bool]:
        """
        Measurement update step with range-aware covariance and gating.
        Returns: (state, accepted_bool)
        """
        z = np.array([meas_X, meas_Z], dtype=np.float64)

        # Dynamic Range-dependent Measurement Covariance R
        sigma_z = self._compute_sigma_z(meas_Z, confidence)
        sigma_x = self._compute_sigma_x(meas_Z)
        self.last_measurement_sigma_z = sigma_z

        R = np.diag([sigma_x**2, sigma_z**2])

        # Innovation / Residual
        y = z - (self.H @ self.x)

        # Innovation Covariance S
        S = self.H @ self.P @ self.H.T + R

        # Mahalanobis gating to reject implausible jumps (chi-square 2-DOF at 99.5% = 10.6)
        try:
            S_inv = np.linalg.inv(S)
            d_m2 = float(y.T @ S_inv @ y)
            self.last_mahalanobis = d_m2
        except np.linalg.LinAlgError:
            d_m2 = 0.0
            S_inv = np.diag([1.0 / (sigma_x**2), 1.0 / (sigma_z**2)])

        # Gating: reject jumps exceeding 16.0 except on initial track acquisition
        if self.hits > 3 and d_m2 > 16.0:
            # Plausibility rejection: do not collapse filter on a wild spike
            return self.x, False

        # Kalman Gain K
        K = self.P @ self.H.T @ S_inv

        # State and Covariance Update
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        # Joseph form for numerical stability
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T

        self.hits += 1
        return self.x, True

    def get_state(self) -> Dict:
        return {
            "X": float(self.x[0]),
            "Z": float(self.x[1]),
            "Vx": float(self.x[2]),
            "Vz": float(self.x[3]),
            "sigma_X": float(np.sqrt(max(1e-6, self.P[0, 0]))),
            "sigma_Z": float(np.sqrt(max(1e-6, self.P[1, 1]))),
            "sigma_Vx": float(np.sqrt(max(1e-6, self.P[2, 2]))),
            "sigma_Vz": float(np.sqrt(max(1e-6, self.P[3, 3]))),
            "meas_sigma_z": float(self.last_measurement_sigma_z),
            "age": self.age,
            "hits": self.hits,
        }
