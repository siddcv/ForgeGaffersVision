"""Causal filters. Neither one sees future samples."""
import math


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t = None
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, t: float, x: float) -> float:
        if self._t is None or t <= self._t:
            self._t, self._x = t, x
            return x
        dt = t - self._t
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class Kalman1D:
    """Constant-velocity Kalman filter for one coordinate."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.q = process_noise
        self.r = measurement_noise
        self.t = None
        self.x = 0.0
        self.v = 0.0
        self.p = [[0.0, 0.0], [0.0, 0.0]]

    def predict(self, t: float) -> float | None:
        if self.t is None:
            return None
        dt = t - self.t
        if dt <= 0:
            return self.x
        (p00, p01), (p10, p11) = self.p
        q = self.q
        self.x += self.v * dt
        self.p = [
            [
                p00 + dt * (p01 + p10) + dt * dt * p11 + q * dt**4 / 4,
                p01 + dt * p11 + q * dt**3 / 2,
            ],
            [
                p10 + dt * p11 + q * dt**3 / 2,
                p11 + q * dt * dt,
            ],
        ]
        self.t = t
        return self.x

    def update(self, t: float, z: float) -> float:
        if self.t is None:
            self.t, self.x = t, z
            self.p = [[self.r, 0.0], [0.0, 1.0e6]]
            return z
        self.predict(t)
        (p00, p01), (p10, p11) = self.p
        s = p00 + self.r
        k0, k1 = p00 / s, p10 / s
        residual = z - self.x
        self.x += k0 * residual
        self.v += k1 * residual
        self.p = [
            [(1 - k0) * p00, (1 - k0) * p01],
            [p10 - k1 * p00, p11 - k1 * p01],
        ]
        return self.x
