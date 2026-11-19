// Port of src/filters.py. Causal filters: neither one sees future samples.
// Arithmetic follows the Python expression by expression, so results match to rounding.
import Foundation

public final class OneEuroFilter {
    public let minCutoff: Double
    public let beta: Double
    public let dCutoff: Double
    private var t: Double? = nil
    private var x: Double? = nil
    private var dx: Double = 0.0

    public init(minCutoff: Double, beta: Double, dCutoff: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    private static func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    public func update(_ t: Double, _ x: Double) -> Double {
        guard let previousT = self.t, let previousX = self.x, t > previousT else {
            self.t = t
            self.x = x
            return x
        }
        let dt = t - previousT
        let aD = Self.alpha(dCutoff, dt)
        dx = aD * (x - previousX) / dt + (1 - aD) * dx
        let a = Self.alpha(minCutoff + beta * abs(dx), dt)
        let smoothed = a * x + (1 - a) * previousX
        self.x = smoothed
        self.t = t
        return smoothed
    }
}

/// Constant-velocity Kalman filter for one coordinate.
public final class Kalman1D {
    public let q: Double
    public let r: Double
    public private(set) var t: Double? = nil
    public private(set) var x: Double = 0.0
    public private(set) var v: Double = 0.0
    private var p00 = 0.0, p01 = 0.0, p10 = 0.0, p11 = 0.0

    public init(processNoise: Double, measurementNoise: Double) {
        q = processNoise
        r = measurementNoise
    }

    /// Predicted position at time t, without changing the filter.
    public func peek(_ t: Double) -> Double? {
        guard let last = self.t else { return nil }
        return x + v * max(0.0, t - last)
    }

    @discardableResult
    public func predict(_ t: Double) -> Double? {
        guard let last = self.t else { return nil }
        let dt = t - last
        if dt <= 0 {
            return x
        }
        x += v * dt
        let n00 = p00 + dt * (p01 + p10) + dt * dt * p11 + q * pow(dt, 4) / 4
        let n01 = p01 + dt * p11 + q * pow(dt, 3) / 2
        let n10 = p10 + dt * p11 + q * pow(dt, 3) / 2
        let n11 = p11 + q * dt * dt
        (p00, p01, p10, p11) = (n00, n01, n10, n11)
        self.t = t
        return x
    }

    public func update(_ t: Double, _ z: Double) -> Double {
        if self.t == nil {
            self.t = t
            x = z
            (p00, p01, p10, p11) = (r, 0.0, 0.0, 1.0e6)
            return z
        }
        predict(t)
        let s = p00 + r
        let k0 = p00 / s
        let k1 = p10 / s
        let residual = z - x
        x += k0 * residual
        v += k1 * residual
        (p00, p01, p10, p11) = ((1 - k0) * p00, (1 - k0) * p01, p10 - k1 * p00, p11 - k1 * p01)
        return x
    }
}
