// Shared value types. Every position is in pixels of the upright, unmirrored full frame,
// origin top-left (docs/PREPROCESSING.md). Python uses tuples for all of these.
import Foundation

public struct Point: Equatable, Sendable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// Euclidean distance (Python math.dist).
@inlinable
public func distance(_ a: Point, _ b: Point) -> Double {
    hypot(a.x - b.x, a.y - b.y)
}

/// A detector box with its confidence (Python (x1, y1, x2, y2, confidence)).
public struct Detection: Equatable, Sendable {
    public var x1: Double
    public var y1: Double
    public var x2: Double
    public var y2: Double
    public var confidence: Double

    public init(x1: Double, y1: Double, x2: Double, y2: Double, confidence: Double) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
    }

    /// Box centre (ball.box_center).
    public var center: Point {
        Point(x: (x1 + x2) / 2, y: (y1 + y2) / 2)
    }

    /// The shorter side, since motion blur stretches the longer one (ball.box_diameter_px).
    public var diameterPx: Double {
        min(x2 - x1, y2 - y1)
    }
}

/// The square foot crop in whole pixels; it may extend past the frame edge (ball.foot_crop).
public struct CropBox: Equatable, Sendable {
    public var x1: Int
    public var y1: Int
    public var x2: Int
    public var y2: Int

    public init(x1: Int, y1: Int, x2: Int, y2: Int) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }
}
