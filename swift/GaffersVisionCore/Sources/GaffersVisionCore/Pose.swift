// Port of the geometry in src/pose.py that the per-frame logic reads from the 33 MediaPipe Pose
// landmarks. Running MediaPipe itself is outside this package.
import Foundation

/// One pose landmark in full-frame pixels. Missing values (no person found) are NaN, which fails
/// every visibility test, as in Python.
public struct Landmark: Equatable, Sendable {
    public var x: Double
    public var y: Double
    public var visibility: Double

    public init(x: Double, y: Double, visibility: Double) {
        self.x = x
        self.y = y
        self.visibility = visibility
    }

    public static let missing = Landmark(x: .nan, y: .nan, visibility: .nan)
}

public enum PoseIndex {
    public static let nose = 0, leftEar = 7, rightEar = 8
    public static let leftHeel = 29, rightHeel = 30, leftToe = 31, rightToe = 32
    public static let leftFoot = [leftHeel, leftToe]
    public static let rightFoot = [rightHeel, rightToe]
}

/// (x, y) of the listed landmarks that are visible enough (pose.visible_points).
public func visiblePoints(_ landmarks: [Landmark], _ indices: [Int], _ minVisibility: Double) -> [Point] {
    indices.filter { landmarks[$0].visibility >= minVisibility }
        .map { Point(x: landmarks[$0].x, y: landmarks[$0].y) }
}

/// Rough pitch from nose height relative to the ears; negative is looking down
/// (pose.coarse_head_pitch_deg). Only a fallback: the head thresholds in Config are calibrated
/// for Face Landmarker pitch, which is on a different scale.
public func coarseHeadPitchDeg(_ landmarks: [Landmark], _ minVisibility: Double) -> Double? {
    let indices = [PoseIndex.nose, PoseIndex.leftEar, PoseIndex.rightEar]
    // Written so that NaN visibility (no person detected) also returns nil.
    guard indices.allSatisfy({ landmarks[$0].visibility >= minVisibility }) else { return nil }
    let left = landmarks[PoseIndex.leftEar], right = landmarks[PoseIndex.rightEar]
    let nose = landmarks[PoseIndex.nose]
    let earSpan = hypot(left.x - right.x, left.y - right.y)
    if earSpan == 0 {
        return nil
    }
    let earY = (left.y + right.y) / 2
    return -(atan2(nose.y - earY, earSpan) * (180.0 / Double.pi))
}
