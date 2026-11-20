// Port of src/face_pose.py's geometry: pitch from Face Landmarker's 4x4 face transform, and the
// head crop the sampler feeds it. Plain Swift (no simd or Accelerate), so it is tested against the
// Python reference on any platform.
import Foundation
import GaffersVisionCore

/// Face pitch in degrees from a 4x4 transform (row-major rows), or nil if it is not usable.
/// Python takes the nearest rotation with an SVD (u @ vh); this uses the polar decomposition
/// (Newton iteration R = (R + R^-T) / 2), which converges to the same matrix for a proper rotation.
func pitchFromTransform(_ m: [[Double]]) -> Double? {
    guard m.count == 4, m.allSatisfy({ $0.count == 4 }), m.joined().allSatisfy({ $0.isFinite }) else { return nil }
    var r = (0..<3).map { i in (0..<3).map { j in m[i][j] } }
    guard determinant(r) > 0 else { return nil }  // a reflection or degenerate: Python flips it; never seen in practice
    for _ in 0..<30 {
        guard let inverse = inverse3(r) else { return nil }
        let next = (0..<3).map { i in (0..<3).map { j in 0.5 * (r[i][j] + inverse[j][i]) } }
        let change = zip(next.joined(), r.joined()).map { abs($0 - $1) }.max()!
        r = next
        if change < 1e-12 { break }
    }
    let pitch = -atan2(r[2][1], r[1][1]) * 180 / .pi
    guard pitch.isFinite else { return nil }
    // Principal range, as in Python: (pitch + 180) % 360 - 180 with a floor modulo.
    let shifted = pitch + 180
    return shifted - 360 * (shifted / 360).rounded(.down) - 180
}

private func determinant(_ a: [[Double]]) -> Double {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

private func inverse3(_ a: [[Double]]) -> [[Double]]? {
    let det = determinant(a)
    guard abs(det) > 1e-15 else { return nil }
    let c: [[Double]] = [
        [a[1][1] * a[2][2] - a[1][2] * a[2][1], a[0][2] * a[2][1] - a[0][1] * a[2][2], a[0][1] * a[1][2] - a[0][2] * a[1][1]],
        [a[1][2] * a[2][0] - a[1][0] * a[2][2], a[0][0] * a[2][2] - a[0][2] * a[2][0], a[0][2] * a[1][0] - a[0][0] * a[1][2]],
        [a[1][0] * a[2][1] - a[1][1] * a[2][0], a[0][1] * a[2][0] - a[0][0] * a[2][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]],
    ]
    return c.map { $0.map { $0 / det } }
}

/// pose.head_box: box around the visible face landmarks, padded by `padding` of its larger side.
func headBox(_ landmarks: [Landmark], minVisibility: Double, padding: Double) -> (Double, Double, Double, Double)? {
    let points = visiblePoints(landmarks, Array(0..<11), minVisibility)
    guard points.count >= 2 else { return nil }
    let xs = points.map(\.x), ys = points.map(\.y)
    let margin = padding * max(xs.max()! - xs.min()!, ys.max()! - ys.min()!)
    return (xs.min()! - margin, ys.min()! - margin, xs.max()! + margin, ys.max()! + margin)
}

/// HeadPitchSampler's crop: square around the head box (1.5x its larger side), clamped to the
/// frame (so it may end up not square), in top-left pixel coordinates (x1, y1, x2, y2).
func headCrop(_ landmarks: [Landmark], frameWidth: Int, frameHeight: Int, cfg: Config) -> CropBox? {
    guard let (x1, y1, x2, y2) = headBox(landmarks, minVisibility: cfg.poseMinVisibility, padding: cfg.headBoxPadding)
    else { return nil }
    let cx = (x1 + x2) / 2, cy = (y1 + y2) / 2
    let side = max(x2 - x1, y2 - y1) * 1.5  // HeadPitchSampler.CROP_SCALE
    let crop = CropBox(x1: max(0, Int(cx - side / 2)), y1: max(0, Int(cy - side / 2)),
                       x2: min(frameWidth, Int(cx + side / 2)), y2: min(frameHeight, Int(cy + side / 2)))
    return crop.x2 > crop.x1 && crop.y2 > crop.y1 ? crop : nil
}
