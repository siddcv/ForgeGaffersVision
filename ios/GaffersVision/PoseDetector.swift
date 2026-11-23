// MediaPipe Pose Landmarker (lite), the same model file as the Python reference (src/pose.py).
// Video mode: one synchronous call per frame, timestamps in whole, strictly increasing milliseconds.
// Output: 33 landmarks in full-frame pixels, or all Landmark.missing when no person is found.
import CoreVideo
import GaffersVisionCore
import MediaPipeTasksVision

final class PoseDetector {
    private let landmarker: PoseLandmarker
    private var lastMs = -1

    init() throws {
        guard let path = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task") else {
            throw ModelError.missing("pose_landmarker_lite.task")
        }
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = path
        options.runningMode = .video
        options.numPoses = 1
        landmarker = try PoseLandmarker(options: options)
    }

    /// pixelBuffer: the upright, unmirrored BGRA frame. Never throws for "no person".
    func detect(_ pixelBuffer: CVPixelBuffer, tS: Double) -> [GaffersVisionCore.Landmark] {
        let width = Double(CVPixelBufferGetWidth(pixelBuffer))
        let height = Double(CVPixelBufferGetHeight(pixelBuffer))
        // MediaPipe needs strictly increasing times: add 1 ms on a tie (docs/PREPROCESSING.md).
        let ms = max(Int(tS * 1000), lastMs + 1)
        lastMs = ms
        guard let image = try? MPImage(pixelBuffer: pixelBuffer),
              let result = try? landmarker.detect(videoFrame: image, timestampInMilliseconds: ms),
              let person = result.landmarks.first, person.count == 33
        else {
            return Array(repeating: .missing, count: 33)
        }
        return person.map { point in
            GaffersVisionCore.Landmark(x: Double(point.x) * width, y: Double(point.y) * height,
                     visibility: point.visibility?.doubleValue ?? 0)
        }
    }
}

enum ModelError: Error, CustomStringConvertible {
    case missing(String)
    case badOutput(String)

    var description: String {
        switch self {
        case .missing(let name): return "\(name) is not in the app bundle (see ios/README.md, step 2)"
        case .badOutput(let what): return "unexpected model output: \(what)"
        }
    }
}
