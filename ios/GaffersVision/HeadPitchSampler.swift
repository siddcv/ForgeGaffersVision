// Port of HeadPitchSampler in src/face_pose.py: Face Landmarker pitch, sampled causally every
// headPoseSampleIntervalS (10 Hz). Face Landmarker misses the small face in the full portrait
// frame, so each sample crops around the pose head box (1.5x, clamped to the frame) and enlarges
// it to 256 x 256 (bilinear) first. The latest sample is held until the next; nil means no face
// at the last sample. Same model file as Python: face_landmarker.task.
import CoreImage
import CoreVideo
import GaffersVisionCore
import MediaPipeTasksVision

final class HeadPitchSampler {
    private static let inputSize = 256  // HeadPitchSampler.INPUT_SIZE

    private let cfg: Config
    private let landmarker: FaceLandmarker
    private let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(),
                                              .cacheIntermediates: false])
    private var buffer: CVPixelBuffer?
    private var nextSampleS = -Double.infinity
    private var lastMs = -1
    private(set) var latest: Double?

    /// nil if the model is missing or Face Landmarker cannot start; callers then use the pose
    /// estimate, as Python does (open_head_pitch_sampler).
    init?(cfg: Config) {
        guard let path = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else { return nil }
        let options = FaceLandmarkerOptions()
        options.baseOptions.modelAssetPath = path
        options.runningMode = .video
        options.numFaces = 1
        options.outputFacialTransformationMatrixes = true
        guard let landmarker = try? FaceLandmarker(options: options) else { return nil }
        self.cfg = cfg
        self.landmarker = landmarker
    }

    /// Pitch for this frame (held between samples). tS: the camera's time for this frame.
    func update(_ frame: CVPixelBuffer, landmarks: [Landmark], tS: Double) -> Double? {
        if tS + 1e-9 < nextSampleS { return latest }
        latest = nil
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
        if let crop = headCrop(landmarks, frameWidth: width, frameHeight: height, cfg: cfg),
           let input = render(frame, crop: crop, frameHeight: height) {
            let ms = max(Int(tS * 1000), lastMs + 1)  // strictly increasing, as MediaPipe needs
            lastMs = ms
            if let image = try? MPImage(pixelBuffer: input),
               let result = try? landmarker.detect(videoFrame: image, timestampInMilliseconds: ms),
               let matrix = result.facialTransformationMatrixes.first {
                let rows = (0..<4).map { r in (0..<4).map { c in Double(matrix.value(atRow: UInt(r), column: UInt(c))) } }
                latest = pitchFromTransform(rows)
            }
        }
        nextSampleS = tS + cfg.headPoseSampleIntervalS
        return latest
    }

    /// The head crop stretched to 256 x 256 (it may not be square after clamping, as in Python).
    private func render(_ frame: CVPixelBuffer, crop: CropBox, frameHeight: Int) -> CVPixelBuffer? {
        let size = CGFloat(Self.inputSize)
        let w = CGFloat(crop.x2 - crop.x1), h = CGFloat(crop.y2 - crop.y1)
        // Core Image's origin is bottom-left.
        let rect = CGRect(x: CGFloat(crop.x1), y: CGFloat(frameHeight - crop.y2), width: w, height: h)
        let picture = CIImage(cvPixelBuffer: frame).cropped(to: rect)
            .samplingLinear()
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .transformed(by: CGAffineTransform(scaleX: size / w, y: size / h))
        if buffer == nil {
            let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
            CVPixelBufferCreate(kCFAllocatorDefault, Self.inputSize, Self.inputSize, kCVPixelFormatType_32BGRA,
                                attributes, &buffer)
        }
        guard let buffer else { return nil }
        context.render(picture, to: buffer, bounds: CGRect(x: 0, y: 0, width: size, height: size), colorSpace: nil)
        return buffer
    }
}
