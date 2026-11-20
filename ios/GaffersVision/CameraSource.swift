// Camera capture: upright (portrait), unmirrored BGRA frames with their real presentation times,
// delivered on one background queue. Late frames are dropped, not queued, so feedback never falls
// behind the camera (the same newest-frame-only rule as src/live.py). See docs/PREPROCESSING.md.
import AVFoundation
import CoreMedia

final class CameraSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Called on `queue` for every frame that arrives while the previous one is done.
    /// (pixel buffer, presentation time in seconds)
    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "gaffersvision.camera")
    private(set) var position: AVCaptureDevice.Position = .front

    /// Frame size after rotation (width, height), known once configured.
    private(set) var frameSize: (Int, Int)? = nil

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    /// Set up the session for one camera. Front: the player sees the screen while doing the drill.
    func configure(position: AVCaptureDevice.Position) throws {
        self.position = position
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        // 720p, 30 fps: the models resize to 320-640 px anyway (docs/PREPROCESSING.md, Capture).
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw CameraError.noCamera
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAdd }
        session.addInput(input)
        try device.lockForConfiguration()
        device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        device.unlockForConfiguration()

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAdd }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            // Upright portrait frames for the models.
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            // Never mirror what the models see, or the left and right feet swap. The screen
            // mirrors the front camera for display only (ContentView).
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let (long, short) = (Int(max(dims.width, dims.height)), Int(min(dims.width, dims.height)))
        frameSize = (short, long)  // portrait
    }

    func start() {
        queue.async { [session] in
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        queue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // The camera's presentation time, never frame index / fps (phones vary their frame rate).
        let tS = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        onFrame?(pixelBuffer, tS)
    }
}

enum CameraError: Error {
    case noCamera, cannotAdd
}
