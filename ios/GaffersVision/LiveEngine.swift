// The live loop, as in src/live.py: camera -> pose -> foot crop -> ball detector ->
// FramePipeline.step (GaffersVisionCore) -> screen. Drills: READY (overlay runs, nothing counted;
// START waits for head-to-toe framing) -> START -> 3 s countdown -> DRILL (counts from zero) -> STOP
// -> summary. All per-frame work runs on the camera queue; the screen gets a snapshot per frame.
import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import GaffersVisionCore
import QuartzCore

enum DrillPhase: Equatable {
    case ready
    case countdown(secondsLeft: Double)
    case drill(elapsedS: Double)
}

struct DrillSummary: Equatable {
    var durationS: Double
    var tickTocksLeft = 0, tickTocksRight = 0
    var contactsLeft = 0, contactsRight = 0
    var lostControl = 0
    var headDown = 0  // head-down events (looking down >= headDownMinS)
    var recenterWarnings = 0
    var lastMinute = 0
    var processedFps = 0.0
    var medianProcMs = 0.0
    var fullFrameRuns = 0
    var frames = 0
}

/// Everything the screen draws for one processed frame.
struct Snapshot {
    var image: CGImage?
    var frameWidth = 720.0, frameHeight = 1280.0
    var landmarks: [Landmark] = []
    var crop: CropBox?
    var ball: Detection?
    var predictedXY: Point?
    var searching = false
    var phase: DrillPhase = .ready
    var startEnabled = false
    var framingProblems: [String] = []
    var recenterProblems: [String] = []  // non-empty: show RECENTER YOURSELF
    var counts = DrillSummary(durationS: 0)
    var pacePerMin: Double?
    var touchFlash = false, contactFlash = false, lostFlash = false
    var headDown = false  // between a head_down and the next head_up event
    var headPitchDeg: Double?
    var faceLandmarker = false  // false: pose-only pitch estimate (uncalibrated)
    var fps = 0.0, procMs = 0.0
}

@MainActor
final class LiveEngine: ObservableObject {
    @Published private(set) var snapshot = Snapshot()
    @Published private(set) var summary: DrillSummary?
    @Published private(set) var error: String?
    @Published private(set) var mirrored = true

    private let camera = CameraSource()
    private var processor: Processor?

    func start() async {
        guard await CameraSource.requestAccess() else {
            error = "Camera access is off: Settings > Privacy & Security > Camera > Gaffer's Vision."
            return
        }
        do {
            let processor = try Processor(cfg: Config())
            processor.publish = { [weak self] snapshot, summary in
                Task { @MainActor in
                    self?.snapshot = snapshot
                    if let summary { self?.summary = summary }
                }
            }
            self.processor = processor
            camera.onFrame = { [processor] buffer, tS in processor.process(buffer, tS: tS) }
            try camera.configure(position: .front)
            camera.start()
        } catch {
            self.error = "Could not start: \(error)"
        }
    }

    func switchCamera() {
        let next: AVCaptureDevice.Position = camera.position == .front ? .back : .front
        camera.queue.async { [camera, weak self] in
            try? camera.configure(position: next)
            Task { @MainActor in self?.mirrored = next == .front }
        }
    }

    /// The START / CANCEL / STOP button (and the same for a tap on the summary's close).
    func buttonTapped() {
        guard let processor else { return }
        camera.queue.async { processor.buttonTapped() }
    }

    func dismissSummary() { summary = nil }
}

/// Per-frame work, confined to the camera queue.
final class Processor {
    var publish: ((Snapshot, DrillSummary?) -> Void)?

    private let cfg: Config
    private let pose: PoseDetector
    private let detector: BallDetector
    private let headSampler: HeadPitchSampler?  // nil: face_landmarker.task missing, pose estimate
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    private let countdownS = 3.0
    private let flashS = 0.3

    private var preview: FramePipeline
    private var drill: FramePipeline?
    private var phase = Phase.ready
    private var drillT0: Double?
    private var framing = FramingGate()
    private var recenter = FramingGate(ready: true)
    private var counts = DrillSummary(durationS: 0)
    private var lastFlash: [String: Double] = [:]
    private var headDown = false
    private var lastTS = 0.0
    private var procMs: [Double] = []  // this drill
    private var recentProc: [Double] = []  // last 30 frames, for the status line

    private enum Phase { case ready, countdown(endS: Double), drill }

    init(cfg: Config) throws {
        self.cfg = cfg
        pose = try PoseDetector()
        detector = try BallDetector(cfg: cfg)
        headSampler = HeadPitchSampler(cfg: cfg)
        preview = FramePipeline(cfg)
    }

    func buttonTapped() {
        switch phase {
        case .ready:
            guard framing.ready else { return }  // disabled until head to toe and ball are in frame
            phase = .countdown(endS: lastTS + countdownS)
            recenter = FramingGate(ready: true)
        case .countdown:
            phase = .ready
        case .drill:
            let summary = finishDrill()
            phase = .ready
            preview = FramePipeline(cfg)
            framing = FramingGate()
            publish?(makeSnapshot(nil, record: nil, landmarks: []), summary)
        }
    }

    func process(_ frame: CVPixelBuffer, tS: Double) {
        let started = CACurrentMediaTime()
        lastTS = tS
        if case .countdown(let endS) = phase, tS >= endS {
            phase = .drill
            drillT0 = tS
            drill = FramePipeline(cfg)
            counts = DrillSummary(durationS: 0)
            lastFlash = [:]
            headDown = false
            procMs = []
        }
        let counting: Bool
        if case .drill = phase { counting = true } else { counting = false }
        let pipeline = counting ? drill! : preview
        let tUsed = counting ? tS - drillT0! : tS
        let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)

        let landmarks = pose.detect(frame, tS: tS)
        let crop = pipeline.footCrop(landmarks, frameWidth: width, frameHeight: height)
        let cropCandidates = crop.map { detector.detectCrop(frame, crop: $0) } ?? []
        // Face Landmarker pitch at 10 Hz (held between samples), as in replay and src/live.py.
        let headPitch: HeadPitch = headSampler.map { HeadPitch.face($0.update(frame, landmarks: landmarks, tS: tS)) }
            ?? .poseEstimate
        let record = pipeline.step(tS: tUsed, landmarks: landmarks, crop: crop, cropCandidates: cropCandidates,
                                   fullCandidates: { [detector] in detector.detectFull(frame) },
                                   headPitch: headPitch)

        let problems = framingProblems(landmarks, ball: record.ball, width: Double(width), height: Double(height),
                                       minVisibility: cfg.poseMinVisibility)
        if case .ready = phase {
            framing.update(tS, problems)
        } else {
            let wasIn = recenter.ready
            recenter.update(tS, bodyProblems(problems))
            if counting && wasIn && !recenter.ready { counts.recenterWarnings += 1 }
        }
        if counting {
            count(record, tUsed)
            if record.ranFullFrame { counts.fullFrameRuns += 1 }
        }
        let ms = (CACurrentMediaTime() - started) * 1000
        recentProc = Array((recentProc + [ms]).suffix(30))
        if counting {
            procMs.append(ms)
            counts.durationS = tUsed
            counts.frames += 1
        }
        var snapshot = makeSnapshot(frame, record: record, landmarks: landmarks)
        snapshot.pacePerMin = record.touchRate.pacePerMin
        snapshot.headPitchDeg = record.headPitchDeg
        publish?(snapshot, nil)
    }

    private func count(_ record: FrameRecord, _ tS: Double) {
        for event in record.events {
            switch (event.kind, event.foot) {
            case (.touch, .left?): counts.tickTocksLeft += 1; lastFlash["touch"] = tS
            case (.touch, .right?): counts.tickTocksRight += 1; lastFlash["touch"] = tS
            case (.contact, .left?): counts.contactsLeft += 1; lastFlash["contact"] = tS
            case (.contact, .right?): counts.contactsRight += 1; lastFlash["contact"] = tS
            case (.lostControl, _): counts.lostControl += 1; lastFlash["lost"] = tS
            case (.headDown, _): counts.headDown += 1; headDown = true
            case (.headUp, _): headDown = false
            default: break
            }
        }
        counts.lastMinute = record.touchRate.lastMinute
    }

    private func finishDrill() -> DrillSummary {
        var summary = counts
        let sorted = procMs.sorted()
        summary.medianProcMs = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
        summary.processedFps = summary.durationS > 0 ? Double(summary.frames) / summary.durationS : 0
        drill = nil
        drillT0 = nil
        return summary
    }

    private func makeSnapshot(_ frame: CVPixelBuffer?, record: FrameRecord?, landmarks: [Landmark]) -> Snapshot {
        var s = Snapshot()
        if let frame {
            s.image = imageContext.createCGImage(CIImage(cvPixelBuffer: frame),
                                                 from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(frame),
                                                              height: CVPixelBufferGetHeight(frame)))
            s.frameWidth = Double(CVPixelBufferGetWidth(frame))
            s.frameHeight = Double(CVPixelBufferGetHeight(frame))
        }
        s.landmarks = landmarks
        s.crop = record?.crop
        s.ball = record?.ball
        s.predictedXY = record?.predicted == true ? record?.ballXY : nil
        s.searching = record?.searchedFullFrame ?? false
        switch phase {
        case .ready: s.phase = .ready
        case .countdown(let endS): s.phase = .countdown(secondsLeft: max(0, endS - lastTS))
        case .drill: s.phase = .drill(elapsedS: counts.durationS)
        }
        s.startEnabled = framing.ready
        s.framingProblems = framing.problems
        if case .ready = phase {} else if !recenter.ready { s.recenterProblems = recenter.problems }
        s.counts = counts
        s.faceLandmarker = headSampler != nil
        if case .drill = phase { s.headDown = headDown }
        let now = counts.durationS
        if case .drill = phase {
            s.touchFlash = now - (lastFlash["touch"] ?? -.infinity) < flashS
            s.contactFlash = now - (lastFlash["contact"] ?? -.infinity) < flashS
            s.lostFlash = now - (lastFlash["lost"] ?? -.infinity) < 1.0
        }
        let median = recentProc.sorted()[safe: recentProc.count / 2] ?? 0
        s.procMs = median
        s.fps = median > 0 ? min(30, 1000 / median) : 0
        return s
    }
}

extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}
