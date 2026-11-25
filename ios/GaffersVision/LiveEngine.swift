// The live loop, as in src/live.py: camera -> pose -> foot crop -> ball detector ->
// FramePipeline.step (GaffersVisionCore) -> screen. Drills (DrillFlow in Framing.swift): READY
// (overlay runs, nothing counted) -> START, which works out of frame -> WAITING until the player
// (head to toe) and the ball are in frame -> 3 s countdown (back to WAITING if they leave) -> DRILL
// (counts from zero) -> STOP, or the chosen drill length runs out -> summary. All per-frame work runs
// on the camera queue; the screen gets a snapshot per frame.
import AudioToolbox
import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import GaffersVisionCore
import QuartzCore

enum DrillPhase: Equatable {
    case ready
    case waiting  // START pressed; waiting for the player and ball to be fully in frame
    case countdown(secondsLeft: Double)
    case drill(elapsedS: Double, limitS: Double?)
}

/// Chosen before START (default 30 s); a timed drill stops itself. 60 s matches the 150-in-a-minute goal.
enum DrillLength: String, CaseIterable, Identifiable {
    case s30 = "30 s", s45 = "45 s", s60 = "60 s", open = "Open"
    static let standard = DrillLength.s30
    var id: String { rawValue }
    var seconds: Double? {
        switch self {
        case .s30: return 30
        case .s45: return 45
        case .s60: return 60
        case .open: return nil
        }
    }
}

/// Short system sounds, so a player away from the screen hears the drill start and end.
/// They follow the ring / silent switch.
enum Sounds {
    static func drillStart() { AudioServicesPlaySystemSound(1113) }  // begin_record
    static func drillEnd() { AudioServicesPlaySystemSound(1114) }  // end_record
}

struct DrillSummary: Equatable {
    var durationS: Double
    var tickTocksLeft = 0, tickTocksRight = 0
    var contactsLeft = 0, contactsRight = 0
    var lostControl = 0
    var headDown = 0  // head-down events (looking down >= headDownMinS)
    var recenterWarnings = 0
    var processedFps = 0.0
    var medianProcMs = 0.0
    var fullFrameRuns = 0
    var frames = 0

    static let paceGoalPerMin = 150  // the tick-tock goal: 150 in a minute

    var tickTocks: Int { tickTocksLeft + tickTocksRight }

    /// Tick-tocks so far, scaled to a minute (the drill's average pace); nil before minS seconds,
    /// when a couple of tick-tocks would give a wild number.
    func averagePacePerMin(minS: Double = 0) -> Double? {
        durationS > 0 && durationS >= minS ? Double(tickTocks) / durationS * 60 : nil
    }
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
    var framingProblems: [String] = []  // what is out of frame while waiting
    var recenterProblems: [String] = []  // non-empty: show RECENTER YOURSELF
    var counts = DrillSummary(durationS: 0)
    var pacePerMin: Double?  // the drill's average pace so far (DrillSummary.averagePacePerMin)
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
    @Published var drillLength = DrillLength.standard {
        didSet {
            guard let processor else { return }
            let length = drillLength
            camera.queue.async { processor.setDrillLength(length) }
        }
    }

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
            processor.setDrillLength(drillLength)
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
    private let flashS = 0.3

    private var preview: FramePipeline
    private var drill: FramePipeline?
    private var flow = DrillFlow(countdownS: 3.0, drillS: DrillLength.standard.seconds)
    private var drillT0: Double?  // the drill's first frame; nil until that frame arrives
    private var recenter = FramingGate(ready: true)
    private var counts = DrillSummary(durationS: 0)
    private var lastFlash: [String: Double] = [:]
    private var headDown = false
    private var lastTS = 0.0
    private var procMs: [Double] = []  // this drill
    private var recentProc: [Double] = []  // last 30 frames, for the status line

    init(cfg: Config) throws {
        self.cfg = cfg
        pose = try PoseDetector()
        detector = try BallDetector(cfg: cfg)
        headSampler = HeadPitchSampler(cfg: cfg)
        preview = FramePipeline(cfg)
    }

    /// Takes effect from the next START.
    func setDrillLength(_ length: DrillLength) {
        flow.drillS = length.seconds
    }

    func buttonTapped() {
        if flow.press(lastTS) == .finish {
            endDrill(frame: nil, landmarks: [])
        }
    }

    /// STOP, or the drill's time is up: the summary, and back to READY.
    private func endDrill(frame: CVPixelBuffer?, landmarks: [Landmark]) {
        Sounds.drillEnd()
        let summary = finishDrill()
        preview = FramePipeline(cfg)
        publish?(makeSnapshot(frame, record: nil, landmarks: landmarks), summary)
    }

    func process(_ frame: CVPixelBuffer, tS: Double) {
        let started = CACurrentMediaTime()
        lastTS = tS
        if flow.state == .drill && drill != nil && drillT0 == nil {
            drillT0 = tS  // the first frame after the countdown
        }
        let counting = flow.state == .drill && drill != nil
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
        if counting {  // warn about the body only; a lost ball is lost control
            let wasIn = recenter.ready
            recenter.update(tS, bodyProblems(problems))
            if wasIn && !recenter.ready { counts.recenterWarnings += 1 }
            count(record, tUsed)
            if record.ranFullFrame { counts.fullFrameRuns += 1 }
        }
        let action = flow.update(tS, problems, drillElapsedS: counting ? tUsed : nil)
        if action == .start {  // counting starts with the next frame
            drill = FramePipeline(cfg)
            drillT0 = nil
            recenter = FramingGate(ready: true)
            counts = DrillSummary(durationS: 0)
            lastFlash = [:]
            headDown = false
            procMs = []
            Sounds.drillStart()
        }
        let ms = (CACurrentMediaTime() - started) * 1000
        recentProc = Array((recentProc + [ms]).suffix(30))
        if counting {
            procMs.append(ms)
            counts.durationS = tUsed
            counts.frames += 1
        }
        if action == .finish {  // the drill's time is up
            endDrill(frame: frame, landmarks: landmarks)
            return
        }
        var snapshot = makeSnapshot(frame, record: record, landmarks: landmarks)
        if counting { snapshot.pacePerMin = counts.averagePacePerMin(minS: cfg.touchPaceMinS) }
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
        let inDrill = flow.state == .drill && drill != nil
        switch flow.state {
        case .ready: s.phase = .ready
        case .waiting: s.phase = .waiting
        case .countdown: s.phase = .countdown(secondsLeft: flow.secondsLeft(lastTS))
        case .drill: s.phase = inDrill ? .drill(elapsedS: counts.durationS, limitS: flow.drillS) : .ready
        }
        s.framingProblems = flow.framing.problems
        if inDrill && !recenter.ready { s.recenterProblems = recenter.problems }
        s.counts = counts
        s.faceLandmarker = headSampler != nil
        if inDrill { s.headDown = headDown }
        let now = counts.durationS
        if inDrill {
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
