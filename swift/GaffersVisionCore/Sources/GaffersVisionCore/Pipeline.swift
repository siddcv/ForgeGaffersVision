// Port of src/pipeline.py: everything after the neural networks, once per frame, in this order:
//   1. One Euro smoothing of the heel and toe keypoints
//   2. the tracker picks the best crop candidate
//   3. full-frame candidates are requested only if the search switch needs them
//   4. the search switch chooses this frame's detection
//   5. Kalman tracker update
//   6. Metrics.step
// The app runs the models and passes their outputs in. Full-frame candidates come as a closure,
// called only when the search switch needs them, so the full-frame detector runs only then.
import Foundation

/// Where this frame's head pitch comes from.
public enum HeadPitch: Equatable, Sendable {
    /// The Face Landmarker pitch (nil: no face found at the latest sample). The head thresholds
    /// in Config are calibrated for this.
    case face(Double?)
    /// The rough pose-only estimate (coarseHeadPitchDeg), when Face Landmarker is not available.
    case poseEstimate
}

/// What one frame produced (the dict FramePipeline.step returns in Python).
public struct FrameRecord: Equatable, Sendable {
    public var tS: Double
    public var crop: CropBox?
    public var ball: Detection?  // the chosen detection
    public var ballXY: Point?  // tracked position (may be a prediction)
    public var predicted: Bool
    public var searchedFullFrame: Bool
    public var ranFullFrame: Bool  // the full-frame detector was asked for this frame
    public var events: [Event]
    public var touchRate: TouchRate
    public var headPitchDeg: Double?
}

public final class FramePipeline {
    public let cfg: Config
    public private(set) var search = SearchState()
    public let tracker: BallTracker
    public private(set) var metricsState = Metrics.initialState()
    private var euro: [Int: OneEuroFilter] = [:]  // key: landmark index * 2 + axis

    public init(_ cfg: Config) {
        self.cfg = cfg
        tracker = BallTracker(cfg)
    }

    /// Crop for this frame's ball detector, sized from the tracked ball's diameter. The app calls
    /// this between pose and ball detection.
    public func footCrop(_ landmarks: [Landmark], frameWidth: Int, frameHeight: Int) -> CropBox? {
        let feet = visiblePoints(landmarks, PoseIndex.leftFoot + PoseIndex.rightFoot, cfg.poseMinVisibility)
        return GaffersVisionCore.footCrop(feet, frameWidth: frameWidth, frameHeight: frameHeight, cfg,
                                          ballDiameterPx: tracker.diameterPx)
    }

    private func smoothPoints(_ indices: [Int], _ landmarks: [Landmark], _ tS: Double) -> [Point] {
        var smoothed: [Point] = []
        for index in indices {
            // Written as "not >=" so a frame with no pose (NaN) is skipped too: one NaN fed to a
            // One Euro filter stays in its state and turns every later foot position into NaN.
            if !(landmarks[index].visibility >= cfg.poseMinVisibility) {
                continue
            }
            let values = [landmarks[index].x, landmarks[index].y]
            var xy: [Double] = []
            for axis in 0..<2 {
                let key = index * 2 + axis
                let filter = euro[key] ?? OneEuroFilter(minCutoff: cfg.oneEuroMinCutoff, beta: cfg.oneEuroBeta,
                                                        dCutoff: cfg.oneEuroDCutoff)
                euro[key] = filter
                xy.append(filter.update(tS, values[axis]))
            }
            smoothed.append(Point(x: xy[0], y: xy[1]))
        }
        return smoothed
    }

    /// Process one frame. landmarks: the 33 pose landmarks in full-frame pixels (all .missing when
    /// no person was found); candidates: detector boxes in full-frame pixels, best first.
    public func step(tS: Double, landmarks: [Landmark], crop: CropBox?, cropCandidates: [Detection],
                     fullCandidates: () -> [Detection], headPitch: HeadPitch = .poseEstimate) -> FrameRecord {
        let headPitchDeg: Double?
        switch headPitch {
        case .face(let pitch):
            headPitchDeg = pitch
        case .poseEstimate:
            headPitchDeg = coarseHeadPitchDeg(landmarks, cfg.poseMinVisibility)
        }
        let leftFoot = smoothPoints(PoseIndex.leftFoot, landmarks, tS)
        let rightFoot = smoothPoints(PoseIndex.rightFoot, landmarks, tS)
        let feet = leftFoot + rightFoot

        let cropDet = tracker.choose(tS, cropCandidates, feet: feet)
        var fullDet: Detection? = nil
        let ranFullFrame = needsFullFrame(search, tS, crop, cropDet, cfg)
        if ranFullFrame {
            fullDet = tracker.choose(tS, fullCandidates(), feet: feet)
        }
        let (nextSearch, detection, searched) = searchStep(search, tS, crop, cropDet, fullDet, cfg)
        search = nextSearch
        let (ballXY, predicted) = tracker.update(tS, detection, searchedFullFrame: searched)

        let observation = Observation(
            tS: tS,
            // Metrics see detections only: snapping from a prediction back to the next detection
            // looks like a direction change and adds false touches.
            ballXY: predicted ? nil : ballXY,
            ballDiameterPx: detection?.diameterPx,
            searchedFullFrame: searched,
            leftFoot: leftFoot,
            rightFoot: rightFoot,
            headPitchDeg: headPitchDeg
        )
        let (nextState, events) = Metrics.step(metricsState, observation, cfg)
        metricsState = nextState
        return FrameRecord(
            tS: tS, crop: crop, ball: detection, ballXY: ballXY, predicted: predicted,
            searchedFullFrame: searched, ranFullFrame: ranFullFrame, events: events,
            touchRate: Metrics.touchRate(metricsState, tS, cfg), headPitchDeg: headPitchDeg
        )
    }
}
