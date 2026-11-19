// Port of the plain-code parts of src/ball.py: foot crop, crop-to-frame mapping, the crop /
// full-frame search switch and the ball tracker. Running the detector itself (Core ML on the
// phone) is outside this package.
import Foundation

/// Square integer crop centred on the feet; it may extend past the frame edge. Once the ball has
/// been seen, the side is set so the ball fills about cfg.cropBallFrac of it (ball.foot_crop).
public func footCrop(_ footPoints: [Point], frameWidth: Int, frameHeight: Int, _ cfg: Config,
                     ballDiameterPx: Double? = nil) -> CropBox? {
    guard !footPoints.isEmpty else { return nil }
    let xs = footPoints.map(\.x)
    let ys = footPoints.map(\.y)
    let span = max(xs.max()! - xs.min()!, ys.max()! - ys.min()!)
    let sideValue: Double
    if let diameter = ballDiameterPx, diameter != 0 {
        sideValue = max(diameter / cfg.cropBallFrac, span * cfg.cropSpanMinScale)
    } else {
        sideValue = max(span * cfg.cropSpanScale, cfg.cropMinFrac * Double(frameHeight))
    }
    let side = Int(sideValue)  // truncates toward zero, as Python int()
    let centerX = (xs.min()! + xs.max()!) / 2
    let centerY = (ys.min()! + ys.max()!) / 2
    let x1 = Int(centerX - Double(side) / 2)
    let y1 = Int(centerY - Double(side) / 2)
    let x2 = x1 + side
    let y2 = y1 + side
    // Drop crops that lie entirely outside the frame.
    if x2 <= 0 || y2 <= 0 || x1 >= frameWidth || y1 >= frameHeight {
        return nil
    }
    return CropBox(x1: x1, y1: y1, x2: x2, y2: y2)
}

/// A detection made on the crop, moved to full-frame pixels (ball.to_full_frame). The detector's
/// boxes must first be scaled from model input pixels to crop pixels (docs/PREPROCESSING.md).
public func toFullFrame(_ detection: Detection, _ crop: CropBox) -> Detection {
    Detection(x1: detection.x1 + Double(crop.x1), y1: detection.y1 + Double(crop.y1),
              x2: detection.x2 + Double(crop.x1), y2: detection.y2 + Double(crop.y1),
              confidence: detection.confidence)
}

func inside(_ detection: Detection, _ crop: CropBox?) -> Bool {
    guard let crop else { return false }
    let center = detection.center
    return Double(crop.x1) <= center.x && center.x <= Double(crop.x2)
        && Double(crop.y1) <= center.y && center.y <= Double(crop.y2)
}

public struct SearchState: Equatable, Sendable {
    public var searching: Bool = false
    public var cropMissSinceS: Double? = nil

    public init(searching: Bool = false, cropMissSinceS: Double? = nil) {
        self.searching = searching
        self.cropMissSinceS = cropMissSinceS
    }
}

/// True when searchStep will read the full-frame detection this frame, so the app runs the
/// full-frame detector only then. Must stay in step with the branches of searchStep.
public func needsFullFrame(_ state: SearchState, _ tS: Double, _ crop: CropBox?, _ cropDet: Detection?,
                           _ cfg: Config) -> Bool {
    if state.searching {
        return true
    }
    if cropDet != nil {
        return false
    }
    let missSince = state.cropMissSinceS ?? tS
    return crop == nil || tS - missSince >= cfg.ballSearchAfterS
}

/// Choose this frame's detection: returns (state, detection, searchedFullFrame). fullDet is only
/// read when needsFullFrame is true; pass nil otherwise.
public func searchStep(_ stateIn: SearchState, _ tS: Double, _ crop: CropBox?, _ cropDet: Detection?,
                       _ fullDet: Detection?, _ cfg: Config) -> (SearchState, Detection?, Bool) {
    var state = stateIn
    if state.searching {
        guard let fullDet else { return (state, nil, true) }
        if inside(fullDet, crop) {
            return (SearchState(), fullDet, true)
        }
        return (state, fullDet, true)
    }

    if let cropDet {
        return (SearchState(), cropDet, false)
    }

    let missSince = state.cropMissSinceS ?? tS
    state.cropMissSinceS = missSince
    if crop == nil || tS - missSince >= cfg.ballSearchAfterS {
        state.searching = true
        return (state, fullDet, true)
    }
    return (state, nil, false)
}

/// Causal track on top of per-frame detections (ball.BallTracker). choose() picks the candidate
/// nearest the predicted position, within trackGateDiam ball diameters, or with no recent track
/// the one nearest the feet. update() smooths it with a Kalman filter and predicts through gaps
/// up to trackMaxPredictS, never once a full-frame search has started.
public final class BallTracker {
    public let cfg: Config
    private let kx: Kalman1D
    private let ky: Kalman1D
    public private(set) var diameterPx: Double? = nil
    public private(set) var lastSeenS: Double? = nil

    public init(_ cfg: Config) {
        self.cfg = cfg
        kx = Kalman1D(processNoise: cfg.kalmanProcessNoise, measurementNoise: cfg.kalmanMeasurementNoise)
        ky = Kalman1D(processNoise: cfg.kalmanProcessNoise, measurementNoise: cfg.kalmanMeasurementNoise)
    }

    private func tracking(_ tS: Double, _ limitS: Double) -> Bool {
        guard let lastSeenS else { return false }
        return tS - lastSeenS <= limitS
    }

    public func choose(_ tS: Double, _ candidates: [Detection], feet: [Point] = []) -> Detection? {
        guard !candidates.isEmpty else { return nil }
        if tracking(tS, cfg.trackGateMaxS) {
            let expected = Point(x: kx.peek(tS)!, y: ky.peek(tS)!)
            let best = candidates.min(by: { distance($0.center, expected) < distance($1.center, expected) })!
            let gate = cfg.trackGateDiam * diameterPx!
            return distance(best.center, expected) <= gate ? best : nil
        }
        if !feet.isEmpty {
            let n = Double(feet.count)
            let center = Point(x: feet.reduce(0.0) { $0 + $1.x } / n, y: feet.reduce(0.0) { $0 + $1.y } / n)
            return candidates.min(by: { distance($0.center, center) < distance($1.center, center) })
        }
        return candidates.max(by: { $0.confidence < $1.confidence })
    }

    /// Returns (ball position or nil, predicted).
    public func update(_ tS: Double, _ detection: Detection?, searchedFullFrame: Bool) -> (Point?, Bool) {
        if let detection {
            let center = detection.center
            diameterPx = detection.diameterPx
            lastSeenS = tS
            return (Point(x: kx.update(tS, center.x), y: ky.update(tS, center.y)), false)
        }
        if searchedFullFrame || !tracking(tS, cfg.trackMaxPredictS) {
            return (nil, false)
        }
        return (Point(x: kx.predict(tS)!, y: ky.predict(tS)!), true)
    }
}
