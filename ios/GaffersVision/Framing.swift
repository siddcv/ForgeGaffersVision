// Port of the framing check in src/live.py (framing_problems, FramingGate, DrillFlow): after START,
// the countdown waits until the whole player (head to toe) and the ball are in frame; mid-drill, RECENTER YOURSELF
// shows while the head or a foot is out. Values mirror src/config.py (framing_*); change them
// there first. Not in GaffersVisionCore's Config, because the test vectors' config predates them.
import GaffersVisionCore

enum FramingSettings {
    static let marginFrac = 0.02  // framing_margin_frac
    static let headTopFrac = 0.8  // framing_head_top_frac
    static let holdS = 0.5  // framing_hold_s
    static let graceS = 0.5  // framing_grace_s
}

private let face = Array(0..<11)

/// What keeps the player and ball from being fully in frame; empty when they are.
func framingProblems(_ landmarks: [Landmark], ball: Detection?, width: Double, height: Double,
                     minVisibility: Double) -> [String] {
    let marginX = FramingSettings.marginFrac * width, marginY = FramingSettings.marginFrac * height
    func inside(_ x: Double, _ y: Double) -> Bool {
        x >= marginX && x <= width - marginX && y >= marginY && y <= height - marginY
    }
    func visible(_ i: Int) -> Bool { landmarks[i].visibility >= minVisibility }  // false for NaN

    guard landmarks.indices.contains(where: visible) else { return ["no player found"] }
    var problems: [String] = []
    let faceIdx = face.filter(visible)
    if faceIdx.isEmpty {
        problems.append("head not in frame")
    } else {
        let xs = faceIdx.map { landmarks[$0].x }, ys = faceIdx.map { landmarks[$0].y }
        let headTop = ys.min()! - FramingSettings.headTopFrac * (xs.max()! - xs.min()!)
        if !zip(xs, ys).allSatisfy({ inside($0, $1) }) || headTop < marginY {
            problems.append("head not in frame")
        }
    }
    for (name, foot) in [("left foot", PoseIndex.leftFoot), ("right foot", PoseIndex.rightFoot)] {
        if !foot.allSatisfy({ visible($0) && inside(landmarks[$0].x, landmarks[$0].y) }) {
            problems.append("\(name) not in frame")
        }
    }
    if let ball {
        if !(inside(ball.x1, ball.y1) && inside(ball.x2, ball.y2)) { problems.append("ball not in frame") }
    } else {
        problems.append("ball not found")
    }
    return problems
}

func bodyProblems(_ problems: [String]) -> [String] {
    problems.filter { !$0.hasPrefix("ball") }
}

/// In frame for holdS before `ready`, out for graceS before not ready, so one missed detection
/// does not make the button or the warning flicker.
struct FramingGate {
    private(set) var ready: Bool
    private(set) var problems: [String]
    private var since: Double? = nil

    init(ready: Bool = false) {
        self.ready = ready
        problems = ready ? [] : ["no player found"]
    }

    mutating func update(_ tS: Double, _ newProblems: [String]) {
        let inFrame = newProblems.isEmpty
        if !inFrame { problems = newProblems }
        if inFrame != ready {
            if since == nil { since = tS }
            let wait = inFrame ? FramingSettings.holdS : FramingSettings.graceS
            if tS - since! >= wait {
                ready = inFrame
                since = nil
            }
        } else {
            since = nil
        }
    }
}

/// Port of DrillFlow in src/live.py: when a drill starts and stops, so a player can press START at
/// the phone and then walk into place. READY -> START -> WAITING until the player (head to toe) and
/// the ball have been fully in frame for holdS -> COUNTDOWN -> DRILL -> READY. Out of frame for
/// graceS during the countdown goes back to WAITING. A drill ends at STOP or, when drillS is set,
/// by itself after drillS seconds. START pressed again (CANCEL) leaves WAITING or COUNTDOWN.
struct DrillFlow {
    enum State: Equatable { case ready, waiting, countdown, drill }
    enum Action { case start, finish }

    let countdownS: Double
    var drillS: Double?  // nil: open-ended
    private(set) var state = State.ready
    private(set) var countdownEnd: Double?
    private(set) var framing = FramingGate()  // player and ball in frame (waiting and countdown)

    init(countdownS: Double = 3.0, drillS: Double? = 60.0) {
        self.countdownS = countdownS
        self.drillS = drillS
    }

    /// START / CANCEL / STOP, by state.
    mutating func press(_ tS: Double) -> Action? {
        switch state {
        case .ready:
            framing = FramingGate()
            state = .waiting
        case .waiting, .countdown:
            state = .ready
            countdownEnd = nil
        case .drill:
            state = .ready
            return .finish
        }
        return nil
    }

    /// Once per frame, with this frame's framing problems and, during a drill, its elapsed time.
    mutating func update(_ tS: Double, _ problems: [String], drillElapsedS: Double? = nil) -> Action? {
        switch state {
        case .ready:
            return nil
        case .drill:
            if let drillS, let elapsed = drillElapsedS, elapsed >= drillS {
                state = .ready
                return .finish
            }
            return nil
        case .waiting, .countdown:
            framing.update(tS, problems)
            if state == .waiting && framing.ready {
                state = .countdown
                countdownEnd = tS + countdownS
            } else if state == .countdown && !framing.ready {
                state = .waiting
                countdownEnd = nil
            } else if state == .countdown, let end = countdownEnd, tS >= end {
                state = .drill
                countdownEnd = nil
                return .start
            }
            return nil
        }
    }

    func secondsLeft(_ tS: Double) -> Double { countdownEnd.map { max(0, $0 - tS) } ?? 0 }
}
