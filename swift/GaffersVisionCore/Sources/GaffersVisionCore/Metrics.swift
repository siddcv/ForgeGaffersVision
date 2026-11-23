// Port of src/metrics.py: streaming tick-tock metrics. Never reads future frames.
//
// Call Metrics.step(state, observation, config) once per frame. State is a value type, so each
// step returns a new state, as Python's dataclasses.replace does.
//
// Two touch counts:
//   contact  any touch on the ball: a direction change next to a foot. Reported straight away.
//   touch    a tick-tock: a contact by one foot that sends the ball across the midpoint between
//            the feet and whose next contact is by the other foot, within ticktockMaxArrivalS.
//            Stamped at the first foot's contact, reported when the other foot touches the ball.
//
// The functions follow the Python ones line by line; comments explaining the rules are in the
// Python file, which stays the reference. Change Python first, regenerate the test vectors,
// then update this file until they pass.
import Foundation

public enum Foot: String, Codable, Sendable {
    case left
    case right
}

public enum EventKind: String, Codable, Sendable {
    case contact
    case touch  // a tick-tock
    case speed
    case lostControl = "lost_control"
    case reset
    case headDown = "head_down"
    case headUp = "head_up"
}

public struct Observation: Equatable, Sendable {
    public var tS: Double
    public var ballXY: Point? = nil
    public var ballDiameterPx: Double? = nil
    public var searchedFullFrame: Bool = false
    public var leftFoot: [Point] = []  // heel, toe (visible ones only)
    public var rightFoot: [Point] = []
    public var headPitchDeg: Double? = nil

    public init(tS: Double, ballXY: Point? = nil, ballDiameterPx: Double? = nil, searchedFullFrame: Bool = false,
                leftFoot: [Point] = [], rightFoot: [Point] = [], headPitchDeg: Double? = nil) {
        self.tS = tS
        self.ballXY = ballXY
        self.ballDiameterPx = ballDiameterPx
        self.searchedFullFrame = searchedFullFrame
        self.leftFoot = leftFoot
        self.rightFoot = rightFoot
        self.headPitchDeg = headPitchDeg
    }
}

public struct Event: Equatable, Sendable {
    public var kind: EventKind
    public var tS: Double  // when it happened
    public var reportedTS: Double  // when the system knew; the difference is detection latency
    public var foot: Foot? = nil
    public var value: Double? = nil

    public init(_ kind: EventKind, _ tS: Double, _ reportedTS: Double, foot: Foot? = nil, value: Double? = nil) {
        self.kind = kind
        self.tS = tS
        self.reportedTS = reportedTS
        self.foot = foot
        self.value = value
    }
}

/// Distance from the ball to each foot (nil: foot not visible).
public struct FootDistances: Equatable, Sendable {
    public var left: Double?
    public var right: Double?

    public init(left: Double?, right: Double?) {
        self.left = left
        self.right = right
    }

    public static let none = FootDistances(left: nil, right: nil)
}

/// Mean x of each foot (nil: foot not visible).
public struct FeetX: Equatable, Sendable {
    public var left: Double?
    public var right: Double?

    public init(left: Double?, right: Double?) {
        self.left = left
        self.right = right
    }
}

public struct MetricsState: Equatable, Sendable {
    public var diameterPx: Double? = nil
    public var lastBallTS: Double? = nil
    public var lastBallXY: Point? = nil
    public var lastVxMps: Double? = nil
    public var lastFootDistances: FootDistances? = nil
    public var lastFeetX: FeetX? = nil
    public var lastTouchTS: Double? = nil
    public var pendingFoot: Foot? = nil
    public var pendingTS: Double? = nil
    public var pendingCrossed: Bool = false
    public var runStartX: Double? = nil
    public var turnTS: Double? = nil
    public var turnX: Double? = nil
    public var turnFoot: Foot? = nil
    public var turnDirection: Int = 0
    public var firstTouchTS: Double? = nil
    public var recentTouchTS: [Double] = []
    public var speedFoot: Foot? = nil
    public var speedStartTS: Double? = nil
    public var speedEndTS: Double? = nil
    public var speedDistanceM: Double = 0.0
    public var lost: Bool = false
    public var farSinceS: Double? = nil
    public var notFoundSinceS: Double? = nil
    public var headLowSinceS: Double? = nil
    public var headDownSinceS: Double? = nil

    public init() {}

    mutating func clearTurn() {
        turnTS = nil
        turnX = nil
        turnFoot = nil
        turnDirection = 0
    }
}

public struct TouchRate: Equatable, Sendable {
    public var lastMinute: Int  // touches in the last touchMinuteS, the number the goal is set on
    public var minuteComplete: Bool  // a full minute has passed since the first touch
    public var pacePerMin: Double?  // touches in the last touchPaceWindowS, scaled to a minute

    public init(lastMinute: Int, minuteComplete: Bool, pacePerMin: Double?) {
        self.lastMinute = lastMinute
        self.minuteComplete = minuteComplete
        self.pacePerMin = pacePerMin
    }
}

public enum Metrics {
    public static func initialState() -> MetricsState {
        MetricsState()
    }

    /// Add a touch to the rolling window, dropping touches older than a minute.
    public static func recordTouch(_ state: MetricsState, _ tS: Double, _ cfg: Config) -> MetricsState {
        var next = state
        next.recentTouchTS = state.recentTouchTS.filter { tS - $0 < cfg.touchMinuteS } + [tS]
        next.firstTouchTS = state.firstTouchTS ?? tS
        return next
    }

    /// Touches per minute as of tS, for a live readout. Call on any frame.
    public static func touchRate(_ state: MetricsState, _ tS: Double, _ cfg: Config) -> TouchRate {
        guard let first = state.firstTouchTS else {
            return TouchRate(lastMinute: 0, minuteComplete: false, pacePerMin: nil)
        }
        let elapsed = tS - first
        let lastMinute = state.recentTouchTS.filter { 0 <= tS - $0 && tS - $0 < cfg.touchMinuteS }.count
        let window = min(cfg.touchPaceWindowS, elapsed)
        var pace: Double? = nil
        if window >= cfg.touchPaceMinS {
            let inWindow = state.recentTouchTS.filter { 0 <= tS - $0 && tS - $0 <= window }.count
            pace = Double(inWindow) / window * 60
        }
        return TouchRate(lastMinute: lastMinute, minuteComplete: elapsed >= cfg.touchMinuteS, pacePerMin: pace)
    }

    static func smoothDiameter(_ previous: Double?, _ measured: Double?, _ weight: Double) -> Double? {
        guard let measured, measured > 0 else { return previous }
        guard let previous else { return measured }
        return weight * measured + (1 - weight) * previous
    }

    static func footDistance(_ ball: Point, _ foot: [Point]) -> Double? {
        foot.map { distance(ball, $0) }.min()
    }

    /// (foot, distance) for the closer foot, or (nil, nil).
    static func nearerFoot(_ distances: FootDistances) -> (Foot?, Double?) {
        let left = distances.left, right = distances.right
        if left == nil && right == nil {
            return (nil, nil)
        }
        if let left, right == nil || left <= right! {
            return (.left, left)
        }
        return (.right, right)
    }

    static func meanX(_ points: [Point]) -> Double? {
        points.isEmpty ? nil : points.reduce(0.0) { $0 + $1.x } / Double(points.count)
    }

    /// The foot the ball was travelling toward before it reversed, or nil if unclear.
    public static func footByDirection(_ incomingVx: Double?, _ feetX: FeetX?) -> Foot? {
        guard let incomingVx, incomingVx != 0, let feetX,
              let leftX = feetX.left, let rightX = feetX.right, leftX != rightX else {
            return nil
        }
        let aheadIsLargerX = incomingVx > 0
        return (leftX > rightX) == aheadIsLargerX ? .left : .right
    }

    /// Whether the ball is past the midpoint between the feet, on the side away from fromFoot.
    public static func crossedMidpoint(_ ballX: Double, _ leftFoot: [Point], _ rightFoot: [Point], _ fromFoot: Foot) -> Bool {
        guard let leftX = meanX(leftFoot), let rightX = meanX(rightFoot), leftX != rightX else {
            return false
        }
        let midpoint = (leftX + rightX) / 2
        let targetX = fromFoot == .left ? rightX : leftX
        return (ballX - midpoint) * (targetX - midpoint) > 0
    }

    static func isReversal(_ previousVx: Double?, _ vx: Double, _ minSpeed: Double) -> Bool {
        guard let previousVx else { return false }
        return previousVx * vx < 0 && abs(previousVx) >= minSpeed && abs(vx) >= minSpeed
    }

    static func speedEvent(_ state: MetricsState, _ reportedTS: Double) -> Event? {
        guard let start = state.speedStartTS, let end = state.speedEndTS else { return nil }
        let elapsed = end - start
        if elapsed <= 0 {
            return nil
        }
        return Event(.speed, start, reportedTS, foot: state.speedFoot, value: state.speedDistanceM / elapsed)
    }

    /// Velocity, touches and speed from a frame where the ball was detected. `state` holds the
    /// values read (Python's `state`); `next` collects the changes (Python's `updates`).
    static func stepBall(_ stateIn: MetricsState, _ obs: Observation, _ distances: FootDistances,
                         _ cfg: Config) -> (MetricsState, [Event]) {
        var events: [Event] = []
        var state = stateIn
        let ball = obs.ballXY!
        let diameter = state.diameterPx!
        let metersPerPx = cfg.ballDiameterM / diameter
        if let pendingFoot = state.pendingFoot, !state.pendingCrossed,
           crossedMidpoint(ball.x, obs.leftFoot, obs.rightFoot, pendingFoot) {
            state.pendingCrossed = true
        }
        var next = state
        next.lastBallTS = obs.tS
        next.lastBallXY = ball
        next.lastFootDistances = distances
        next.lastFeetX = FeetX(left: meanX(obs.leftFoot), right: meanX(obs.rightFoot))

        guard let lastBallTS = state.lastBallTS, let lastBallXY = state.lastBallXY,
              obs.tS - lastBallTS > 0, obs.tS - lastBallTS <= cfg.touchMaxGapS else {
            // New track: no direction yet, and the run of motion starts here.
            next.lastVxMps = nil
            next.runStartX = ball.x
            next.clearTurn()
            return (next, events)
        }
        let dt = obs.tS - lastBallTS

        let vx = (ball.x - lastBallXY.x) * metersPerPx / dt
        if abs(vx) >= cfg.touchMinSpeedMps {
            next.lastVxMps = vx
        }
        let movedM = distance(ball, lastBallXY) * metersPerPx
        let minTravelPx = cfg.touchMinTravelDiam * diameter
        let runStartX = state.runStartX ?? lastBallXY.x

        if isReversal(state.lastVxMps, vx, cfg.touchMinSpeedMps) {
            let turnX = lastBallXY.x
            let travelledIn = abs(turnX - runStartX)
            next.runStartX = turnX
            next.clearTurn()
            var recent = false
            if let lastTouchTS = state.lastTouchTS {
                recent = lastBallTS - lastTouchTS < cfg.touchMinIntervalS
            }
            // Check both frames, since a foot may hide the ball right at the touch.
            let (before, beforeDistance) = nearerFoot(state.lastFootDistances ?? .none)
            let (after, afterDistance) = nearerFoot(distances)
            let candidates: [(Double, Foot)] = [(beforeDistance, before), (afterDistance, after)]
                .compactMap { pair in pair.1.map { (pair.0!, $0) } }
            let limit = cfg.touchFootDistanceDiam * diameter
            if !candidates.isEmpty && !recent && candidates.map({ $0.0 }).min()! <= limit
                && travelledIn >= minTravelPx {
                var foot: Foot? = nil
                if cfg.touchFootRule == .direction {
                    foot = footByDirection(state.lastVxMps, state.lastFeetX)
                }
                if foot == nil {
                    foot = before ?? candidates.min(by: { $0.0 < $1.0 })!.1
                }
                next.turnTS = lastBallTS
                next.turnX = turnX
                next.turnFoot = foot
                next.turnDirection = vx > 0 ? 1 : -1
            }
        }

        // A turn becomes a contact once the ball has travelled far enough in the new direction.
        var touched = false
        if let turnTS = next.turnTS, let turnX = next.turnX {
            let direction = Double(next.turnDirection)
            if (ball.x - turnX) * direction >= minTravelPx {
                touched = true
                let foot = next.turnFoot
                let touchTS = turnTS
                next.clearTurn()
                if let pending = speedEvent(state, obs.tS) {
                    events.append(pending)
                }
                events.append(Event(.contact, touchTS, obs.tS, foot: foot))
                if let pendingFoot = state.pendingFoot, let pendingTS = state.pendingTS, foot != pendingFoot,
                   state.pendingCrossed, touchTS - pendingTS <= cfg.ticktockMaxArrivalS {
                    events.append(Event(.touch, pendingTS, obs.tS, foot: pendingFoot))
                    let counted = recordTouch(state, pendingTS, cfg)
                    next.recentTouchTS = counted.recentTouchTS
                    next.firstTouchTS = counted.firstTouchTS
                }
                next.pendingFoot = foot
                next.pendingTS = touchTS
                next.pendingCrossed = false
                next.lastTouchTS = touchTS
                next.speedFoot = foot
                // Start after the turnaround frame, whose motion partly cancels out.
                next.speedStartTS = obs.tS
                next.speedEndTS = obs.tS
                next.speedDistanceM = 0.0
            }
        }

        if !touched, let speedStart = state.speedStartTS {
            next.speedEndTS = obs.tS
            next.speedDistanceM = state.speedDistanceM + movedM
            if obs.tS - speedStart >= cfg.speedWindowS {
                if let finished = speedEvent(next, obs.tS) {
                    events.append(finished)
                }
                next.speedStartTS = nil
                next.speedEndTS = nil
                next.speedDistanceM = 0.0
            }
        }
        return (next, events)
    }

    static func stepLostControl(_ stateIn: MetricsState, _ obs: Observation, _ distances: FootDistances,
                                _ cfg: Config) -> (MetricsState, [Event]) {
        var state = stateIn
        if obs.ballXY == nil {
            // A miss in the foot crop proves nothing; only a full-frame search counts.
            if !obs.searchedFullFrame {
                return (state, [])
            }
            let since = state.notFoundSinceS ?? obs.tS
            state.notFoundSinceS = since
            state.farSinceS = nil
            if !state.lost && obs.tS - since >= cfg.lostControlMinS {
                state.lost = true
                return (state, [Event(.lostControl, since, obs.tS)])
            }
            return (state, [])
        }

        state.notFoundSinceS = nil
        let (_, nearest) = nearerFoot(distances)
        guard let nearest else {
            state.farSinceS = nil
            return (state, [])
        }

        if state.lost {
            if nearest <= cfg.touchFootDistanceDiam * state.diameterPx! {
                state.lost = false
                state.farSinceS = nil
                return (state, [Event(.reset, obs.tS, obs.tS)])
            }
            return (state, [])
        }

        if nearest <= cfg.lostDistanceDiam * state.diameterPx! {
            state.farSinceS = nil
            return (state, [])
        }
        let since = state.farSinceS ?? obs.tS
        state.farSinceS = since
        if obs.tS - since >= cfg.lostControlMinS {
            state.lost = true
            return (state, [Event(.lostControl, since, obs.tS)])
        }
        return (state, [])
    }

    static func stepHead(_ stateIn: MetricsState, _ obs: Observation, _ cfg: Config) -> (MetricsState, [Event]) {
        var state = stateIn
        guard let pitch = obs.headPitchDeg else { return (state, []) }

        if let downSince = state.headDownSinceS {
            if pitch >= cfg.headDownPitchDeg {
                let duration = obs.tS - downSince
                state.headDownSinceS = nil
                state.headLowSinceS = nil
                return (state, [Event(.headUp, obs.tS, obs.tS, value: duration)])
            }
            return (state, [])
        }

        if pitch >= cfg.headDownPitchDeg {
            state.headLowSinceS = nil
            return (state, [])
        }
        let since = state.headLowSinceS ?? obs.tS
        state.headLowSinceS = since
        if obs.tS - since >= cfg.headDownMinS {
            state.headDownSinceS = since
            return (state, [Event(.headDown, since, obs.tS)])
        }
        return (state, [])
    }

    /// Drop the pending contact once the other foot can no longer confirm it.
    static func expirePending(_ stateIn: MetricsState, _ tS: Double, _ cfg: Config) -> MetricsState {
        var state = stateIn
        guard state.pendingFoot != nil, let pendingTS = state.pendingTS else { return state }
        var answerComing = false
        if let turnTS = state.turnTS {
            answerComing = turnTS - pendingTS <= cfg.ticktockMaxArrivalS
        }
        if state.lost || (tS - pendingTS > cfg.ticktockMaxArrivalS && !answerComing) {
            state.pendingFoot = nil
            state.pendingTS = nil
            state.pendingCrossed = false
        }
        return state
    }

    /// Advance one frame. Returns (new state, events).
    public static func step(_ stateIn: MetricsState, _ obs: Observation, _ cfg: Config) -> (MetricsState, [Event]) {
        var events: [Event] = []
        var state = stateIn
        state.diameterPx = smoothDiameter(state.diameterPx, obs.ballDiameterPx, cfg.diameterSmoothing)
        var distances = FootDistances.none
        if let ball = obs.ballXY, state.diameterPx != nil {
            distances = FootDistances(left: footDistance(ball, obs.leftFoot), right: footDistance(ball, obs.rightFoot))
            let (afterBall, ballEvents) = stepBall(state, obs, distances, cfg)
            state = afterBall
            events += ballEvents
        }
        let (afterLost, lostEvents) = stepLostControl(state, obs, distances, cfg)
        state = afterLost
        events += lostEvents
        state = expirePending(state, obs.tS, cfg)
        let (afterHead, headEvents) = stepHead(state, obs, cfg)
        state = afterHead
        events += headEvents
        return (state, events)
    }
}
