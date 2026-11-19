// Port of src/config.py: every threshold the per-frame logic uses, durations in seconds.
// Values are the Python defaults; the explanations of each value live in src/config.py, which
// stays the reference. Decodes the "config" object of a test vector (snake_case keys) with
// JSONDecoder.keyDecodingStrategy = .convertFromSnakeCase; keys the phone does not use are ignored.

/// Ball diameter in meters by ball size, the pixels-to-meters ruler.
public let ballDiameterBySize: [Int: Double] = [1: 0.153, 2: 0.170, 3: 0.190, 4: 0.205, 5: 0.220]

/// Which foot made a touch: the foot the ball was travelling toward, or the foot nearest to it.
public enum FootRule: String, Codable, Sendable {
    case direction
    case nearest
}

public struct Config: Codable, Equatable, Sendable {
    // Ball size
    public var ballSize: Int = 5
    public var ballDiameterOverrideM: Double? = nil
    public var diameterSmoothing: Double = 0.2

    // Touches
    public var touchMinSpeedMps: Double = 0.2
    public var touchMinTravelDiam: Double = 0.1
    public var touchFootDistanceDiam: Double = 1.5
    public var touchFootRule: FootRule = .direction
    public var touchMinIntervalS: Double = 0.08
    public var ticktockMaxArrivalS: Double = 1.0
    public var touchMaxGapS: Double = 0.4
    public var speedWindowS: Double = 0.10

    // Touches per minute
    public var touchesPerMinuteGoal: Int = 150
    public var touchMinuteS: Double = 60.0
    public var touchPaceWindowS: Double = 10.0
    public var touchPaceMinS: Double = 3.0

    // Lost control
    public var lostDistanceDiam: Double = 1.5
    public var lostControlMinS: Double = 0.5

    // Head position (Face Landmarker pitch)
    public var headDownPitchDeg: Double = 17.0
    public var headUpPitchDeg: Double = 22.0
    public var headDownMinS: Double = 1.0
    public var headPoseSampleIntervalS: Double = 0.10

    // Ball detection
    public var ballSearchAfterS: Double = 0.15
    public var ballConfidence: Double = 0.15
    public var ballClassId: Int = 32
    public var ballCandidates: Int = 3
    public var ballNmsIou: Double = 0.7
    public var cropImgsz: Int = 320
    public var fullImgsz: Int = 640
    public var cropBallFrac: Double = 0.18
    public var cropSpanMinScale: Double = 1.2
    public var cropSpanScale: Double = 2.0
    public var cropMinFrac: Double = 0.25
    public var cropPadValue: Int = 114

    // Ball tracking on top of detection
    public var trackGateDiam: Double = 2.5
    public var trackGateMaxS: Double = 0.3
    public var trackMaxPredictS: Double = 0.1

    // Pose
    public var poseMinVisibility: Double = 0.5
    public var headBoxPadding: Double = 0.5

    // Filters
    public var kalmanProcessNoise: Double = 4.0e6
    public var kalmanMeasurementNoise: Double = 4.0
    public var oneEuroMinCutoff: Double = 1.5
    public var oneEuroBeta: Double = 0.02
    public var oneEuroDCutoff: Double = 1.0

    public init() {}

    /// The configured ball's diameter: the override if set, else the size table.
    public var ballDiameterM: Double {
        ballDiameterOverrideM ?? ballDiameterBySize[ballSize]!
    }
}
