// Decodable shapes of the portable test vectors in tests/vectors (written by
// scripts/export_test_vectors.py; format in tests/vectors/README.md). Decoded with
// keyDecodingStrategy = .convertFromSnakeCase, so "t_s" arrives as tS, "ball_xy" as ballXy.
import Foundation
@testable import GaffersVisionCore

struct ObservationJSON: Decodable {
    let tS: Double
    let ballXy: [Double]?
    let ballDiameterPx: Double?
    let searchedFullFrame: Bool
    let leftFoot: [[Double]]
    let rightFoot: [[Double]]
    let headPitchDeg: Double?

    var observation: Observation {
        Observation(
            tS: tS,
            ballXY: ballXy.map { Point(x: $0[0], y: $0[1]) },
            ballDiameterPx: ballDiameterPx,
            searchedFullFrame: searchedFullFrame,
            leftFoot: leftFoot.map { Point(x: $0[0], y: $0[1]) },
            rightFoot: rightFoot.map { Point(x: $0[0], y: $0[1]) },
            headPitchDeg: headPitchDeg
        )
    }
}

struct EventJSON: Decodable {
    let frame: Int?  // metrics vectors only: which observation's step reported it
    let kind: String
    let tS: Double
    let reportedTS: Double
    let foot: String?
    let value: Double?

    enum CodingKeys: String, CodingKey {
        case frame, kind, tS, foot, value
        case reportedTS = "reportedTS"
    }
}

struct TouchRateJSON: Decodable {
    let lastMinute: Int
    let minuteComplete: Bool
    let pacePerMin: Double?
}

struct MetricsVector: Decodable {
    struct Expected: Decodable {
        let events: [EventJSON]
        let finalTouchRate: TouchRateJSON
    }

    let description: String
    let config: Config
    let observations: [ObservationJSON]
    let expected: Expected
}

struct FiltersVector: Decodable {
    struct OneEuroRow: Decodable {
        let tS: Double
        let x: Double
        let out: Double
    }

    struct KalmanRow: Decodable {
        let op: String
        let tS: Double
        let z: Double?
        let out: Double?
    }

    let config: Config
    let oneEuro: [OneEuroRow]
    let kalman: [KalmanRow]
}

struct SearchVector: Decodable {
    struct Row: Decodable {
        struct Expected: Decodable {
            struct State: Decodable {
                let searching: Bool
                let cropMissSinceS: Double?
            }

            let needsFullFrame: Bool
            let detection: [Double]?
            let searchedFullFrame: Bool
            let state: State
        }

        let tS: Double
        let crop: [Int]?
        let cropDet: [Double]?
        let fullDet: [Double]?
        let expected: Expected
    }

    let config: Config
    let sequences: [String: [Row]]
}

struct PipelineVector: Decodable {
    struct Input: Decodable {
        let tS: Double
        let landmarks: [[Double?]]
        let crop: [Int]?
        let cropCandidates: [[Double]]
        let fullCandidates: [[Double]]
        let headPitch: HeadPitch  // .face(value) when the key is present (even null), else .poseEstimate

        enum CodingKeys: String, CodingKey {
            case tS, landmarks, crop, cropCandidates, fullCandidates, headPitchDeg
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            tS = try c.decode(Double.self, forKey: .tS)
            landmarks = try c.decode([[Double?]].self, forKey: .landmarks)
            crop = try c.decodeIfPresent([Int].self, forKey: .crop)
            cropCandidates = try c.decode([[Double]].self, forKey: .cropCandidates)
            fullCandidates = try c.decode([[Double]].self, forKey: .fullCandidates)
            headPitch = c.contains(.headPitchDeg)
                ? .face(try c.decodeIfPresent(Double.self, forKey: .headPitchDeg))
                : .poseEstimate
        }

        var landmarkValues: [Landmark] {
            landmarks.map { row in
                Landmark(x: row[0] ?? .nan, y: row[1] ?? .nan, visibility: row[2] ?? .nan)
            }
        }
    }

    struct Expected: Decodable {
        let ball: [Double]?
        let ballXy: [Double]?
        let predicted: Bool
        let searchedFullFrame: Bool
        let ranFullFrame: Bool
        let events: [EventJSON]
        let touchRate: TouchRateJSON
    }

    struct Frame: Decodable {
        let input: Input
        let expected: Expected
    }

    let config: Config
    let frames: [Frame]
}

func detection(_ values: [Double]) -> Detection {
    Detection(x1: values[0], y1: values[1], x2: values[2], y2: values[3], confidence: values[4])
}

func cropBox(_ values: [Int]) -> CropBox {
    CropBox(x1: values[0], y1: values[1], x2: values[2], y2: values[3])
}
