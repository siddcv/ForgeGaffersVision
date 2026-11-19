// Replays every portable test vector (tests/vectors, written by the Python code) through the
// Swift port. Passing means: strings, booleans, integers, nils and counts exactly equal; numbers
// within 1e-6 + 1e-9 * |expected| (tests/vectors/README.md, the same tolerance as the Python
// replay in tests/test_vectors.py).
import Foundation
import XCTest
@testable import GaffersVisionCore

let absTolerance = 1e-6
let relTolerance = 1e-9

/// The repository root: this file is <repo>/swift/GaffersVisionCore/Tests/GaffersVisionCoreTests/.
let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
let vectorsDir = repoRoot.appendingPathComponent("tests").appendingPathComponent("vectors")

func load<T: Decodable>(_ type: T.Type, _ url: URL) throws -> T {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(T.self, from: Data(contentsOf: url))
}

func close(_ expected: Double, _ actual: Double) -> Bool {
    abs(expected - actual) <= absTolerance + relTolerance * abs(expected)
}

func assertClose(_ expected: Double?, _ actual: Double?, _ where_: String,
                 file: StaticString = #filePath, line: UInt = #line) {
    switch (expected, actual) {
    case (nil, nil):
        return
    case let (e?, a?):
        XCTAssertTrue(close(e, a), "\(where_): expected \(e), got \(a)", file: file, line: line)
    default:
        XCTFail("\(where_): expected \(String(describing: expected)), got \(String(describing: actual))",
                file: file, line: line)
    }
}

func assertEvents(_ expected: [EventJSON], _ actual: [(frame: Int?, event: Event)], _ where_: String,
                  file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(expected.count, actual.count, "\(where_): event count", file: file, line: line)
    for (i, (e, a)) in zip(expected, actual).enumerated() {
        let at = "\(where_) event \(i) (\(e.kind) at \(e.tS))"
        if let frame = e.frame {
            XCTAssertEqual(frame, a.frame, "\(at): frame", file: file, line: line)
        }
        XCTAssertEqual(e.kind, a.event.kind.rawValue, "\(at): kind", file: file, line: line)
        XCTAssertEqual(e.foot, a.event.foot?.rawValue, "\(at): foot", file: file, line: line)
        assertClose(e.tS, a.event.tS, "\(at): t_s", file: file, line: line)
        assertClose(e.reportedTS, a.event.reportedTS, "\(at): reported_t_s", file: file, line: line)
        assertClose(e.value, a.event.value, "\(at): value", file: file, line: line)
    }
}

func assertTouchRate(_ expected: TouchRateJSON, _ actual: TouchRate, _ where_: String,
                     file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(expected.lastMinute, actual.lastMinute, "\(where_): last_minute", file: file, line: line)
    XCTAssertEqual(expected.minuteComplete, actual.minuteComplete, "\(where_): minute_complete", file: file, line: line)
    assertClose(expected.pacePerMin, actual.pacePerMin, "\(where_): pace_per_min", file: file, line: line)
}

func assertDetection(_ expected: [Double]?, _ actual: Detection?, _ where_: String,
                     file: StaticString = #filePath, line: UInt = #line) {
    guard let expected else {
        XCTAssertNil(actual, "\(where_): expected no detection", file: file, line: line)
        return
    }
    guard let actual else {
        XCTFail("\(where_): expected \(expected), got none", file: file, line: line)
        return
    }
    let values = [actual.x1, actual.y1, actual.x2, actual.y2, actual.confidence]
    for (e, a) in zip(expected, values) {
        assertClose(e, a, where_, file: file, line: line)
    }
}

final class VectorTests: XCTestCase {
    func testVectorsDirectoryIsFound() throws {
        XCTAssertTrue(FileManager.default.fileExists(atPath: vectorsDir.path), "no vectors at \(vectorsDir.path)")
    }

    /// The 25 metric scenarios: Metrics.step on scripted observations.
    func testMetricsVectors() throws {
        let dir = vectorsDir.appendingPathComponent("metrics")
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertGreaterThanOrEqual(files.count, 25)
        for url in files {
            let name = url.deletingPathExtension().lastPathComponent
            let vector = try load(MetricsVector.self, url)
            var state = Metrics.initialState()
            var events: [(frame: Int?, event: Event)] = []
            let observations = vector.observations.map(\.observation)
            for (frame, observation) in observations.enumerated() {
                let (next, new) = Metrics.step(state, observation, vector.config)
                state = next
                events += new.map { (frame: frame, event: $0) }
            }
            assertEvents(vector.expected.events, events, name)
            assertTouchRate(vector.expected.finalTouchRate,
                            Metrics.touchRate(state, observations.last!.tS, vector.config), name)
        }
    }

    /// One Euro and Kalman, including peek / predict through a gap.
    func testFiltersVector() throws {
        let vector = try load(FiltersVector.self, vectorsDir.appendingPathComponent("filters.json"))
        let cfg = vector.config
        let euro = OneEuroFilter(minCutoff: cfg.oneEuroMinCutoff, beta: cfg.oneEuroBeta, dCutoff: cfg.oneEuroDCutoff)
        for (i, row) in vector.oneEuro.enumerated() {
            assertClose(row.out, euro.update(row.tS, row.x), "one_euro[\(i)]")
        }
        let kalman = Kalman1D(processNoise: cfg.kalmanProcessNoise, measurementNoise: cfg.kalmanMeasurementNoise)
        for (i, row) in vector.kalman.enumerated() {
            let out: Double?
            switch row.op {
            case "update": out = kalman.update(row.tS, row.z!)
            case "predict": out = kalman.predict(row.tS)
            case "peek": out = kalman.peek(row.tS)
            default: XCTFail("unknown op \(row.op)"); continue
            }
            assertClose(row.out, out, "kalman[\(i)] \(row.op)")
        }
    }

    /// The crop / full-frame search switch.
    func testSearchStepVector() throws {
        let vector = try load(SearchVector.self, vectorsDir.appendingPathComponent("search_step.json"))
        for (name, rows) in vector.sequences.sorted(by: { $0.key < $1.key }) {
            var state = SearchState()
            for (i, row) in rows.enumerated() {
                let at = "\(name)[\(i)]"
                let crop = row.crop.map(cropBox)
                let cropDet = row.cropDet.map(detection)
                let needs = needsFullFrame(state, row.tS, crop, cropDet, vector.config)
                let (next, chosen, searched) = searchStep(state, row.tS, crop, cropDet,
                                                          needs ? row.fullDet.map(detection) : nil, vector.config)
                state = next
                XCTAssertEqual(row.expected.needsFullFrame, needs, "\(at): needs_full_frame")
                assertDetection(row.expected.detection, chosen, "\(at): detection")
                XCTAssertEqual(row.expected.searchedFullFrame, searched, "\(at): searched_full_frame")
                XCTAssertEqual(row.expected.state.searching, state.searching, "\(at): state.searching")
                assertClose(row.expected.state.cropMissSinceS, state.cropMissSinceS, "\(at): state.crop_miss_since_s")
            }
        }
    }

    /// Everything after the models, on 10 s of a real clip: FramePipeline.step frame by frame.
    func testPipelineVectors() throws {
        let dir = vectorsDir.appendingPathComponent("pipeline")
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        XCTAssertFalse(files.isEmpty)
        for url in files {
            let name = url.deletingPathExtension().lastPathComponent
            let vector = try load(PipelineVector.self, url)
            let pipeline = FramePipeline(vector.config)
            var failuresBefore = 0
            for (i, frame) in vector.frames.enumerated() {
                let at = "\(name) frame \(i) (t \(frame.input.tS))"
                let given = frame.input
                var fullCandidatesUsed = false
                let record = pipeline.step(
                    tS: given.tS, landmarks: given.landmarkValues, crop: given.crop.map(cropBox),
                    cropCandidates: given.cropCandidates.map(detection),
                    fullCandidates: { fullCandidatesUsed = true; return given.fullCandidates.map(detection) },
                    headPitch: given.headPitch
                )
                let expected = frame.expected
                assertDetection(expected.ball, record.ball, "\(at): ball")
                assertClose(expected.ballXy?[0], record.ballXY?.x, "\(at): ball_xy x")
                assertClose(expected.ballXy?[1], record.ballXY?.y, "\(at): ball_xy y")
                XCTAssertEqual(expected.predicted, record.predicted, "\(at): predicted")
                XCTAssertEqual(expected.searchedFullFrame, record.searchedFullFrame, "\(at): searched_full_frame")
                XCTAssertEqual(expected.ranFullFrame, record.ranFullFrame, "\(at): ran_full_frame")
                XCTAssertEqual(record.ranFullFrame, fullCandidatesUsed, "\(at): full-frame candidates read only when needed")
                assertEvents(expected.events, record.events.map { (frame: nil, event: $0) }, at)
                assertTouchRate(expected.touchRate, record.touchRate, at)
                // Stop at the first frame that goes wrong: every later frame inherits its state.
                let failures = testRun?.failureCount ?? 0
                if failures > failuresBefore {
                    XCTFail("\(name): first mismatch at frame \(i); later frames not checked")
                    break
                }
                failuresBefore = failures
            }
        }
    }
}
