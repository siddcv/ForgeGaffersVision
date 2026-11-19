// swift-tools-version:5.9
// The per-frame tick-tock logic for the phone app: a Swift port of the Python reference in src/
// (metrics.py, filters.py, ball.py search and tracking, pipeline.py, config.py). Plain Swift and
// Foundation only, so it builds on macOS, iOS, Linux and Windows. The tests replay the portable
// test vectors in ../../tests/vectors, written by the Python code; see README.md.
import PackageDescription

let package = Package(
    name: "GaffersVisionCore",
    products: [
        .library(name: "GaffersVisionCore", targets: ["GaffersVisionCore"]),
    ],
    targets: [
        .target(name: "GaffersVisionCore"),
        .testTarget(name: "GaffersVisionCoreTests", dependencies: ["GaffersVisionCore"]),
    ]
)
