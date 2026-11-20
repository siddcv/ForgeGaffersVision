// The live screen: the processed camera frame with the overlay (skeleton, foot crop, ball), counts,
// the START / CANCEL / STOP button, framing hints and the summary after STOP. Like src/live.py's
// window, it shows each processed frame, so the overlay always lines up with the picture.
// The front camera's picture and overlay are mirrored for display only; text is not.
import GaffersVisionCore
import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var engine = LiveEngine()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            frameView
            VStack(spacing: 0) {
                topBar
                Spacer()
                if !engine.snapshot.recenterProblems.isEmpty {
                    warning("RECENTER YOURSELF", engine.snapshot.recenterProblems)
                }
                if case .countdown(let left) = engine.snapshot.phase {
                    Text("\(max(1, Int(left.rounded(.up))))")
                        .font(.system(size: 140, weight: .heavy)).foregroundStyle(.orange)
                        .shadow(radius: 6)
                }
                Spacer()
                if engine.snapshot.phase == .ready && !engine.snapshot.startEnabled {
                    warning("Fit head to toe and the ball in frame", engine.snapshot.framingProblems)
                }
                bottomBar
            }
            .padding()
            if let summary = engine.summary {
                SummaryView(summary: summary) { engine.dismissSummary() }
            }
            if let error = engine.error {
                Text(error).padding().background(.red.opacity(0.85)).foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12)).padding()
            }
        }
        .statusBarHidden()
        .task { await engine.start() }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }  // no screen lock mid-drill
    }

    private var frameView: some View {
        let s = engine.snapshot
        return GeometryReader { geo in
            ZStack {
                if let image = s.image {
                    Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fit)
                }
                OverlayView(snapshot: s)
                    .aspectRatio(CGSize(width: s.frameWidth, height: s.frameHeight), contentMode: .fit)
            }
            .scaleEffect(x: engine.mirrored ? -1 : 1, y: 1)
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .ignoresSafeArea()
    }

    private var topBar: some View {
        let s = engine.snapshot
        let c = s.counts
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("TICK-TOCKS  L \(c.tickTocksLeft)  R \(c.tickTocksRight)")
                    .font(.title2.bold())
                    .foregroundStyle(s.touchFlash ? .green : .white)
                Spacer()
                phaseLabel
            }
            Text("contacts  L \(c.contactsLeft)  R \(c.contactsRight)    last minute \(c.lastMinute)"
                 + (s.pacePerMin.map { String(format: "    pace %.0f/min", $0) } ?? ""))
                .font(.subheadline).foregroundStyle(s.contactFlash ? .yellow : .white.opacity(0.85))
            // Pitch is shown for the head-down calibration readings (looking ahead vs at the ball).
            Text(s.headPitchDeg.map { String(format: "pitch %.0f deg", $0) } ?? "pitch n/a")
                .font(.subheadline.monospaced())
                .foregroundStyle(s.headDown ? .red : .white.opacity(0.85))
                + Text(s.faceLandmarker ? "" : "  (pose estimate)").font(.caption).foregroundStyle(.orange)
            if s.headDown {
                Text("HEAD UP!").font(.title.bold()).foregroundStyle(.red)
            }
            if s.lostFlash {
                Text("LOST CONTROL").font(.headline.bold()).foregroundStyle(.red)
            }
        }
        .padding(10)
        .background(.black.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var phaseLabel: some View {
        switch engine.snapshot.phase {
        case .ready:
            Text(engine.snapshot.startEnabled ? "READY" : "STEP INTO FRAME")
                .foregroundStyle(engine.snapshot.startEnabled ? .white : .orange).bold()
        case .countdown:
            Text("GET SET").foregroundStyle(.orange).bold()
        case .drill(let elapsed):
            Label(String(format: "%d:%02d", Int(elapsed) / 60, Int(elapsed) % 60), systemImage: "circle.fill")
                .foregroundStyle(.red).bold()
        }
    }

    private var bottomBar: some View {
        let s = engine.snapshot
        return HStack(alignment: .bottom) {
            VStack(alignment: .leading) {
                Text(String(format: "%.0f fps  proc %.0f ms", s.fps, s.procMs))
                Text(s.searching ? "SEARCH" : "TRACK")
            }
            .font(.caption.monospaced()).foregroundStyle(.white.opacity(0.8))
            Spacer()
            Button(action: engine.switchCamera) {
                Image(systemName: "arrow.triangle.2.circlepath.camera").font(.title2)
                    .padding(12).background(.black.opacity(0.45)).clipShape(Circle())
            }
            .foregroundStyle(.white)
            .disabled(s.phase != .ready)
            mainButton
        }
    }

    private var mainButton: some View {
        let s = engine.snapshot
        let (label, color, enabled): (String, Color, Bool) = {
            switch s.phase {
            case .ready: return ("START", s.startEnabled ? .green : .gray, s.startEnabled)
            case .countdown: return ("CANCEL", .orange, true)
            case .drill: return ("STOP", .red, true)
            }
        }()
        return Button(action: engine.buttonTapped) {
            Text(label).font(.title2.bold()).frame(width: 130, height: 56)
                .background(color).foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .disabled(!enabled)
    }

    private func warning(_ title: String, _ problems: [String]) -> some View {
        VStack(spacing: 4) {
            Text(title).font(.title2.bold())
            Text(problems.joined(separator: ", ")).font(.subheadline)
        }
        .foregroundStyle(.orange)
        .padding(10)
        .background(.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

/// Skeleton, foot crop and ball, drawn in frame pixels scaled to the view.
struct OverlayView: View {
    let snapshot: Snapshot

    private static let bones: [(Int, Int)] = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
        (0, 7), (0, 8),
    ]

    var body: some View {
        Canvas { context, size in
            let s = snapshot
            let k = size.width / s.frameWidth
            func p(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: x * k, y: y * k) }
            func visible(_ i: Int) -> Bool { s.landmarks.indices.contains(i) && s.landmarks[i].visibility >= 0.5 }

            var skeleton = Path()
            for (a, b) in Self.bones where visible(a) && visible(b) {
                skeleton.move(to: p(s.landmarks[a].x, s.landmarks[a].y))
                skeleton.addLine(to: p(s.landmarks[b].x, s.landmarks[b].y))
            }
            context.stroke(skeleton, with: .color(.green.opacity(0.8)), lineWidth: 2)
            for i in PoseIndex.leftFoot + PoseIndex.rightFoot where visible(i) {
                let c = p(s.landmarks[i].x, s.landmarks[i].y)
                let dot = Path(ellipseIn: CGRect(x: c.x - 4, y: c.y - 4, width: 8, height: 8))
                context.fill(dot, with: .color(PoseIndex.leftFoot.contains(i) ? .cyan : .pink))
            }
            if let crop = s.crop {
                let r = CGRect(origin: p(Double(crop.x1), Double(crop.y1)),
                               size: CGSize(width: Double(crop.x2 - crop.x1) * k, height: Double(crop.y2 - crop.y1) * k))
                context.stroke(Path(r), with: .color(.yellow.opacity(0.6)), lineWidth: 1)
            }
            if let ball = s.ball {
                let r = CGRect(origin: p(ball.x1, ball.y1),
                               size: CGSize(width: (ball.x2 - ball.x1) * k, height: (ball.y2 - ball.y1) * k))
                context.stroke(Path(r), with: .color(.orange), lineWidth: 3)
            } else if let xy = s.predictedXY {
                let c = p(xy.x, xy.y)
                context.stroke(Path(ellipseIn: CGRect(x: c.x - 12, y: c.y - 12, width: 24, height: 24)),
                               with: .color(.orange.opacity(0.6)), lineWidth: 2)
            }
        }
    }
}

struct SummaryView: View {
    let summary: DrillSummary
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Drill done").font(.largeTitle.bold())
            row("Time", String(format: "%d:%02d", Int(summary.durationS) / 60, Int(summary.durationS) % 60))
            row("Tick-tocks", "\(summary.tickTocksLeft + summary.tickTocksRight)  (L \(summary.tickTocksLeft), R \(summary.tickTocksRight))")
            row("Contacts", "L \(summary.contactsLeft), R \(summary.contactsRight)")
            row("Lost control", "\(summary.lostControl)")
            row("Head down", "\(summary.headDown)")
            row("Recenter warnings", "\(summary.recenterWarnings)")
            Divider()
            row("Processed", String(format: "%.1f fps, median %.0f ms/frame", summary.processedFps, summary.medianProcMs))
            row("Full-frame searches", "\(summary.fullFrameRuns) of \(summary.frames) frames")
            Button("Next drill", action: close).buttonStyle(.borderedProminent).frame(maxWidth: .infinity)
        }
        .padding(24)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .padding()
    }

    private func row(_ name: String, _ value: String) -> some View {
        HStack { Text(name).foregroundStyle(.secondary); Spacer(); Text(value).bold() }
    }
}
