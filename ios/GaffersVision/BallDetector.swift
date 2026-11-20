// Ball detector: the exported YOLO models (Core ML, no NMS inside) on the foot crop (320) and,
// only in search mode, on the letterboxed full frame (640). Image preparation, decoding and the
// mapping back to full-frame pixels follow docs/PREPROCESSING.md and src/onnx_detector.py exactly:
//   crop:  square crop, gray 114 past the frame edge, bilinear to 320 x 320, no letterbox
//   full:  longer side to 640 (bilinear), gray 114 letterbox, odd padding pixel right / bottom
//   both:  RGB, pixel / 255 (Core ML's image input does the scaling), raw 1 x 84 x N output,
//          ball score row 4 + ballClassId, threshold, greedy NMS, keep ballCandidates
import CoreImage
import CoreML
import CoreVideo
import GaffersVisionCore

final class BallDetector {
    private let cfg: Config
    private let models: [Int: (model: MLModel, input: String, output: String)]
    private var buffers: [Int: CVPixelBuffer] = [:]
    // No colour management: pixel values go to the model unchanged, as in OpenCV.
    private let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(),
                                              .cacheIntermediates: false])
    private let gray: CIImage

    init(cfg: Config) throws {
        self.cfg = cfg
        let value = CGFloat(cfg.cropPadValue) / 255
        gray = CIImage(color: CIColor(red: value, green: value, blue: value))
        var loaded: [Int: (MLModel, String, String)] = [:]
        for size in Set([cfg.cropImgsz, cfg.fullImgsz]) {
            loaded[size] = try BallDetector.load("yolov8n_\(size)")
        }
        models = loaded.mapValues { (model: $0.0, input: $0.1, output: $0.2) }
    }

    private static func load(_ name: String) throws -> (MLModel, String, String) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all  // Neural Engine when it can
        let url: URL
        if let compiled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            url = compiled
        } else if let package = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            url = try MLModel.compileModel(at: package)  // only if Xcode copied it uncompiled
        } else {
            throw ModelError.missing("\(name).mlpackage")
        }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let description = model.modelDescription
        guard let input = description.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key,
              let output = description.outputDescriptionsByName.first(where: { $0.value.type == .multiArray })?.key
        else {
            throw ModelError.badOutput("\(name): expected an image input and a multi-array output")
        }
        return (model, input, output)
    }

    // MARK: Foot crop (320)

    /// Detections in full-frame pixels, best first. crop may extend past the frame edge.
    func detectCrop(_ frame: CVPixelBuffer, crop: CropBox) -> [Detection] {
        let size = cfg.cropImgsz
        let frameHeight = CGFloat(CVPixelBufferGetHeight(frame))
        let side = CGFloat(crop.x2 - crop.x1)
        guard side > 0 else { return [] }
        // Core Image's origin is bottom-left: the crop's rectangle in its coordinates.
        let rect = CGRect(x: CGFloat(crop.x1), y: frameHeight - CGFloat(crop.y2), width: side, height: side)
        let source = CIImage(cvPixelBuffer: frame)
        let picture = source.cropped(to: rect).composited(over: gray.cropped(to: rect))
            .samplingLinear()
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(size) / side, y: CGFloat(size) / side))
        guard let boxes = run(picture, size: size) else { return [] }
        let k = Double(side) / Double(size)
        return boxes.map { box in
            toFullFrame(Detection(x1: box.x1 * k, y1: box.y1 * k, x2: box.x2 * k, y2: box.y2 * k,
                                  confidence: box.confidence), crop)
        }
    }

    // MARK: Full frame (640, search mode only)

    func detectFull(_ frame: CVPixelBuffer) -> [Detection] {
        let size = Double(cfg.fullImgsz)
        let width = Double(CVPixelBufferGetWidth(frame)), height = Double(CVPixelBufferGetHeight(frame))
        let scale = min(size / height, size / width)
        let newW = (width * scale).rounded(.toNearestOrEven), newH = (height * scale).rounded(.toNearestOrEven)
        let left = ((size - newW) / 2 - 0.1).rounded(.toNearestOrEven)
        let top = ((size - newH) / 2 - 0.1).rounded(.toNearestOrEven)
        let picture = CIImage(cvPixelBuffer: frame)
            .samplingLinear()
            .transformed(by: CGAffineTransform(scaleX: CGFloat(newW / width), y: CGFloat(newH / height)))
            // Bottom-left origin: the picture's bottom edge sits above the bottom padding.
            .transformed(by: CGAffineTransform(translationX: CGFloat(left), y: CGFloat(size - top - newH)))
            .composited(over: gray.cropped(to: CGRect(x: 0, y: 0, width: size, height: size)))
        guard let boxes = run(picture, size: Int(size)) else { return [] }
        return boxes.map { box in
            Detection(x1: (box.x1 - left) / scale, y1: (box.y1 - top) / scale,
                      x2: (box.x2 - left) / scale, y2: (box.y2 - top) / scale, confidence: box.confidence)
        }
    }

    // MARK: Model call and decoding

    /// Boxes in model-input pixels (onnx_detector.decode), or nil if the model call failed.
    private func run(_ picture: CIImage, size: Int) -> [Detection]? {
        guard let entry = models[size], let buffer = inputBuffer(size) else { return nil }
        context.render(picture, to: buffer, bounds: CGRect(x: 0, y: 0, width: size, height: size), colorSpace: nil)
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [entry.input: MLFeatureValue(pixelBuffer: buffer)]),
              let result = try? entry.model.prediction(from: provider),
              let output = result.featureValue(for: entry.output)?.multiArrayValue
        else { return nil }
        return decode(output)
    }

    private func decode(_ output: MLMultiArray) -> [Detection] {
        // Shape 1 x (4 + classes) x anchors.
        guard output.shape.count == 3 else { return [] }
        let anchors = output.shape[2].intValue
        let rowStride = output.strides[1].intValue, anchorStride = output.strides[2].intValue
        let scoreRow = 4 + cfg.ballClassId
        guard scoreRow < output.shape[1].intValue else { return [] }
        let values = readFloats(output)
        func at(_ row: Int, _ anchor: Int) -> Double { Double(values[row * rowStride + anchor * anchorStride]) }

        var order: [(Int, Double)] = []
        for anchor in 0..<anchors {
            let score = at(scoreRow, anchor)
            if score >= cfg.ballConfidence { order.append((anchor, score)) }
        }
        order.sort { $0.1 > $1.1 }
        var kept: [Detection] = []
        for (anchor, score) in order {
            let cx = at(0, anchor), cy = at(1, anchor), w = at(2, anchor), h = at(3, anchor)
            let box = Detection(x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, confidence: score)
            if kept.allSatisfy({ iou(box, $0) <= cfg.ballNmsIou }) {
                kept.append(box)
                if kept.count == cfg.ballCandidates { break }
            }
        }
        return kept
    }

    /// The whole array as Float, whatever its storage type (the export is float16).
    private func readFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float16:
            return array.withUnsafeBufferPointer(ofType: Float16.self) { $0.prefix(count).map(Float.init) }
        case .float32:
            return array.withUnsafeBufferPointer(ofType: Float.self) { Array($0.prefix(count)) }
        case .double:
            return array.withUnsafeBufferPointer(ofType: Double.self) { $0.prefix(count).map(Float.init) }
        default:
            return (0..<count).map { array[$0].floatValue }
        }
    }

    private func inputBuffer(_ size: Int) -> CVPixelBuffer? {
        if let buffer = buffers[size] { return buffer }
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attributes, &buffer)
        buffers[size] = buffer
        return buffer
    }
}

func iou(_ a: Detection, _ b: Detection) -> Double {
    let ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    let iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    let inter = ix * iy
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return union > 0 ? inter / union : 0
}
