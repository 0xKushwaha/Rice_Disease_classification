package com.ricedisease.classifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Rice Disease Classifier using ONNX Runtime
 * Model: MambaCNN FP16 (98.44% accuracy)
 * Input: 128x128 RGB image with ImageNet normalization
 * Output: 6 disease classes
 * Supported formats: JPG, PNG, BMP, WebP
 */
class RiceClassifier(context: Context) {

    companion object {
        private const val MODEL_FILE = "mamba_fp16.onnx"
        private const val INPUT_SIZE = 128
        private const val NUM_CLASSES = 6

        // ImageNet normalization constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Class labels for rice diseases
        val CLASS_NAMES = arrayOf(
            "Bacterial Leaf Blight",
            "Brown Spot",
            "Healthy Rice Leaf",
            "Leaf Blast",
            "Leaf Scald",
            "Sheath Blight"
        )
    }

    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    init {
        // Load ONNX model from assets
        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        ortSession = ortEnvironment.createSession(modelBytes)
    }

    data class ClassificationResult(
        val className: String,
        val confidence: Float,
        val inferenceTimeMs: Long,
        val allProbabilities: FloatArray
    )

    /**
     * Run inference on a bitmap image
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Preprocess: convert to normalized float tensor [1, 3, 128, 128]
        val inputTensor = preprocessImage(resizedBitmap)

        // Measure inference time
        val startTime = System.currentTimeMillis()

        // Run inference
        val inputName = ortSession.inputNames.iterator().next()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val onnxTensor = OnnxTensor.createTensor(ortEnvironment, inputTensor, shape)

        val output = ortSession.run(mapOf(inputName to onnxTensor))

        val inferenceTime = System.currentTimeMillis() - startTime

        // Get output logits
        val outputTensor = output[0].value as Array<FloatArray>
        val logits = outputTensor[0]

        // Apply softmax to get probabilities
        val probabilities = softmax(logits)

        // Find the class with highest probability
        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }

        // Clean up
        onnxTensor.close()
        output.close()

        return ClassificationResult(
            className = CLASS_NAMES[maxIdx],
            confidence = maxProb,
            inferenceTimeMs = inferenceTime,
            allProbabilities = probabilities
        )
    }

    /**
     * Preprocess image: resize, normalize with ImageNet stats, convert to CHW format
     */
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Create float buffer for [1, 3, H, W] tensor (CHW format)
        val buffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

        // Process each channel separately (CHW format expected by PyTorch models)
        // Red channel
        for (i in pixels.indices) {
            val r = ((pixels[i] shr 16) and 0xFF) / 255.0f
            buffer.put((r - MEAN[0]) / STD[0])
        }
        // Green channel
        for (i in pixels.indices) {
            val g = ((pixels[i] shr 8) and 0xFF) / 255.0f
            buffer.put((g - MEAN[1]) / STD[1])
        }
        // Blue channel
        for (i in pixels.indices) {
            val b = (pixels[i] and 0xFF) / 255.0f
            buffer.put((b - MEAN[2]) / STD[2])
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Apply softmax to convert logits to probabilities
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }

    /**
     * Check if the prediction indicates a healthy leaf
     */
    fun isHealthy(result: ClassificationResult): Boolean {
        return result.className == "Healthy Rice Leaf"
    }

    /**
     * Release resources
     */
    fun close() {
        ortSession.close()
        ortEnvironment.close()
    }
}
