package com.ricedisease.classifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.ricedisease.classifier.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "RiceClassifier"
    }

    private lateinit var binding: ActivityMainBinding
    private var classifier: RiceClassifier? = null
    private var currentPhotoUri: Uri? = null
    private var isModelLoaded = false

    // Camera permission launcher
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    // Camera capture launcher
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            currentPhotoUri?.let { uri ->
                processImage(uri)
            }
        }
    }

    // Gallery picker launcher
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { processImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupButtons()
        initializeClassifier()
    }

    private fun initializeClassifier() {
        binding.resultText.text = "Loading model..."
        binding.btnCamera.isEnabled = false
        binding.btnGallery.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                classifier = RiceClassifier(this@MainActivity)
                isModelLoaded = true
                Log.d(TAG, "Model loaded successfully")

                withContext(Dispatchers.Main) {
                    binding.resultText.text = getString(R.string.result_placeholder)
                    binding.btnCamera.isEnabled = true
                    binding.btnGallery.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                withContext(Dispatchers.Main) {
                    binding.resultText.text = "Failed to load model: ${e.message}"
                    binding.resultText.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.disease))
                    Toast.makeText(this@MainActivity, "Model loading failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setupButtons() {
        binding.btnCamera.setOnClickListener {
            if (!isModelLoaded) {
                Toast.makeText(this, "Model not loaded yet", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                launchCamera()
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        binding.btnGallery.setOnClickListener {
            if (!isModelLoaded) {
                Toast.makeText(this, "Model not loaded yet", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            pickImageLauncher.launch("image/*")
        }
    }

    private fun launchCamera() {
        val photoFile = File(cacheDir, "photo_${System.currentTimeMillis()}.jpg")
        currentPhotoUri = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            photoFile
        )
        takePictureLauncher.launch(currentPhotoUri)
    }

    private fun processImage(uri: Uri) {
        val currentClassifier = classifier
        if (currentClassifier == null) {
            showError("Classifier not initialized")
            return
        }

        showLoading(true)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Load and prepare bitmap
                val bitmap = loadBitmapFromUri(uri)

                if (bitmap != null) {
                    // Run classification
                    val result = currentClassifier.classify(bitmap)

                    withContext(Dispatchers.Main) {
                        showLoading(false)
                        displayResult(bitmap, result)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        showLoading(false)
                        showError("Failed to load image")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Classification error", e)
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    showError("Error: ${e.message}")
                }
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (bitmap == null) {
                Log.e(TAG, "Failed to decode bitmap from uri: $uri")
                return null
            }

            // Handle rotation from EXIF data
            handleImageRotation(uri, bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading bitmap", e)
            null
        }
    }

    private fun handleImageRotation(uri: Uri, bitmap: Bitmap): Bitmap {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val exif = inputStream?.let { ExifInterface(it) }
            inputStream?.close()

            val orientation = exif?.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            ) ?: ExifInterface.ORIENTATION_UNDEFINED

            val rotationDegrees = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }

            if (rotationDegrees != 0f) {
                val matrix = Matrix().apply { postRotate(rotationDegrees) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling image rotation", e)
            bitmap
        }
    }

    private fun displayResult(bitmap: Bitmap, result: RiceClassifier.ClassificationResult) {
        // Display image
        binding.imageView.setImageBitmap(bitmap)

        // Display classification result
        binding.resultText.text = result.className

        // Set color based on health status
        val colorRes = if (classifier?.isHealthy(result) == true) {
            R.color.healthy
        } else {
            R.color.disease
        }
        binding.resultText.setTextColor(ContextCompat.getColor(this, colorRes))

        // Display confidence
        val confidencePercent = (result.confidence * 100).toInt()
        binding.confidenceText.text = "$confidencePercent%"

        // Display inference time
        binding.inferenceTimeText.text = "${result.inferenceTimeMs} ms"
    }

    private fun showLoading(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnCamera.isEnabled = !show
        binding.btnGallery.isEnabled = !show

        if (show) {
            binding.resultText.text = getString(R.string.loading)
            binding.resultText.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
            binding.confidenceText.text = "--"
            binding.inferenceTimeText.text = "--"
        }
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        binding.resultText.text = "Error occurred"
        binding.resultText.setTextColor(ContextCompat.getColor(this, R.color.disease))
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
