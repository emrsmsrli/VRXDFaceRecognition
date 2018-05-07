package tr.edu.iyte.vrxdfacerecognition

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfRect
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import tr.edu.iyte.vrxd.api.IPlugin

class Main : IPlugin {
    private var cascade: CascadeClassifier? = null
    private var frameC = 0

    override fun isOpenCvExclusive() = true

    override fun onStart(ctx: Context) {
        Log.i(TAG, "starting face recog plugin")
        cascade = CascadeClassifier("/sdcard/VRXD/a.xml")
    }

    override fun onResume(ctx: Context) {
        TODO("not implemented")
    }

    override fun onPause(ctx: Context) {
        TODO("not implemented")
    }

    override fun onStop(ctx: Context) {
        TODO("not implemented")
    }

    override fun onFrame(mat: Mat) {
        frameC++
        val ms = System.currentTimeMillis()
        val detected = MatOfRect()

        val decoded = Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)
        cascade?.detectMultiScale(decoded, detected)
        for(r in detected.toArray()) {
            Imgproc.rectangle(decoded, r, Scalar(255.0, 0.0, 0.0))
        }

        Imgcodecs.imwrite("/sdcard/VRXD/img/$frameC-img.jpg", decoded)

        val arr = detected.toArray()
        Log.i(TAG, "detected faces: ${arr.joinToString()}")
        Log.i(TAG, "elapsed internal ${System.currentTimeMillis() - ms}ms")
    }

    override fun onFrame(width: Int, height: Int, bytes: ByteArray) {
        TODO("not implemented")
    }

    override fun getResources(): List<String> {
        TODO("not implemented")
    }

    companion object {
        const val TAG = "FACE-RECOGNITION"
    }
}