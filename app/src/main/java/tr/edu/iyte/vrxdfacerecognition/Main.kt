package tr.edu.iyte.vrxdfacerecognition

import android.content.Context
import android.util.Log
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import tr.edu.iyte.vrxd.api.IPlugin
import tr.edu.iyte.vrxd.api.data.Shape
import java.util.concurrent.Executors

class Main : IPlugin {
    private val pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2)

    private var cascade: CascadeClassifier? = null
    private var frameC = 0
    private val color = Scalar(255.0, 0.0, 0.0)
    private val minSize = Size(0.0, 0.0)
    private val maxSize = Size(300.0, 300.0)

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

    override fun onFrame(frameId: Int, mat: Mat) {
        frameC++

        pool.submit {
            val ms = System.currentTimeMillis()
            val detected = MatOfRect()

            val img = Mat()
            Core.rotate(Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE), img, 1)

            val ms2 = System.currentTimeMillis()

            cascade?.detectMultiScale(img, detected, 1.05, 1, Objdetect.CASCADE_SCALE_IMAGE
                    and Objdetect.CASCADE_DO_CANNY_PRUNING, minSize, maxSize)

            val ms3 = System.currentTimeMillis()

            detected.toArray().forEach {
                Imgproc.rectangle(img, it, color)
            }

            Imgcodecs.imwrite("/sdcard/VRXD/img/$frameC-img.jpg", img)

            val ms4 = System.currentTimeMillis()

            val arr = detected.toArray()
            Log.i(TAG, "elapsed decode and rotate ${ms2 - ms}ms")
            Log.i(TAG, "elapsed detection ${ms3 - ms2}ms")
            Log.i(TAG, "elapsed draw and write ${ms4 - ms3}ms")
            Log.i(TAG, "detected faces: ${arr.joinToString()}")
            Log.i(TAG, "elapsed frame process total ${System.currentTimeMillis() - ms}ms")
        }
    }

    override fun onFrame(frameId: Int, width: Int, height: Int, bytes: ByteArray) {
        TODO("not implemented")
    }
    override fun getFrameObjCount(frameId: Int): Int {
        TODO("not implemented")
    }

    override fun getFrameObj(frameId: Int, objIdx: Int): Shape {
        TODO("not implemented")
    }

    override fun getResources(): List<String> {
        TODO("not implemented")
    }

    companion object {
        const val TAG = "FACE-RECOGNITION"
    }
}