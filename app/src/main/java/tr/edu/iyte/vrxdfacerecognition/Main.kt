package tr.edu.iyte.vrxdfacerecognition

import android.content.Context
import android.os.Environment
import android.util.Log
///import org.bytedeco.javacpp.DoublePointer
///import org.bytedeco.javacpp.IntPointer
///import org.bytedeco.javacpp.opencv_core
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import tr.edu.iyte.vrxd.api.IPlugin
import tr.edu.iyte.vrxd.api.data.Rectangle
import tr.edu.iyte.vrxd.api.data.Shape
//import java.io.File
//import java.nio.IntBuffer
import java.util.concurrent.Executors

//import org.bytedeco.javacpp.opencv_face
//import org.bytedeco.javacpp.opencv_core.CV_32SC1
//import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE
//import org.bytedeco.javacpp.opencv_imgcodecs.imread
//import org.opencv.face.LBPHFaceRecognizer

class Main : IPlugin {
    private val pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2)
    //private val faceRecognizer = opencv_face.LBPHFaceRecognizer.create()

    private var cascade: CascadeClassifier? = null
    private var frameC = 0
    private val color = Scalar(255.0, 0.0, 0.0)
    private val minSize = Size(10.0, 10.0)
    private val maxSize = Size(100.0, 100.0)

    // int is frame id
    private val frames = hashMapOf<Int, Frame>()

    override fun isOpenCvExclusive() = true

    override fun onStart(ctx: Context) {
        Log.i(TAG, "starting face recog plugin")
        cascade = CascadeClassifier(Environment.getExternalStorageDirectory().path + "/VRXD/a.xml")
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
        frames[frameId] = Frame(frameId, mutableListOf())

        pool.submit {
            val ms = System.currentTimeMillis()
            val detected = MatOfRect()

            val img = Mat()
            Core.rotate(Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE), img, 1)

            val ms2 = System.currentTimeMillis()

            cascade?.detectMultiScale(img, detected, 1.08, 1, Objdetect.CASCADE_SCALE_IMAGE
                    and Objdetect.CASCADE_DO_CANNY_PRUNING, minSize, maxSize)

            val ms3 = System.currentTimeMillis()

            synchronized(frames) {
                val frame = frames[frameId]!!
                detected.toArray().forEach {
                    Imgproc.rectangle(img, it, color)
                    frame.shapes.add(Rectangle(it.x, it.y, it.width, it.height, 0.0, 0L))
                }
                frame.isReady = true
            }

            Imgcodecs.imwrite(Environment.getExternalStorageDirectory().path + "/VRXD/img/$frameC-img.jpg", img)

            val ms4 = System.currentTimeMillis()

            val arr = detected.toArray()
            Log.i(TAG, "elapsed decode and rotate ${ms2 - ms}ms")
            Log.i(TAG, "elapsed detection ${ms3 - ms2}ms")
            Log.i(TAG, "elapsed draw and write ${ms4 - ms3}ms")
            Log.i(TAG, "detected faces: ${arr.joinToString()}")
            Log.i(TAG, "elapsed frame process total ${System.currentTimeMillis() - ms}ms")

            /*val trainData = (0..4).map { val m = Imgcodecs.imread("/sdcard/VRXD/train/$it.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)
                Log.i(TAG, "$it okay ${m.rows()} ${m.cols()}")
                m
            }

            try {
                val recognizer = LBPHFaceRecognizer.create()
                recognizer.train(trainData.toMutableList(), MatOfInt(1, 1, 1, 1, 1))
                var label = intArrayOf(0)
                var conf = doubleArrayOf(0.0)
                recognizer.predict(Imgcodecs.imread("/sdcard/VRXD/train/5.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE), label, conf)
                Log.i(TAG, "predicted label1: ${label.joinToString()} ${conf.joinToString()}")
                label = intArrayOf(0)
                conf = doubleArrayOf(0.0)
                recognizer.predict(Imgcodecs.imread("/sdcard/VRXD/train/6.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE), label, conf)
                Log.i(TAG, "predicted label2: ${label.joinToString()} ${conf.joinToString()}")
                //Log.i(TAG, "predicted label1: ${recognizer.predict_label(Imgcodecs.imread("/sdcard/VRXD/train/5.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE))}")
                //Log.i(TAG, "predicted label2: ${recognizer.predict_label(Imgcodecs.imread("/sdcard/VRXD/train/6.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE))}")
            } catch(e: Exception) {e.printStackTrace()}*/
        }
    }

    override fun onFrame(frameId: Int, width: Int, height: Int, bytes: ByteArray) {
        TODO("not implemented")
    }

    override fun getFrameShapes(frameId: Int): List<Shape> {
        if(frames[frameId] == null)
            return listOf()

        synchronized(frames) {
            val s = ArrayList(frames[frameId]!!.shapes)
            frames.remove(frameId)
            return s
        }
    }

    override fun getResources(): List<String> {
        TODO("not implemented")
    }

    /*private fun train(trainingDir: String) {
        val trainingFolder = File(trainingDir)

        val imageFiles = trainingFolder.listFiles{ dir, name ->
            name.toLowerCase().endsWith(".jpg") || name.endsWith(".png")
        }

        val images: opencv_core.MatVector
        if (imageFiles != null) {
            images = opencv_core.MatVector(imageFiles.size.toLong())
        } else {
            println("No image found in root folder!")
            return
        }
        val labels = opencv_core.Mat(imageFiles.size, 1, CV_32SC1)
        val labelsBuffer = labels.createBuffer<IntBuffer>()
        for ((counter, image) in imageFiles.withIndex()) {
            val img = imread(image.absolutePath, CV_LOAD_IMAGE_GRAYSCALE)
            val label = Integer.parseInt(image.name.split("-".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[0])
            images.put(counter.toLong(), img)
            labelsBuffer.put(label)
        }
        faceRecognizer.train(images, labels)
    }

    private fun recognize(testImage: opencv_core.Mat): Int {
        val label = IntPointer(1)
        val confidence = DoublePointer(1)
        faceRecognizer.predict(testImage, label, confidence)
        val predictedLabel = label.get(0)
        val pConfidence = confidence.get(0)
        return predictedLabel
    }*/

    companion object {
        const val TAG = "FACE-RECOGNITION"
    }
}