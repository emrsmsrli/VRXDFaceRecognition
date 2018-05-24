package tr.edu.iyte.vrxdfacerecognition

import android.content.Context
import android.os.Environment
import android.util.Log
import com.google.gson.Gson
import org.opencv.core.*
import org.opencv.face.LBPHFaceRecognizer
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import tr.edu.iyte.vrxd.api.IPlugin
import tr.edu.iyte.vrxd.api.data.Rectangle
import tr.edu.iyte.vrxd.api.data.Shape
import tr.edu.iyte.vrxd.api.data.Text
import tr.edu.iyte.vrxdfacerecognition.BuildConfig.DEBUG
import tr.edu.iyte.vrxdfacerecognition.db.Db
import java.io.File
import java.util.*
import java.util.concurrent.Executors

class Main : IPlugin {
    private val pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2)

    private lateinit var debugFolder: File
    private var detector: CascadeClassifier? = null
    private var recognizer: LBPHFaceRecognizer? = null
    private var frameC = 0
    private val color = Scalar(255.0, 0.0, 0.0)
    private val minSize = Size(10.0, 10.0)
    private val maxSize = Size(100.0, 100.0)

    private lateinit var db: Db
    private val predictedLabel = intArrayOf(0)
    private val predictionAccuracy = doubleArrayOf(0.0)

    // int is frame id
    private val frames = hashMapOf<Int, Frame>()

    override fun isOpenCvExclusive() = true

    override fun onStart(ctx: Context) {
        Log.i(TAG, "starting face recog plugin")
        detector = CascadeClassifier(VRXD_LOC.path + "/frontal_face.xml")

        if(DEBUG) {
            debugFolder = File(VRXD_LOC, "debug")
        }

        if(!VRXD_TRAIN_LOC.exists())
            return

        pool.submit {
            val json = Scanner(File(VRXD_TRAIN_LOC, "db.json")).use {
                val builder = StringBuilder(128)
                while(it.hasNextLine())
                    builder.append(it.nextLine())
                builder.toString()
            }

            db = Gson().fromJson(json, Db::class.java) // any exception disables recognition

            // files are like "1 (1).jpg"
            val group = VRXD_TRAIN_LOC.listFiles().groupBy { it.name.split(" ")[0] }
            val labels = mutableListOf<Int>()
            val trainingData = mutableListOf<Mat>()
            for((label, files) in group) {
                var i = 0
                for(file in files) {
                    try {
                        labels.add(Integer.parseInt(label))
                    } catch(e: NumberFormatException) { // defected label, skip samples
                        break
                    }
                    trainingData.add(Imgcodecs.imread(file.path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE))
                    if(DEBUG)
                        Log.i(TAG, "training data loaded label: $label count: ${++i}")
                }
            }

            recognizer = LBPHFaceRecognizer.create()
            recognizer?.train(trainingData, MatOfInt(*labels.toIntArray()))
        }
    }

    override fun onFrame(frameId: Int, mat: Mat) {
        frameC++
        val frame = Frame(frameId, Collections.synchronizedList(mutableListOf()))
        frames[frameId] = frame

        pool.submit {
            val faces = detect(frame, mat)
            if(recognizer != null)
                recognize(frame, faces)
            frame.isReady = true
        }
    }

    override fun onFrame(frameId: Int, width: Int, height: Int, bytes: ByteArray) {
        TODO("not implemented")
    }

    override fun getFrameShapes(frameId: Int): List<Shape> {
        val frame = frames[frameId]
        if(frame == null || !frame.isReady)
            return emptyList()

        return ArrayList(frames[frameId]!!.shapes)
    }

    private fun detect(frame: Frame, mat: Mat): MutableList<Mat> {
        val detected = MatOfRect()
        val img = Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)

        detector?.detectMultiScale(img, detected, 1.08, 1, Objdetect.CASCADE_SCALE_IMAGE
                and Objdetect.CASCADE_DO_CANNY_PRUNING, minSize, maxSize)

        val faces = mutableListOf<Mat>()
        detected.toArray().forEach {
            faces.add(mat.submat(it))
            if(DEBUG)
                Imgproc.rectangle(img, it, color)
            frame.shapes.add(Rectangle(it.x, it.y, it.width, it.height, 0.0, 0L))
        }

        if(DEBUG) {
            if(!debugFolder.exists())
                debugFolder.mkdir()
            Imgcodecs.imwrite(debugFolder.path + "/$frameC-img.jpg", img)
            Log.d(TAG, "detection complete for frame: ${frame.id}, found faces: ${faces.size}")
        }

        return faces
    }

    private fun recognize(frame: Frame, faces: MutableList<Mat>) {
        if(faces.isEmpty())
            return

        val texts = mutableListOf<Shape>()
        synchronized(frame.shapes) {
            val it = frame.shapes.iterator()
            val fit = faces.iterator()

            fun rm() {
                it.remove()
                fit.remove()
            }

            while(it.hasNext()) {
                val shape = it.next()
                val face = fit.next()

                recognizer?.predict(face, predictedLabel, predictionAccuracy)
                Log.d(TAG, "prediction ${predictedLabel[0]} ${predictionAccuracy[0]}")
                if(predictionAccuracy[0] > 50) {
                    rm()
                    continue
                }

                val person = db.find { it.personId == predictedLabel[0] }
                if(person == null) {
                    rm()
                    continue
                }

                texts.add(Text(shape.x, shape.y, 0, 0.0, person.describe() ,0L))
            }
        }
        frame.shapes.addAll(texts)
        frame.isReady = true
    }

    companion object {
        const val TAG = "FACE-RECOGNITION"
        val VRXD_LOC = File(Environment.getExternalStorageDirectory(), "VRXD")
        val VRXD_TRAIN_LOC = File(VRXD_LOC, "face-recognition-db")
    }
}