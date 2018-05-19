package tr.edu.iyte.vrxdfacerecognition

import tr.edu.iyte.vrxd.api.data.Shape

data class Frame(val id: Int, val shapes: MutableList<Shape>, var isReady: Boolean = false)