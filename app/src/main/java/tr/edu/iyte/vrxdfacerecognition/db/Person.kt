package tr.edu.iyte.vrxdfacerecognition.db

class Person(val personId: Int,
             val info: PersonInfo) {
    fun describe(): String {
        return info.toString()
    }
}