package tr.edu.iyte.vrxdfacerecognition.db

class Db(val people: Array<Person>) {
    inline fun find(predicate: (Person) -> Boolean): Person? {
        return people.firstOrNull(predicate)
    }
}