package tr.edu.iyte.vrxdfacerecognition.db

class PersonInfo(val name: String,
                 val surname: String,
                 val email: String) {
    override fun toString(): String {
        return "Name: $name\nSurname: $surname\nEmail: $email"
    }
}