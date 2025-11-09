import mysql.connector

class MemoryDB:
    def __init__(self, host="localhost", user="allie", password="StrongPassword123!", database="allie_memory"):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()

    def add_fact(self, keyword, fact, source="user"):
        sql = "INSERT INTO facts (keyword, fact, source) VALUES (%s, %s, %s)"
        self.cursor.execute(sql, (keyword, fact, source))
        self.conn.commit()

    def get_fact(self, keyword):
        sql = "SELECT fact FROM facts WHERE keyword = %s ORDER BY updated_at DESC LIMIT 1"
        self.cursor.execute(sql, (keyword,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def timeline(self):
        sql = "SELECT keyword, fact, updated_at FROM facts ORDER BY updated_at ASC"
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def update_fact(self, keyword, new_fact, source="external"):
        sql = "UPDATE facts SET fact = %s, source = %s WHERE keyword = %s"
        self.cursor.execute(sql, (new_fact, source, keyword))
        self.conn.commit()

    def delete_fact(self, keyword):
        sql = "DELETE FROM facts WHERE keyword = %s"
        self.cursor.execute(sql, (keyword,))
        self.conn.commit()
