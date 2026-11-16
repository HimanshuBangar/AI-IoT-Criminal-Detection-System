import sqlite3
from config import DATABASE_PATH


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Registered reference persons
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Detection event history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            description TEXT,
            image_path TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def add_person(name, image_path):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO persons (name, image_path) VALUES (?, ?)",
        (name, str(image_path))
    )

    connection.commit()
    connection.close()


def get_all_persons():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT id, name, image_path, created_at FROM persons ORDER BY id DESC"
    )

    records = cursor.fetchall()
    connection.close()

    return records


def add_detection(event_type, description="", image_path=None,
                  confidence=None):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO detections
        (event_type, description, image_path, confidence)
        VALUES (?, ?, ?, ?)
    """, (event_type, description, image_path, confidence))

    connection.commit()
    connection.close()


def get_all_detections():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, event_type, description, image_path,
               confidence, created_at
        FROM detections
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return records
