import sqlite3
import logging
import os
from src.constants import SUPPORTED_EXTENSIONS

def init_db(db_path):
    """Initialize the database and create necessary tables if they don't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create processed_files table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                transcription TEXT,
                text TEXT,
                status TEXT DEFAULT 'processed',
                recorded_at TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create transcription_stats table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_stats (
                timestamp TEXT,
                file_path TEXT,
                duration REAL,
                transcription_time REAL,
                speed REAL
            )
        ''')
        
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        return None

def setup_database():
    """Set up the main database for processed files."""
    db_connection = sqlite3.connect('processed_files.db')
    cursor = db_connection.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='processed_files'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        cursor.execute('''
            CREATE TABLE processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                transcription TEXT,
                text TEXT,
                status TEXT DEFAULT 'processed',
                recorded_at TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        # Check if the recorded_at column exists
        cursor.execute("PRAGMA table_info(processed_files)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'recorded_at' not in columns:
            # Add the recorded_at column if it doesn't exist
            cursor.execute("ALTER TABLE processed_files ADD COLUMN recorded_at TIMESTAMP")
    
    db_connection.commit()
    return db_connection

def clean_stats(db_path):
    """Clean the transcription stats database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM transcription_stats")
    conn.commit()
    conn.close()
    logging.info("Transcription stats have been cleaned.")

def get_unprocessed_files(db_path, recordings_path):
    """Get a list of unprocessed files."""
    init_db(db_path)  # Ensure the database and tables exist
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM processed_files WHERE status = 'processed'")
        processed_files = set(row[0] for row in cursor.fetchall())
    
    all_files = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) 
                 if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    unprocessed_files = [f for f in all_files if f not in processed_files]
    
    return unprocessed_files

def get_total_files(db_path):
    """Get the total number of files in the processed_files table."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_files")
        return cursor.fetchone()[0]

def get_previous_transcription(db_connection, limit=1):
    """Retrieve the most recent transcription(s) from the database."""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT text
        FROM processed_files
        WHERE status = 'processed'
        ORDER BY processed_at DESC
        LIMIT ?
    """, (limit,))
    results = cursor.fetchall()
    return [result[0] for result in results] if results else []

def get_previous_transcription_thread_safe(db_path, limit=1):
    """Retrieve the most recent transcription(s) from the database in a thread-safe manner."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT text
            FROM processed_files
            WHERE status = 'processed'
            ORDER BY processed_at DESC
            LIMIT ?
        """, (limit,))
        results = cursor.fetchall()
    return [result[0] for result in results] if results else []

def migrate_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if the 'text' column exists
    c.execute("PRAGMA table_info(processed_files)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'text' not in columns:
        # Add the 'text' column if it doesn't exist
        c.execute("ALTER TABLE processed_files ADD COLUMN text TEXT")
        
        # Update existing rows to populate the 'text' field
        c.execute("SELECT id, transcription FROM processed_files")
        for row in c.fetchall():
            id, transcription_json = row
            transcription = json.loads(transcription_json)
            full_text = transcription.get('text', '')
            c.execute("UPDATE processed_files SET text = ? WHERE id = ?", (full_text, id))
    
    conn.commit()
    conn.close()