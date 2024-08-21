import sqlite3
import logging
import os
from src.constants import SUPPORTED_EXTENSIONS

def init_db(db_path):
    """Initialize the database and create necessary tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create transcription_stats table
    c.execute('''CREATE TABLE IF NOT EXISTS transcription_stats
                 (timestamp TEXT, file_path TEXT, duration REAL, transcription_time REAL, speed REAL)''')
    
    # Create processed_files table
    c.execute('''CREATE TABLE IF NOT EXISTS processed_files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_path TEXT UNIQUE,
                  transcription TEXT,
                  status TEXT DEFAULT 'processed',
                  recorded_at TIMESTAMP,
                  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    return conn

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

def get_unprocessed_files(db_connection, recordings_path):
    """Retrieve unprocessed files from the recordings directory."""
    cursor = db_connection.cursor()
    
    try:
        cursor.execute("SELECT file_path FROM processed_files WHERE status = 'processed'")
        processed_files = set(row[0] for row in cursor.fetchall())
    except sqlite3.OperationalError:
        # Table doesn't exist, so no files have been processed yet
        processed_files = set()
    
    all_files = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) 
                 if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    unprocessed_files = [f for f in all_files if f not in processed_files]
    
    return unprocessed_files

def get_total_files(db_connection):
    """Get the total number of files in the processed_files table."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM processed_files")
    return cursor.fetchone()[0]