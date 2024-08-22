import logging
import os
import time
import numpy as np
from pydub import AudioSegment
import csv
import json
from datetime import datetime
import statistics
import sqlite3

def is_file_ready(file_path, min_size=1024, check_interval=1, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) >= min_size:
            return True
        time.sleep(check_interval)
    return False

def is_audio_non_empty(file_path, segment_duration=30000, speech_threshold=-40):
    start_time = time.time()
    audio = AudioSegment.from_file(file_path)
    
    segments = [audio[i:i+segment_duration] for i in range(0, len(audio), segment_duration)]
    
    max_db = -np.inf
    for i, segment in enumerate(segments):
        samples = np.array(segment.get_array_of_samples())
        rms = np.sqrt(np.mean(samples**2))
        db = 20 * np.log10(rms) if rms > 0 else -np.inf
        max_db = max(max_db, db)
        
        if db > speech_threshold:
            end_time = time.time()
            check_duration = end_time - start_time
            logging.debug(f"Audio check for {os.path.basename(file_path)}: Non-empty (took {check_duration:.2f}s, max dB: {max_db:.2f}, triggered at segment {i+1}/{len(segments)})")
            return True
    
    end_time = time.time()
    check_duration = end_time - start_time
    logging.debug(f"Audio check for {os.path.basename(file_path)}: Empty (took {check_duration:.2f}s, max dB: {max_db:.2f})")
    return False

def log_stats_to_db(db_path, file_path, duration, transcription_time, speed):
    """Log transcription statistics to the database."""
    with sqlite3.connect(db_path) as db_conn:
        c = db_conn.cursor()
        c.execute('''INSERT INTO transcription_stats
                     (timestamp, file_path, duration, transcription_time, speed)
                     VALUES (?, ?, ?, ?, ?)''',
                  (datetime.now(), file_path, duration, transcription_time, speed))
        db_conn.commit()

def append_to_csv(file_path, transcription):
    csv_file = 'transcriptions.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Extract timestamp from filename
    filename = os.path.basename(file_path)
    timestamp = datetime.strptime(filename.split('.')[0], "%Y-%m-%d_%H-%M-%S")
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'text', 'provider_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'text': transcription['text'],
            'provider_output': json.dumps(transcription)
        })

def log_performance_stats(processing_times):
    if not processing_times:
        return
    
    avg_time = statistics.mean(processing_times)
    med_time = statistics.median(processing_times)
    total_time = sum(processing_times)
    num_files = len(processing_times)
    
    logging.info(f"Stats: Files: {num_files} | Total Time: {total_time:.2f}s | Avg Time: {avg_time:.2f}s | Median Time: {med_time:.2f}s")

def get_stats_from_db(db_path):
    """Get statistics from the database."""
    with sqlite3.connect(db_path) as db_conn:
        c = db_conn.cursor()
        c.execute('''SELECT COUNT(*) as count,
                            SUM(transcription_time) as total_time,
                            AVG(transcription_time) as avg_time,
                            MIN(transcription_time) as min_time
                     FROM transcription_stats''')
        row = c.fetchone()
        
        c.execute('SELECT transcription_time FROM transcription_stats ORDER BY transcription_time')
        all_times = c.fetchall()
        
    files_count, total_time, avg_time, min_time = row
    
    if files_count:
        median_time = all_times[len(all_times) // 2][0] if len(all_times) % 2 != 0 else \
                      (all_times[len(all_times) // 2 - 1][0] + all_times[len(all_times) // 2][0]) / 2
    else:
        median_time = 0

    return files_count, total_time or 0, avg_time or 0, median_time, min_time or 0