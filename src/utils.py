import logging
import signal
import time
import os
import numpy as np
from pydub import AudioSegment

def signal_handler(signum, frame):
    global shutdown_requested
    print("Shutdown requested. Cleaning up...")
    shutdown_requested = True
    if processing_task:
        processing_task.cancel()

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

def log_stats_to_db(db_conn, file_path, duration, transcription_time, speed):
    c = db_conn.cursor()
    c.execute("INSERT INTO transcription_stats VALUES (?, ?, ?, ?, ?)",
              (time.strftime('%Y-%m-%d %H:%M:%S'), file_path, duration, transcription_time, speed))
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

def log_performance_stats():
    if not processing_times:
        return
    
    avg_time = statistics.mean(processing_times)
    med_time = statistics.median(processing_times)
    total_time = sum(processing_times)
    num_files = len(processing_times)
    
    logging.info(f"Stats: Files: {num_files} | Total Time: {total_time:.2f}s | Avg Time: {avg_time:.2f}s | Median Time: {med_time:.2f}s")