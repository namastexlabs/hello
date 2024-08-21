import asyncio
import json
import argparse
import os
import sqlite3
import csv
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from datetime import datetime
import time
from fastapi import FastAPI, Query
from typing import Optional
import uvicorn
from providers.groq_whisper_provider import create_groq_whisper_provider
from providers.faster_whisper_provider import create_faster_whisper_provider
import signal
import sys
import statistics  # Make sure this import is at the top of your file
from rich.console import Console
from rich.text import Text
from pydub import AudioSegment
import numpy as np

console = Console()

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load API keys from environment variables
api_keys = [key.strip() for key in os.getenv("GROQ_API_KEYS", "").split(",") if key.strip()]
if not api_keys:
    raise ValueError("No API keys found. Please set GROQ_API_KEYS in your .env file.")

# Global variables to track queue size and processed files
global_queue_size = 0
total_processed_files = 0

# Global flag to indicate shutdown
shutdown_requested = False
processing_task = None

# Global variables for performance tracking
processing_times = []

def signal_handler(signum, frame):
    global shutdown_requested
    print("Shutdown requested. Cleaning up...")
    shutdown_requested = True
    if processing_task:
        processing_task.cancel()

SUPPORTED_EXTENSIONS = ('.mp4', '.mp3')

def create_provider(provider_name: str, **kwargs):
    if provider_name == "groq":
        return create_groq_whisper_provider(kwargs.get("api_keys", []))
    elif provider_name == "faster_whisper":
        return create_faster_whisper_provider(
            model_size=kwargs.get("model_size", "large-v3"),
            device=kwargs.get("device", "cuda"),
            compute_type=kwargs.get("compute_type", "float16")
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

class FileHandler(FileSystemEventHandler):
    def __init__(self, queue, db_connection):
        self.queue = queue
        self.db_connection = db_connection
        self.processed_files = set()

    def process_file(self, file_path):
        if file_path not in self.processed_files and file_path.lower().endswith(SUPPORTED_EXTENSIONS):
            logging.info(f"New file detected: {file_path}")
            self.queue.put_nowait(file_path)
            self.processed_files.add(file_path)

    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

async def process_files(queue, db_connection, stats_db_conn, language, provider, **provider_kwargs):
    global global_queue_size
    global total_processed_files
    global shutdown_requested
    while not shutdown_requested:
        try:
            file_path = await asyncio.wait_for(queue.get(), timeout=1.0)
            global_queue_size = queue.qsize()
            logging.debug(f"File ready for processing: {file_path}")
            
            # Process the file
            await asyncio.shield(process_single_file(file_path, db_connection, stats_db_conn, language, provider, **provider_kwargs))

            logging.debug(f"Finished processing: {file_path}")
            total_processed_files += 1
            queue.task_done()
            global_queue_size = queue.qsize()

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logging.info("Processing task cancelled")
            break
        except Exception as e:
            logging.error(f"Unexpected error in process_files: {str(e)}")
        
        if shutdown_requested:
            break

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

def get_stats_from_db(db_conn):
    c = db_conn.cursor()
    c.execute("SELECT COUNT(*), SUM(transcription_time), AVG(transcription_time), MIN(transcription_time) FROM transcription_stats")
    result = c.fetchone()
    count, total_time, avg_time, min_time = result if result else (0, 0, 0, 0)
    
    if count:
        c.execute("SELECT transcription_time FROM transcription_stats ORDER BY transcription_time LIMIT 2 - (? % 2) OFFSET (?-1)/2", (count, count))
        median_result = c.fetchall()
        median_time = median_result[0][0] if count % 2 != 0 else sum(row[0] for row in median_result) / 2
    else:
        median_time = 0
    
    return count, total_time or 0, avg_time or 0, median_time, min_time or 0

async def process_single_file(file_path, db_connection, stats_db_conn, language, provider, **provider_kwargs):
    global global_queue_size
    
    # Wait for the file to be completely written
    while not is_file_ready(file_path):
        await asyncio.sleep(1)
        if shutdown_requested:
            return
    
    logging.debug(f"File ready for processing: {file_path}")
    
    if not is_audio_non_empty(file_path):
        logging.info(f"Skipping file due to no active speech: {file_path}")
        cursor = db_connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO processed_files (file_path, transcription, status, recorded_at) VALUES (?, ?, ?, ?)",
            (file_path, json.dumps({"text": "No active speech"}), "empty", datetime.now())
        )
        db_connection.commit()
        return

    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM processed_files WHERE file_path = ?", (file_path,))
    if cursor.fetchone():
        logging.debug(f"File already processed: {file_path}")
        return

    start_time = time.time()
    try:
        logging.debug(f"Starting transcription for {file_path}")
        transcription = await asyncio.to_thread(provider.transcribe, file_path, language=language, **provider_kwargs)
        transcription_time = time.time() - start_time
        logging.debug(f"Transcription completed for {file_path}. Time taken: {transcription_time:.2f}s")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        audio_duration = transcription['segments'][-1]['end'] if transcription['segments'] else 0
        speed = audio_duration / processing_time
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        log_stats_to_db(stats_db_conn, file_path, audio_duration, processing_time, speed)
        
        # Get updated stats from the database
        files_count, total_time, avg_time, median_time, min_time = get_stats_from_db(stats_db_conn)
        
        # Format the audio duration in mm:ss
        audio_duration_formatted = f"{int(audio_duration // 60)}:{int(audio_duration % 60):02}"
        
        status = Text()
        status.append("🎵 ", style="cyan")
        status.append(f"Queue: {global_queue_size} | ", style="blue")
        status.append("🎬 ", style="yellow")
        status.append(f"{os.path.basename(file_path)} ({file_size:.2f}MB) | ", style="yellow")
        status.append("⏳ ", style="green")
        status.append(f"{audio_duration_formatted} | ", style="green")
        status.append("⚡ ", style="magenta")
        status.append(f"{processing_time:.2f}s | ", style="magenta")
        status.append("📊 ", style="cyan")
        status.append(f"Avg: {avg_time:.2f}s | ", style="cyan")
        status.append("⚡ ", style="red")
        status.append(f"{speed:.2f}x | ", style="red")
        status.append("📈 ", style="blue")
        status.append(f"Total: {total_time:.2f}s | Med: {median_time:.2f}s | Files: {files_count}", style="blue")
        
        console.print(status)
        
        cursor = db_connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO processed_files (file_path, transcription, status, recorded_at) VALUES (?, ?, ?, ?)",
            (file_path, json.dumps(transcription), "processed", datetime.now())
        )
        db_connection.commit()

        append_to_csv(file_path, transcription)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error processing file {os.path.basename(file_path)}: {str(e)}[/bold red]")

def is_file_ready(file_path, min_size=1024, check_interval=1, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) >= min_size:
            return True
        time.sleep(check_interval)
    return False

def log_performance_stats():
    if not processing_times:
        return
    
    avg_time = statistics.mean(processing_times)
    med_time = statistics.median(processing_times)
    total_time = sum(processing_times)
    num_files = len(processing_times)
    
    logging.info(f"Stats: Files: {num_files} | Total Time: {total_time:.2f}s | Avg Time: {avg_time:.2f}s | Median Time: {med_time:.2f}s")

async def start_file_processing(language, provider, db_connection, stats_db_conn, **provider_kwargs):
    global global_queue_size
    global shutdown_requested
    global processing_task
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    queue = asyncio.Queue()
    
    # Set up SQLite database
    db_connection = setup_database()

    # Process existing files
    recordings_path = os.getenv("RECORDINGS_PATH", r"./recordings")
    
    # Create folder if it doesn't exist
    if not os.path.exists(recordings_path):
        os.makedirs(recordings_path)
    
    unprocessed_files = get_unprocessed_files(db_connection, recordings_path)
    for file_path in unprocessed_files:
        queue.put_nowait(file_path)
    
    global_queue_size = queue.qsize()

    # Display initial stats
    logging.info(f"Initial stats:")
    logging.info(f"  Files queued for processing: {len(unprocessed_files)}")
    logging.info(f"  Total files in database: {get_total_files(db_connection)}")

    # Set up file system observer
    event_handler = FileHandler(queue, db_connection)
    observer = Observer()
    observer.schedule(event_handler, path=recordings_path, recursive=False)
    observer.start()

    logging.info(f"Watchdog started. Monitoring folder: {recordings_path}")

    # Start the file processing task
    processing_task = asyncio.create_task(process_files(queue, db_connection, stats_db_conn, language, provider, **provider_kwargs))

    # Start the FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

    try:
        while not shutdown_requested:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        shutdown_requested = True
        observer.stop()
        await processing_task
        observer.join()
        db_connection.close()
        
        # Log final performance stats
        log_performance_stats()

def get_unprocessed_files(db_connection, recordings_path):
    cursor = db_connection.cursor()
    cursor.execute("SELECT file_path FROM processed_files WHERE status = 'processed'")
    processed_files = set(row[0] for row in cursor.fetchall())
    
    all_files = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) 
                 if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    unprocessed_files = [f for f in all_files if f not in processed_files]
    
    return unprocessed_files

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.get("/status")
async def status():
    global global_queue_size
    global total_processed_files
    return {
        "total_files_processed": total_processed_files,
        "queue_size": global_queue_size
    }

@app.get("/search-files")
async def search_files(
    file_id: Optional[int] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
):
    db_connection = setup_database()
    cursor = db_connection.cursor()
    
    query = "SELECT * FROM processed_files WHERE 1=1"
    params = []

    if file_id:
        query += " AND id = ?"
        params.append(file_id)
    
    if start_time:
        try:
            start_datetime = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")
            query += " AND recorded_at >= ?"
            params.append(start_datetime)
        except ValueError:
            return {"error": "Invalid start_time format. Use YYYY-MM-DD_HH-MM-SS"}
    
    if end_time:
        try:
            end_datetime = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S")
            query += " AND recorded_at <= ?"
            params.append(end_datetime)
        except ValueError:
            return {"error": "Invalid end_time format. Use YYYY-MM-DD_HH-MM-SS"}

    cursor.execute(query, params)
    results = cursor.fetchall()
    
    db_connection.close()
    
    return [
        {
            "id": row[0],
            "file_path": row[1],
            "transcription": json.loads(row[2]),
            "status": row[3],
            "recorded_at": row[4],
            "processed_at": row[5]
        }
        for row in results
    ]


@app.get("/api-key-status")
async def api_key_status():
    return {
        "total_keys": len(api_keys),
        "active_keys": len([key for key in api_keys if key.strip()])
    }

def display_transcription(transcription):
    logging.debug(f"Transcription: {transcription['text']}")
    for segment in transcription['segments']:
        logging.debug(f"Segment {segment['id']}: {segment['text']}")
        logging.debug(f"  Start: {segment['start']}, End: {segment['end']}")
    logging.debug("---")

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

def get_total_files(db_connection):
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM processed_files")
    return cursor.fetchone()[0]

def setup_database():
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

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transcription_stats
                 (timestamp TEXT, file_path TEXT, duration REAL, transcription_time REAL, speed REAL)''')
    conn.commit()
    return conn

def log_stats_to_db(db_conn, file_path, duration, transcription_time, speed):
    c = db_conn.cursor()
    c.execute("INSERT INTO transcription_stats VALUES (?, ?, ?, ?, ?)",
              (time.strftime('%Y-%m-%d %H:%M:%S'), file_path, duration, transcription_time, speed))
    db_conn.commit()

def clean_stats(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM transcription_stats")
    conn.commit()
    conn.close()
    logging.info("Transcription stats have been cleaned.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for transcription using either Groq or Faster Whisper.")
    parser.add_argument("--language", default="pt", help="Language code for transcription (default: pt)")
    parser.add_argument("--provider", default="faster_whisper", choices=["groq", "faster_whisper"], help="Transcription provider (default: faster_whisper)")
    
    # Faster Whisper specific arguments
    faster_whisper_group = parser.add_argument_group('Faster Whisper options')
    faster_whisper_group.add_argument("--model_size", default="large-v3", help="Model size for Faster Whisper (default: large-v3)")
    faster_whisper_group.add_argument("--device", default="cuda", help="Device for Faster Whisper (default: cuda)")
    faster_whisper_group.add_argument("--compute_type", default="float16", help="Compute type for Faster Whisper (default: float16)")
    faster_whisper_group.add_argument("--batch_size", type=int, default=16, help="Batch size for Faster Whisper (default: 16)")
    faster_whisper_group.add_argument("--beam_size", type=int, default=5, help="Beam size for transcription (default: 5)")
    faster_whisper_group.add_argument("--word_timestamps", action="store_true", help="Enable word-level timestamps")
    faster_whisper_group.add_argument("--best_of", type=int, default=5, help="Number of candidates to generate for each audio segment (default: 5)")
    faster_whisper_group.add_argument("--patience", type=float, default=1, help="Beam search patience factor (default: 1)")
    faster_whisper_group.add_argument("--length_penalty", type=float, default=1, help="Exponential penalty factor for sequence length (default: 1)")
    faster_whisper_group.add_argument("--temperature", type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="Temperature for sampling (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])")
    faster_whisper_group.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="Threshold for the compression ratio (default: 2.4)")
    faster_whisper_group.add_argument("--log_prob_threshold", type=float, default=-1.0, help="Threshold for the average log probability (default: -1.0)")
    faster_whisper_group.add_argument("--no_speech_threshold", type=float, default=0.6, help="Threshold for the no_speech probability (default: 0.6)")
    faster_whisper_group.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalty applied to repeated tokens (default: 1.0)")
    faster_whisper_group.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Prevent repetitions of ngrams with this size (default: 0)")
    faster_whisper_group.add_argument("--log_prob_low_threshold", type=float, default=None, help="Low threshold for log probability (default: None)")
    faster_whisper_group.add_argument("--initial_prompt", type=str, default=None, help="Optional text to provide as a prompt for the first window")
    faster_whisper_group.add_argument("--prefix", type=str, default=None, help="Optional text to provide as a prefix for the first window")
    faster_whisper_group.add_argument("--suppress_blank", action="store_true", help="Suppress blank outputs at the beginning of the sampling")
    faster_whisper_group.add_argument("--suppress_tokens", type=int, nargs='+', default=[-1], help="List of token IDs to suppress")
    faster_whisper_group.add_argument("--prepend_punctuations", type=str, default=u"\"'¿([{-", help="Punctuations to prepend to each segment")
    faster_whisper_group.add_argument("--append_punctuations", type=str, default=u"\"'.。,，!！?？:：)]}、", help="Punctuations to append to each segment")
    faster_whisper_group.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of tokens to sample")
    faster_whisper_group.add_argument("--hotwords", type=str, default=None, help="Comma-separated list of words to boost in the transcription")
    faster_whisper_group.add_argument("--without_timestamps", action="store_true", help="Only sample text tokens, without timestamps")

    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--clean-stats", action="store_true", help="Clean the transcription stats database")
    parser.add_argument("--stats-db", type=str, default="transcription_stats.db", help="Path to the stats database")
    parser.add_argument("--database", type=str, default="processed_files.db", help="Path to the main database")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if args.clean_stats:
        clean_stats(args.stats_db)
        sys.exit(0)

    db_connection = init_db(args.database)
    stats_db_conn = init_db(args.stats_db)

    # Separate initialization parameters
    init_kwargs = {
        "model_size": args.model_size,
        "device": args.device,
        "compute_type": args.compute_type,
    }

    # Transcription parameters
    transcribe_kwargs = {
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "word_timestamps": args.word_timestamps,
        "best_of": args.best_of,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "log_prob_threshold": args.log_prob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "log_prob_low_threshold": args.log_prob_low_threshold,
        "initial_prompt": args.initial_prompt,
        "prefix": args.prefix,
        "suppress_blank": args.suppress_blank,
        "suppress_tokens": args.suppress_tokens,
        "prepend_punctuations": args.prepend_punctuations,
        "append_punctuations": args.append_punctuations,
        "max_new_tokens": args.max_new_tokens,
        "hotwords": args.hotwords,
        "without_timestamps": args.without_timestamps,
    }
    
    if args.provider == "groq":
        provider = create_provider(args.provider, api_keys=api_keys)
    else:
        provider = create_provider(args.provider, **init_kwargs)

    log_level = getattr(logging, args.log_level.upper())

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(start_file_processing(args.language, provider, db_connection, stats_db_conn, **transcribe_kwargs))
    
    db_connection.close()
    stats_db_conn.close()
    print("Shutdown complete.")
    sys.exit(0)

