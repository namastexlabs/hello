import asyncio
import logging
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI
import uvicorn
from src.utils import is_file_ready, is_audio_non_empty, log_stats_to_db, append_to_csv, log_performance_stats
from src.database import setup_database, get_unprocessed_files, get_total_files
from src.constants import SUPPORTED_EXTENSIONS
from rich.console import Console
from rich.text import Text
from datetime import datetime
import json


app = FastAPI()

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