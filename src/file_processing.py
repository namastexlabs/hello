import asyncio
import logging
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.utils import is_file_ready, is_audio_non_empty, log_stats_to_db, append_to_csv, log_performance_stats, get_stats_from_db
from src.database import get_unprocessed_files, get_total_files
from src.constants import SUPPORTED_EXTENSIONS
from rich.console import Console
from rich.text import Text
from datetime import datetime
import json
import sqlite3

console = Console()

# Global variables
global_queue_size = 0
total_processed_files = 0
shutdown_requested = False
processing_times = []

class FileHandler(FileSystemEventHandler):
    def __init__(self, queue, db_path):
        self.queue = queue
        self.db_path = db_path
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

    def on_moved(self, event):
        if not event.is_directory:
            self.process_file(event.dest_path)

async def process_files(queue, db_path, stats_db_path, language, provider, **provider_kwargs):
    global global_queue_size
    global total_processed_files
    global shutdown_requested
    while not shutdown_requested:
        try:
            # Update queue size at the beginning of each iteration
            global_queue_size = queue.qsize()
            
            if global_queue_size == 0:
                # Queue is empty, wait for a short time and check for new files
                await asyncio.sleep(60)  # Wait for 60 seconds
                unprocessed_files = get_unprocessed_files(db_path, os.getenv('RECORDINGS_PATH', './recordings'))
                for file in unprocessed_files:
                    await queue.put(file)
                global_queue_size = queue.qsize()
                continue  # Start the loop again to process new files if any were added

            file_path = await asyncio.wait_for(queue.get(), timeout=1.0)
            logging.debug(f"File ready for processing: {file_path}")
            
            # Process the file
            await asyncio.shield(process_single_file(file_path, db_path, stats_db_path, language, provider, **provider_kwargs))

            logging.debug(f"Finished processing: {file_path}")
            total_processed_files += 1
            queue.task_done()
            
            # Update queue size after processing
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

async def process_single_file(file_path, db_path, stats_db_path, language, provider, **provider_kwargs):
    global global_queue_size
    global processing_times
    
    # Wait for the file to be completely written
    while not is_file_ready(file_path):
        await asyncio.sleep(1)
        if shutdown_requested:
            return
    
    logging.debug(f"File ready for processing: {file_path}")
    
    start_time = time.time()
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

    with sqlite3.connect(db_path) as db_connection:
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM processed_files WHERE file_path = ?", (file_path,))
        if cursor.fetchone():
            logging.debug(f"File already processed: {file_path}")
            return

        try:
            logging.debug(f"Starting transcription for {file_path}")
            transcription = await asyncio.to_thread(provider.transcribe, file_path, language=language, **provider_kwargs)
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            audio_duration = transcription['segments'][-1]['end'] if transcription['segments'] else 0
            speed = audio_duration / processing_time if processing_time > 0 else 0
            
            log_stats_to_db(stats_db_path, file_path, audio_duration, processing_time, speed)
            
            # Get updated stats from the database
            files_count, total_time, avg_time, median_time, min_time = get_stats_from_db(stats_db_path)
            
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
            
            if not transcription['segments']:
                status.append(" | ", style="white")
                status.append("Skipped: No active speech", style="red")
            
            console.print(status)

            
            # Extract the full text from the transcription
            full_text = transcription.get('text', '')

            # Extract recorded_at from the filename
            try:
                recorded_at_str = os.path.basename(file_path).split('_')[0]
                recorded_at = datetime.strptime(recorded_at_str, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                try:
                    recorded_at = datetime.fromtimestamp(os.path.getmtime(file_path))  # Fallback to file metadata if parsing fails
                except Exception:
                    recorded_at = datetime.now()  # Fallback to current time if metadata extraction fails

            cursor.execute(
                "INSERT OR REPLACE INTO processed_files (file_path, transcription, text, status, recorded_at) VALUES (?, ?, ?, ?, ?)",
                (file_path, json.dumps(transcription), full_text, "processed" if transcription['segments'] else "skipped", recorded_at)
            )
            db_connection.commit()

            append_to_csv(file_path, transcription)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            console.print(f"[bold red]Error processing file {os.path.basename(file_path)}: {str(e)}[/bold red]")

            # Mark the file as errored in the database
            cursor.execute(
                "INSERT OR REPLACE INTO processed_files (file_path, transcription, text, status, recorded_at) VALUES (?, ?, ?, ?, ?)",
                (file_path, json.dumps({"error": str(e)}), "", "error", datetime.now())
            )
            db_connection.commit()

async def start_file_processing(language, provider, db_path, stats_db_path, **provider_kwargs):
    global global_queue_size
    global shutdown_requested
    global processing_times
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    recordings_path = os.getenv('RECORDINGS_PATH', './recordings')
    if not os.path.exists(recordings_path):
        os.makedirs(recordings_path)

    queue = asyncio.Queue()

    unprocessed_files = get_unprocessed_files(db_path, recordings_path)
    for file in unprocessed_files:
        await queue.put(file)

    global_queue_size = queue.qsize()

    logging.info(f"Initial stats:")
    logging.info(f"  Files queued for processing: {len(unprocessed_files)}")
    logging.info(f"  Total files in database: {get_total_files(db_path)}")

    event_handler = FileHandler(queue, db_path)
    observer = Observer()
    observer.schedule(event_handler, path=recordings_path, recursive=False)
    observer.start()

    logging.info(f"Watchdog started. Monitoring folder: {recordings_path}")

    processing_task = asyncio.create_task(process_files(queue, db_path, stats_db_path, language, provider, **provider_kwargs))

    try:
        while not shutdown_requested:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("File processing cancelled")
    finally:
        shutdown_requested = True
        observer.stop()
        await processing_task
        observer.join()
        
        log_performance_stats(processing_times)

    logging.info("File processing shutdown complete")