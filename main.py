import argparse
import logging
import signal
import sys
import os
import asyncio
from dotenv import load_dotenv
from src.providers.groq_whisper_provider import create_groq_whisper_provider
from src.providers.faster_whisper_provider import create_faster_whisper_provider
from src.file_processing import start_file_processing
from src.database import init_db, clean_stats, get_previous_transcription_thread_safe
from src.api import app
import uvicorn

# Load environment variables from .env file
load_dotenv()

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("Shutdown requested. Cleaning up...")
    shutdown_requested = True

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

    if db_connection is None or stats_db_conn is None:
        logging.error("Failed to initialize one or both databases. Exiting.")
        sys.exit(1)

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
        # Load API keys from environment variables
        api_keys = [key.strip() for key in os.getenv("GROQ_API_KEYS", "").split(",") if key.strip()]
        if not api_keys:
            raise ValueError("No API keys found. Please set GROQ_API_KEYS in your .env file.")
        provider = create_groq_whisper_provider(api_keys)
    else:
        provider = create_faster_whisper_provider(**init_kwargs)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async def main():
        global shutdown_requested
        # Start the file processing task
        file_processing_task = asyncio.create_task(
            start_file_processing(args.language, provider, args.database, args.stats_db, **transcribe_kwargs)
        )

        # Start the FastAPI server
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())

        # Wait for shutdown signal
        while not shutdown_requested:
            await asyncio.sleep(1)

        # Shutdown tasks
        file_processing_task.cancel()
        await file_processing_task
        server.should_exit = True
        await server_task

        print("Shutdown complete.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        if db_connection:
            db_connection.close()
        if stats_db_conn:
            stats_db_conn.close()
        sys.exit(0)