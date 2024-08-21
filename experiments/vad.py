import pyaudio
import websockets
import asyncio
import json
import argparse
import numpy as np

async def start_audio_stream(chunk_length, chunk_offset, processing_strategy, language):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 48000
    OUTPUT_RATE = 16000

    p = pyaudio.PyAudio()

    device_info = p.get_default_input_device_info()
    print(f"Capturing audio from device: {device_info['name']}")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    uri = "ws://192.168.112.188:1888"
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to server. Sending audio config...")

                audio_config = {
                    "type": "config",
                    "data": {
                        "sampleRate": OUTPUT_RATE,
                        "channels": CHANNELS,
                        "language": language,
                        "processing_strategy": processing_strategy,
                        "processing_args": {
                            "chunk_length_seconds": chunk_length,
                            "chunk_offset_seconds": chunk_offset
                        }
                    }
                }
                await websocket.send(json.dumps(audio_config))

                print("Streaming audio...")
                
                # Start a task to receive messages
                receive_task = asyncio.create_task(receive_messages(websocket))
                
                while True:
                    data = stream.read(CHUNK)
                    audio_array = np.frombuffer(data, dtype=np.float32)
                    resampled = resample(audio_array, RATE, OUTPUT_RATE)
                    int16_data = (resampled * 32767).astype(np.int16)
                    await websocket.send(int16_data.tobytes())

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed. Attempting to reconnect...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(5)

async def receive_messages(websocket):
    try:
        while True:
            message = await websocket.recv()
            try:
                transcript_data = json.loads(message)
                display_transcription(transcript_data)
            except json.JSONDecodeError:
                print(f"Received non-JSON message: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")

def display_transcription(transcript_data):
    if 'text' in transcript_data:
        print(f"Transcription: {transcript_data['text']}")
    if 'words' in transcript_data and transcript_data['words']:
        words = ' '.join([word['word'] for word in transcript_data['words']])
        print(f"Words: {words}")
    if 'language' in transcript_data:
        print(f"Detected language: {transcript_data['language']}")
    if 'processing_time' in transcript_data:
        print(f"Processing time: {transcript_data['processing_time']} seconds")
    print("---")

def resample(audio_array, orig_sr, target_sr):
    resampled = np.zeros(int(len(audio_array) * target_sr / orig_sr))
    for i in range(len(resampled)):
        orig_index = i * orig_sr / target_sr
        left_index = int(orig_index)
        right_index = min(left_index + 1, len(audio_array) - 1)
        fraction = orig_index - left_index
        resampled[i] = (1 - fraction) * audio_array[left_index] + fraction * audio_array[right_index]
    return resampled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio streaming with VAD")
    parser.add_argument("--chunk_length", type=float, default=3.0, help="Chunk length in seconds")
    parser.add_argument("--chunk_offset", type=float, default=0.1, help="Chunk offset in seconds")
    parser.add_argument("--processing_strategy", type=str, default="silence_at_end_of_chunk", help="Processing strategy")
    parser.add_argument("--language", type=str, default="portuguese", help="Language for processing")

    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        start_audio_stream(args.chunk_length, args.chunk_offset, args.processing_strategy, args.language)
    )