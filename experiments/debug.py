import socket
import sys
from pydub import AudioSegment
from pydub.playback import play

def start_audio_stream():
    # Load the audio file
    audio = AudioSegment.from_file("Gravando.mp3")
    
    # Convert audio to raw data
    raw_data = audio.raw_data
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('192.168.112.188', 1888))
            print("Connected to server. Streaming audio...")
            
            # Send the audio data in chunks
            for i in range(0, len(raw_data), 1024):
                s.sendall(raw_data[i:i+1024])
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_audio_stream()