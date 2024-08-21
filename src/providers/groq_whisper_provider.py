from groq import Groq
from itertools import cycle
from typing import List

class GroqClientPool:
    def __init__(self, api_keys: List[str]):
        self.clients = [Groq(api_key=key) for key in api_keys]
        self.current_client = cycle(self.clients)

    def get_client(self):
        return next(self.current_client)

class GroqWhisperProvider:
    def __init__(self, api_keys: List[str]):
        self.client_pool = GroqClientPool(api_keys)

    def transcribe(self, file_path: str, language: str):
        with open(file_path, "rb") as file:
            groq_client = self.client_pool.get_client()
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3",
                language=language,
                response_format="verbose_json"
            )
        return transcription

def create_groq_whisper_provider(api_keys: List[str]):
    return GroqWhisperProvider(api_keys)