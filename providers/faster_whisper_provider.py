from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import Dict, Any, Optional

class FasterWhisperProvider:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, file_path: str, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Parameters for transcription
        transcribe_params = {
            # Number of beams for beam search. Higher values can improve accuracy but increase computation time.
            # Typical values: 1-10. Default: 5
            "beam_size": 5,

            # Number of candidates to generate for each audio segment. Should be >= beam_size.
            # Higher values can improve quality but increase computation time. Default: 5
            "best_of": 5,

            # Beam search patience factor. Higher values make beam search more likely to end early.
            # Typical range: 0.5-2.0. Default: 1
            "patience": 1,

            # Exponential penalty factor for sequence length. Values < 1 favor shorter sequences, > 1 favor longer ones.
            # Typical range: 0.5-2.0. Default: 1
            "length_penalty": 1,

            # Penalty applied to the score of previously generated tokens (set > 1 to penalize).
            "repetition_penalty": 1,

            # Prevent repetitions of ngrams with this size (set 0 to disable).
            "no_repeat_ngram_size": 0,

            # Temperature for sampling. Can be a single value or a list of values.
            # Higher values increase randomness in the output. Lower values make the output more deterministic.
            # Typical range: 0.0-1.0. Default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

            # Threshold for the compression ratio. If the gzip compression ratio is above this value, treat as failed.
            # Higher values allow more repetitive text. Typical range: 2.0-3.0. Default: 2.4
            "compression_ratio_threshold": 2.4,

            # Threshold for the average log probability. If below this value, treat as failed.
            # Higher values filter out low-confidence transcriptions. Typical range: -1.0 to -0.5. Default: -1.0
            "log_prob_threshold": -1.0,

            # This parameter alone is sufficient to skip an output text,
            # whereas log_prob_threshold also looks for appropriate no_speech_threshold value.
            # This value should be less than log_prob_threshold.
            "log_prob_low_threshold": None,

            # Threshold for the no_speech probability. If higher than this and the average log probability is below
            # log_prob_threshold, consider the segment as silent. Typical range: 0.3-0.8. Default: 0.6
            "no_speech_threshold": 0.6,

            # If True, include word-level timestamps in the output. Can slightly reduce speed. Default: False
            "word_timestamps": False,

            # Number of audio segments to process in parallel. Higher values can speed up processing but use more memory.
            # Typical range: 1-32, depending on available GPU memory. Default: 16
            "batch_size": 16,

            # Optional text to provide as a prompt for the first window.
            "initial_prompt": None,

            # Optional text to provide as a prefix for the first window.
            "prefix": None,

            # If True, suppress blank outputs at the beginning of the sampling. Default: True
            "suppress_blank": True,

            # List of token IDs to suppress. -1 will suppress a default set of symbols. Default: [-1]
            "suppress_tokens": [-1],

            # Punctuations to prepend to each segment
            "prepend_punctuations": u"\"'¿([{-",

            # Punctuations to append to each segment
            "append_punctuations": u"\"'.。,，!！?？:：)]}、",

            # Maximum number of tokens to sample. Set to None for no limit.
            "max_new_tokens": None,

            # Optional comma-separated list of words to boost in the transcription
            "hotwords": None,

            # If True, only sample text tokens, without timestamps. Can speed up inference. Default: True
            "without_timestamps": True,
        }
        
        # Update with user-provided parameters
        transcribe_params.update(kwargs)

        # Use batched model for transcription
        segments, info = self.batched_model.transcribe(
            file_path,
            language=language,
            **transcribe_params
        )

        # Convert generator to list
        segments = list(segments)

        # Prepare the result in a format similar to Groq's output
        result = {
            "text": " ".join(segment.text for segment in segments),
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [{"word": word.word, "start": word.start, "end": word.end} for word in segment.words] if transcribe_params["word_timestamps"] else None
                }
                for segment in segments
            ],
            "language": info.language,
            "language_probability": info.language_probability
        }

        return result

def create_faster_whisper_provider(model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
    return FasterWhisperProvider(model_size, device, compute_type)