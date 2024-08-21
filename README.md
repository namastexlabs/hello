# Hello 🎙️

Hello is a project aimed at democratizing audio processing and transcription services. It provides an automated solution for monitoring folders, processing audio files, and generating transcriptions using either Faster Whisper (self-hosted) or Groq Whisper.

## 🚀 Features

- 📁 Automatic folder monitoring for new audio files
- 🔄 Real-time audio processing and transcription
- 🗄️ SQLite database for storing processed files and transcriptions
- 📊 Performance tracking and statistics
- 🌐 FastAPI server for status updates and file searching
- 🔌 Support for multiple transcription providers (Faster Whisper and Groq Whisper)
- 📄 CSV export of transcription data

## 🛠️ Installation

<details>
<summary>Click to expand installation instructions</summary>

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for Faster Whisper)
- NVIDIA CUDA Toolkit 12.x
- cuBLAS for CUDA 12
- cuDNN 8 for CUDA 12
- FFmpeg

### NVIDIA Library Installation

<details>
<summary>Option 1: Use Docker</summary>

The libraries are pre-installed in official NVIDIA CUDA Docker images:
- nvidia/cuda:12.0.0-runtime-ubuntu20.04
- nvidia/cuda:12.0.0-runtime-ubuntu22.04
</details>

<details>
<summary>Option 2: Install with pip (Linux only)</summary>

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```
Note: Ensure you're using cuDNN 8, as version 9+ may cause issues.
</details>

<details>
<summary>Option 3: Download from Purfview's repository (Windows & Linux)</summary>

Download the required NVIDIA libraries from Purfview's whisper-standalone-win repository. Extract the archive and add the library directory to your system's PATH.
</details>

For detailed installation instructions, refer to the official NVIDIA documentation.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/namastexlabs/hello.git
   cd hello
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   GROQ_API_KEYS=your_groq_api_key1,your_groq_api_key2
   RECORDINGS_PATH=./recordings
   ```

</details>

## 🚀 Usage

Run the main script with desired options:

```bash
python main.py --language pt --provider faster_whisper --model_size large-v3
```

<details>
<summary>Click to see available command-line arguments</summary>

- `--language`: Language code for transcription (default: pt)
- `--provider`: Transcription provider (choices: groq, faster_whisper; default: faster_whisper)
- `--model_size`: Model size for Faster Whisper (default: large-v3)
- `--device`: Device for Faster Whisper (default: cuda)
- `--compute_type`: Compute type for Faster Whisper (default: float16)
- `--log-level`: Set the logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO)
- `--clean-stats`: Clean the transcription stats database
- `--stats-db`: Path to the stats database (default: transcription_stats.db)
- `--database`: Path to the main database (default: processed_files.db)

For a full list of Faster Whisper-specific options, run:

```bash
python main.py --help
```

</details>

## 🌐 API Endpoints

- `/healthz`: Health check endpoint
- `/status`: Get current processing status
- `/search-files`: Search processed files with optional filters
- `/api-key-status`: Check the status of API keys

## 📊 Example Project

<details>
<summary>Click to see example project details</summary>

The example project (TODO) will showcase an end-to-end solution that:

1. Captures office activity
2. Transcribes recordings every few minutes
3. Saves timestamped database records
4. Provides API access to transcriptions

This setup aims to facilitate easier access to transcriptions for agent systems.

</details>

## 🗺️ Roadmap

- [ ] Implement MONITOR_FOLDER environment variable for dynamic folder monitoring
- [ ] Develop a user interface for easier management and visualization
- [ ] Implement real-time audio streaming and transcription
- [ ] Optimize performance for large-scale deployments
- [ ] Develop plugins for popular audio recording software

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🌟 Community

Join our Discord community to discuss the project, get help, and contribute:
https://discord.gg/MXa5GsVcCB

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for the efficient transcription engine
- [Groq](https://groq.com/) for their Whisper API
- All contributors and supporters of the project