import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the .env file
load_dotenv()

# Get the GROQ API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Check your .env file.")

# Set up the STT model
stt_model = "whisper-large-v3"


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Listening...")

            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording finished.")

            # Convert to WAV and then export to MP3
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))

            # Ensure audio is actually saved
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio successfully saved to {file_path}")

            # Verify file size to avoid empty files
            if os.path.getsize(file_path) == 0:
                logging.error("Audio file is empty after saving!")
                raise ValueError("Audio recording failed, empty file detected.")

    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        raise


def transcribe_with_groq(model, audio_filepath):
    """Transcribes an audio file using Groq's Whisper model."""
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Audio file {audio_filepath} not found!")

    # Check file size to avoid sending empty files
    if os.path.getsize(audio_filepath) == 0:
        raise ValueError("Cannot transcribe an empty file!")

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model=model, file=audio_file, language="en")

    return transcription.text


# File path for recorded audio
audio_file_path = "audio.mp3"

# Run recording and transcription
# try:
#     record_audio(audio_file_path)
#     result = transcribe_with_groq(stt_model, audio_file_path)
#     print("Transcription:", result)
# except Exception as error:
#     logging.error(f"Failed to complete process: {error}")
