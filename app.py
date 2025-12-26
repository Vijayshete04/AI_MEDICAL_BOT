import os
import gradio as gr
from dotenv import load_dotenv
import logging

from think import encode_image, analyze_image_with_query
from user_voice import record_audio, transcribe_with_groq
from assistant_voice import text_to_speech_with_gtts_old

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
print("Groq Key:", os.environ.get("GROQ_API_KEY"))

# Check for required API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing! Please check your .env file.")

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes.
            What's in this image? Do you find anything wrong with it medically?
            If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also, always answer as if you are answering a real person.
            Do not say 'In the image I see' but say 'With what I see, I think you have ....'
            Do not respond as an AI model in markdown. Your answer should mimic that of an actual doctor, not an AI bot.
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def process_input(audio_file_path, image_file_path):
    try:
        logging.info("Processing input...")
        speech_to_text_output = ""
        assistant_response = "Image not provided."

        # Process audio if provided
        if audio_file_path:
            logging.info("Transcribing audio...")
            speech_to_text_output = transcribe_with_groq("whisper-large-v3", audio_file_path)
        else:
            speech_to_text_output = "No audio input provided."
            
        # Process image if provided
        if image_file_path:
            logging.info("Encoding image...")
            encoded_image = encode_image(image_file_path)
            logging.info("Analyzing image with AI...")
            assistant_response = analyze_image_with_query(speech_to_text_output, encoded_image, MODEL)

        # Convert assistant response to speech
        logging.info("Converting text to speech...")
        output_audio_path = "assistant_response.mp3"
        text_to_speech_with_gtts_old(assistant_response, output_audio_path)

        logging.info("Processing completed successfully.")
        return speech_to_text_output, assistant_response, output_audio_path

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return "Error processing input", error_message, None

# Create Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Speak your question"),
        gr.Image(type="filepath", label="Upload medical image")
    ],
    outputs=[
        gr.Textbox(label="Your Question (Speech to Text)"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response", type="filepath")  # Use type="filepath" for audio playback
    ],
    title="AI Doctor with Vision and Voice",
    description="Upload a medical image and speak your question. The AI doctor will analyze the image and respond both in text and voice."
)

if __name__ == "__main__":
    iface.launch(debug=True)
