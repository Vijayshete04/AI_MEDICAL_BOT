import os
import base64
from groq import Groq
from dotenv import load_dotenv  # Import dotenv

# Load the .env file
load_dotenv()

# Get the GROQ API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Use os.getenv instead of os.environ.get

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables. Make sure to create a .env file and load it properly.")

def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_query(query, encoded_image, model):
    """Sends an image along with a query to the Groq model for analysis."""
    client = Groq(api_key=GROQ_API_KEY)  # Pass the API key

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ],
        }
    ]

    chat_completion = client.chat.completions.create(messages=messages, model=model)

    return chat_completion.choices[0].message.content

# Example usage
# encoded_image = encode_image("image.png")
# model = "llama-3.2-90b-vision-preview"
# result = analyze_image_with_query("What is this disease?", encoded_image, model)
# print(result)
