import os
from gtts import gTTS

def text_to_speech_with_gtts_old(input_text,output_file_path):
    
    language = 'en'
    
    audioobj=gTTS(text=input_text,lang=language,slow=False)
    audioobj.save(output_file_path)
    
input_text = "Hello, how are you?"

text_to_speech_with_gtts_old(input_text,"output.mp3")