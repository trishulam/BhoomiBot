import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import os
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import wave
import time

# Import functions from main.py
from main import (
    TextToSpeech,
    transcribe_audio,
    map_language_code,
    get_llm_response,
    initialize_openai_client,
    play_audio_from_base64,
    user_id,
    ulca_api_key
)

class BhoomiBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BhoomiBot")
        self.root.geometry("600x800")
        
        # Initialize recording variables
        self.is_recording = False
        self.frames = []
        self.sample_rate = 44100
        
        # Initialize conversation state
        self.conversation_history = []
        
        # Initialize TTS service
        print("Initializing TTS service...")
        self.tts = TextToSpeech(
            user_id=user_id, 
            ulca_api_key=ulca_api_key, 
            source_language="hi"
        )
        self.tts_params = self.tts.get_params_from_pipeline()
        
        # Create GUI elements
        self.create_widgets()
        
    def record_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.frames.append(indata.copy())
            
    def start_recording(self, event):
        self.frames = []
        self.is_recording = True
        self.record_button.config(text="🔴 Recording...")
        self.update_status("Recording... Speak now")
        
        # Start recording stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.record_callback
        )
        self.stream.start()
        
    def stop_recording(self, event):
        self.is_recording = False
        self.record_button.config(text="🎤 Hold to Record")
        self.update_status("Processing...")
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Save recording
        if self.frames:
            temp_file = "temp_recording.wav"
            data = np.concatenate(self.frames, axis=0)
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((data * 32767).astype(np.int16).tobytes())
            
            # Process the recording in a separate thread
            threading.Thread(target=self.process_recording, args=(temp_file,)).start()
        else:
            self.update_status("No audio detected")
        
    def create_widgets(self):
        # Chat display area
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=30)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Record button
        self.record_button = ttk.Button(
            self.root, 
            text="🎤 Hold to Record", 
            command=self.toggle_recording
        )
        self.record_button.pack(pady=10)
        
        # Bind button press and release events
        self.record_button.bind('<ButtonPress-1>', self.start_recording)
        self.record_button.bind('<ButtonRelease-1>', self.stop_recording)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
    def update_chat(self, message, sender="You"):
        self.chat_area.insert(tk.END, f"\n{sender}: {message}\n")
        self.chat_area.see(tk.END)
        
    def update_status(self, status):
        self.status_label.config(text=status)
        self.root.update()
        
    def toggle_recording(self):
        pass  # Placeholder for the button command
        
    def process_recording(self, audio_file):
        try:
            # Transcribe audio
            user_text, detected_lang = transcribe_audio(audio_file)
            if not user_text:
                self.update_status("Could not transcribe audio")
                return
                
            # Update chat with transcription
            self.update_chat(user_text)
            
            # Map language and get response
            tts_lang = map_language_code(detected_lang)
            self.update_status(f"Detected language: {detected_lang}")
            
            # Get LLM response
            llm_response, self.conversation_history = get_llm_response(
                user_text, 
                language=tts_lang,
                conversation_history=self.conversation_history
            )
            
            # Update chat with bot response
            self.update_chat(llm_response, "BhoomiBot")
            
            # Generate and play audio response
            if tts_lang == 'en':
                # Use OpenAI TTS for English
                self.update_status("Generating English response...")
                client = initialize_openai_client()
                temp_file = "temp_speech.mp3"
                
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=llm_response
                )
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                
                audio = AudioSegment.from_mp3(temp_file)
                samples = np.array(audio.get_array_of_samples())
                
                self.update_status("Playing response...")
                sd.play(samples, audio.frame_rate)
                sd.wait()
                
                try:
                    os.remove(temp_file)
                except:
                    pass
            else:
                # Use ULCA TTS for other languages
                if self.tts.source_language != tts_lang:
                    self.update_status(f"Switching to {tts_lang}...")
                    self.tts = TextToSpeech(
                        user_id=user_id, 
                        ulca_api_key=ulca_api_key, 
                        source_language=tts_lang
                    )
                    self.tts_params = self.tts.get_params_from_pipeline()
                
                self.update_status("Generating response...")
                audio_response_base64 = self.tts.generate_audio(llm_response, self.tts_params)
                
                if audio_response_base64:
                    self.update_status("Playing response...")
                    play_audio_from_base64(audio_response_base64)
            
            self.update_status("Ready")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = BhoomiBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
