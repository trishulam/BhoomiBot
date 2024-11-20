import sounddevice as sd
import wavio
import base64
import requests
from playsound import playsound
import os
import openai
import tiktoken
from openai import OpenAI
from tts_api import TextToSpeech
from dotenv import load_dotenv
import simpleaudio as sa
import wave
from pydub import AudioSegment
import io
import time
import gc
import numpy as np
from langdetect import detect
import langdetect

# Load environment variables
load_dotenv()

# Initialize ASR and TTS classes
user_id = os.getenv("USER_ID")
ulca_api_key = os.getenv("ULCA_API_KEY")

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
print(f"Loaded API key: {openai.api_key[:6]}...") # Print first 6 chars for verification

def initialize_openai_client():
    """Initialize OpenAI client with API key"""
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"Initializing client with key: {api_key[:6]}...") # Print first 6 chars for verification
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")
    return OpenAI(api_key=api_key)

def count_tokens(model, prompt_text):
    """Calculate token count for a prompt"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt_text)
    return len(tokens)

def record_audio(filename="input.wav", silence_threshold=1000, silence_duration=1.5, sample_rate=48000):
    """
    Record audio from the microphone after pressing Enter, until a significant pause is detected.
    
    Args:
        filename: Output filename
        silence_threshold: RMS threshold for silence detection
        silence_duration: Duration of silence in seconds
        sample_rate: Audio sample rate
    """
    input("Press Enter to start recording... (After starting, pause when finished speaking)")
    print("Recording... (Speak and pause when finished)")
    
    # Initialize variables
    chunk_duration = 0.1  # Process audio in 100ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    recording = []
    silence_chunks = 0
    required_silence_chunks = int(silence_duration / chunk_duration)
    
    # Add a flag to ensure minimum recording duration
    min_recording_duration = 0.5  # Minimum 0.5 seconds of recording
    min_chunks = int(min_recording_duration / chunk_duration)
    chunks_recorded = 0
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        recording.append(indata.copy())
        
        # Calculate RMS of the chunk
        rms = np.sqrt(np.mean(indata**2))
        
        # Update silence counter
        nonlocal silence_chunks, chunks_recorded
        chunks_recorded += 1
        
        if rms * 32767 < silence_threshold:  # Convert to 16-bit scale
            silence_chunks += 1
        else:
            silence_chunks = 0
            
    # Start recording
    stream = sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        dtype='float32',
        callback=audio_callback,
        blocksize=chunk_samples
    )
    
    with stream:
        while True:
            sd.sleep(int(chunk_duration * 1000))
            # Only allow stopping if minimum recording duration is met
            if chunks_recorded >= min_chunks and silence_chunks >= required_silence_chunks:
                break
    
    # Convert and save the recording
    if recording:
        recording = np.concatenate(recording, axis=0)
        recording = (recording * 32767).astype(np.int16)
        wavio.write(filename, recording, sample_rate, sampwidth=2)
        print("Recording complete.")
        return filename
    else:
        print("No audio recorded.")
        return None

def convert_audio_to_base64(filename):
    """Convert audio file to base64 encoding."""
    with open(filename, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64

def play_audio_from_base64(audio_base64):
    """Play audio from a base64 string."""
    temp_files = []
    audio_objects = []
    
    try:
        # Decode base64 to audio data
        decoded_audio = base64.b64decode(audio_base64)
        
        # Create a temporary file for the raw audio
        temp_raw = "temp_raw.wav"
        temp_files.append(temp_raw)
        with open(temp_raw, "wb") as f:
            f.write(decoded_audio)
        
        # Convert using pydub
        audio = AudioSegment.from_file(temp_raw)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_sample_width(2)
        
        # Export to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Play using sounddevice
        sd.play(samples, audio.frame_rate)
        sd.wait()  # Wait until the audio is finished playing
        
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {e}")
        
        # Clear any audio objects
        for obj in audio_objects:
            try:
                del obj
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear sounddevice
        try:
            sd.stop()
        except:
            pass

def play_openai_tts(text):
    """Handle OpenAI TTS generation and playback"""
    try:
        client = initialize_openai_client()
        temp_file = "temp_speech.mp3"
        
        print("Generating audio response using OpenAI TTS...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save the audio
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        # Convert and play
        audio = AudioSegment.from_mp3(temp_file)
        samples = np.array(audio.get_array_of_samples())
        
        print("Playing audio response...")
        sd.play(samples, audio.frame_rate)
        sd.wait()
        
    except Exception as e:
        print(f"OpenAI TTS Error: {str(e)}")
        return False
    finally:
        try:
            os.remove(temp_file)
        except:
            pass
    return True

def generate_and_play_audio(text, language, tts=None, tts_params=None):
    """
    Generate and play audio based on language.
    Uses OpenAI TTS for English and ULCA TTS for other languages.
    """
    try:
        if language == 'en':
            return play_openai_tts(text)
        else:
            # Use existing TTS for Hindi and Tamil
            if not tts or not tts_params:
                raise ValueError("TTS service not properly initialized")

            print("Generating audio response using ULCA TTS...")
            audio_response_base64 = tts.generate_audio(text, tts_params)
            
            if not audio_response_base64:
                raise ValueError("Failed to generate audio response")

            print("Playing audio response...")
            play_audio_from_base64(audio_response_base64)
            
        print("Audio playback completed.")
        return True
        
    except Exception as e:
        print(f"Audio generation error: {str(e)}")
        return False

def get_llm_response(user_text, language="English", model="gpt-3.5-turbo", conversation_history=None):
    """
    Generates a response from the ChatGPT API in the specified language while maintaining conversation context.
    """
    # Initialize client
    client = initialize_openai_client()
    
    # Initialize conversation history if None
    if conversation_history is None:
        conversation_history = []
    
    system_content = """

You are BhoomiBot, a knowledgeable AI assistant built to support farmers and agricultural enthusiasts in alignment with the IndiGrow app’s principles. IndiGrow combines traditional farming wisdom with modern technology to guide farmers toward sustainable, eco-friendly practices, with a special emphasis on techniques rooted in Tamil Nadu's agricultural heritage.

### BhoomiBot's Role:
1. **Sustainable and Traditional Practices**: Always provide advice that aligns with eco-friendly, organic farming practices, especially traditional methods from Tamil Nadu. When answering questions on practices like making Panchagavya, Jeevamrutham, or natural pest control, think deeply about the cultural, environmental, and historical aspects of these methods, and provide insights that respect local wisdom. Avoid suggesting chemical fertilizers or synthetic pesticides, focusing instead on natural soil enrichment and crop protection methods.

2. **Culturally Relevant and Deep Understanding**: Tailor responses to reflect the cultural importance and practical benefits of Tamil Nadu's traditional farming methods, like natural fertilizers, crop rotation, and indigenous pest control. Answer with insights that are relevant, emphasizing their long-term benefits, ease of application, and environmental impact, such as soil health, water conservation, and biodiversity enhancement.

3. **Community Empowerment and Support**: Encourage a community-centered approach to farming, promoting knowledge-sharing and resource exchange within the IndiGrow platform. Highlight the benefits of collaboration and support from community volunteers.

4. **Simple, Accessible Language**: Use easy-to-understand, supportive language to make sustainable farming practices accessible to all farmers, regardless of literacy or technical skills. Each response should empower farmers to confidently adopt eco-friendly practices.

5. **Multi-lingual and Culturally Sensitive**: Respond in Tamil, Hindi, or English according to user preference. Use culturally relevant terminology that resonates with Tamil Nadu's farming communities, enhancing relatability.

6. **Motivational and Friendly Tone**: Approach responses with a warm, positive tone, recognizing the challenges of farming and encouraging sustainable practices with empathy and support.

7. **Concise and To-the-Point Answers**: Keep answers focused and direct, with a maximum of 100 words. Avoid lengthy responses, maintaining brevity without sacrificing depth. Use proper punctuation to ensure clarity, with periods, commas, and question marks as needed.

8. **Guidance on Panchatatva Farm Techniques and Principles**: Be ready to explain sustainable methods demonstrated by the Panchatatva Farm, such as energy conversion, topography-based water management, soil moisture conservation, solar drying, fermentation, and mulching, and how they integrate with Tamil Nadu’s traditional methods. Emphasize using natural elements — soil, water, energy, and biodiversity — harmoniously within the farming system.

### App Navigation Assistance:
1. **Splash Screen**: Users first see the IndiGrow logo on the splash screen. No action is needed.
   
2. **Onboarding Screen**: Guide new users through the onboarding process by tapping “Get Started” to proceed to the login screen.

3. **Login Screen**: Explain how to log in as a **Farmer** or **Community Member** by entering a mobile number and selecting the appropriate role, then tapping "Let’s Farm" to enter the main app.

4. **Home Screen**:
   - **Features**: Direct users to view weather updates, soil moisture, and crop selection options. Each crop card provides information about growing days and a "Start" button for more details.
   - **User Action**: Users can tap on a crop (e.g., Rice, Wheat, Ragi) to access crop-specific farming guidance.

5. **Crop-Specific Pages**:
   - **Purpose**: Explain how each crop-specific page breaks down farming tasks by stages (e.g., Land Preparation, Weeding, Pest Control).
   - **User Action**: Farmers can start tasks, view details, mark them as complete, and notify community members or sellers about their progress through options like "Notify Community."

6. **Volunteers Page**: Help users access volunteer information, view expertise (e.g., Seeding, Harvesting), and contact volunteers for support in completing tasks.

7. **Community Screen**: Guide users to connect with community members for knowledge-sharing and collaboration, showing available members and their recent activities.

8. **Hamburger Menu**: Explain the options available in the main menu, such as accessing the Home page, Contact Us, and Logout features.

9. **Page Under Construction**: Inform users if they encounter a page under construction that it will be available in future updates.

### Your Mission:
Provide actionable, insightful, and concise guidance on traditional farming practices, with a special focus on Tamil Nadu’s agricultural heritage. Additionally, guide users in navigating the IndiGrow app, helping them access features related to crop selection, community support, and volunteer connections. Your goal is to empower users to adopt sustainable farming methods while seamlessly utilizing the app’s features to enhance their farming experience.

By balancing depth with brevity, BhoomiBot enables farmers to access valuable insights while maintaining clarity, respect for their time, and ease of navigation within the IndiGrow app.


    """
    
    # Build messages array with conversation history
    messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # Add current user message
    user_content = f"Answer in {language}: {user_text}"
    messages.append({"role": "user", "content": user_content})

    # Calculate tokens to ensure we're within limits
    prompt_text = " ".join([m["content"] for m in messages])
    token_count = count_tokens(model, prompt_text)
    
    token_limit = 8192
    if token_count > token_limit:
        # If exceeding token limit, remove older messages but keep system prompt
        while token_count > token_limit and len(messages) > 2:
            messages.pop(1)  # Remove the second message (keeping system message)
            prompt_text = " ".join([m["content"] for m in messages])
            token_count = count_tokens(model, prompt_text)

    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        top_p=1
    )

    llm_response = response.choices[0].message.content.strip()
    
    # Add assistant's response to conversation history
    conversation_history = messages[1:] + [{"role": "assistant", "content": llm_response}]
    
    return llm_response, conversation_history

def transcribe_audio(audio_file):
    """
    Transcribe audio using OpenAI's Whisper model.
    Returns transcribed text and detected language.
    """
    try:
        client = initialize_openai_client()
        with open(audio_file, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="verbose_json"  # This gives us both text and detected language
            )
        return transcription.text, transcription.language
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None, None

def map_language_code(whisper_lang):
    """
    Map Whisper language codes to TTS language codes.
    """
    language_map = {
        'english': 'en',
        'hindi': 'hi',
        'tamil': 'ta',
        # Add more mappings as needed
        
        # Also add direct codes in case Whisper returns them
        'en': 'en',
        'hi': 'hi',
        'ta': 'ta'
    }
    return language_map.get(whisper_lang.lower(), 'en')  # Default to 'en' if unknown

def main():
    conversation_history = []
    
    print("Initializing TTS service...")
    tts = TextToSpeech(
        user_id=user_id, 
        ulca_api_key=ulca_api_key, 
        source_language="hi"
    )
    
    print("Fetching initial TTS parameters...")
    tts_params = tts.get_params_from_pipeline()
    
    while True:
        try:
            # Step 1: Record audio from user
            print("\nWaiting for your input...")
            audio_file = record_audio(silence_threshold=1000, silence_duration=1.5)
            if not audio_file:
                print("No audio detected. Please try again.")
                continue

            # Step 2: Transcribe audio using Whisper
            user_text, detected_lang = transcribe_audio(audio_file)
            if not user_text:
                print("Could not transcribe audio. Please try again.")
                continue

            # Map the detected language to supported TTS code
            tts_lang = map_language_code(detected_lang)
            
            print(f"\nTranscribed Text: {user_text}")
            print(f"Detected Language: {detected_lang} (TTS code: {tts_lang})")

            # Check for exit command
            if user_text.lower() in ['exit', 'quit', 'bye']:
                print("Ending conversation. Goodbye!")
                break

            # Step 3: LLM - Get response based on transcribed text
            llm_response, conversation_history = get_llm_response(
                user_text, 
                language=tts_lang,  # Use mapped language code
                conversation_history=conversation_history
            )
            print(f"BhoomiBot: {llm_response}")

            # Generate and play audio response
            if tts_lang == 'en':
                # Use OpenAI TTS for English
                client = initialize_openai_client()
                temp_file = "temp_speech.mp3"
                
                print("Generating audio response using OpenAI TTS...")
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=llm_response
                )
                
                # Save the audio
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                
                # Convert and play
                audio = AudioSegment.from_mp3(temp_file)
                samples = np.array(audio.get_array_of_samples())
                
                print("Playing audio response...")
                sd.play(samples, audio.frame_rate)
                sd.wait()
                
                # Cleanup
                try:
                    os.remove(temp_file)
                except:
                    pass
            else:
                # Use existing ULCA TTS implementation for other languages
                if tts.source_language != tts_lang:
                    print(f"Switching TTS to language: {tts_lang}")
                    tts = TextToSpeech(
                        user_id=user_id, 
                        ulca_api_key=ulca_api_key, 
                        source_language=tts_lang
                    )
                    tts_params = tts.get_params_from_pipeline()
                
                if not tts_params:
                    print("Retrying TTS parameters...")
                    tts_params = tts.get_params_from_pipeline()
                    if not tts_params:
                        print("Warning: TTS unavailable. Continuing with text-only response.")
                        continue

                print("Generating audio response...")
                audio_response_base64 = tts.generate_audio(llm_response, tts_params)
                
                if not audio_response_base64:
                    print("Warning: No audio generated. Continuing with text-only response.")
                    continue

                print("Playing audio response...")
                play_audio_from_base64(audio_response_base64)
                print("Audio playback completed.")

        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Let's continue with the conversation...")
            continue

        # Small delay between iterations
        time.sleep(0.1)

if __name__ == "__main__":
    print("Starting conversation with BhoomiBot. Say 'exit', 'quit', or 'bye' to end the conversation.")
    try:
        main()
    except KeyboardInterrupt:
        print("\nConversation ended by user. Goodbye!")
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())