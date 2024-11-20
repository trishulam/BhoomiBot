#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool
import time
from tqdm import tqdm
from pydub import AudioSegment
import requests
import json
import base64
import os
import io
import ffmpeg
import json

class TextToSpeech:
    def __init__(self, user_id, ulca_api_key, source_language, 
                 sampling_rate=44100, gender="male"):
        """
        Initialize TextToSpeech class with user credentials and language parameters.

        Args:
        user_id (str): User ID for authentication.
        ulca_api_key (str): API key for authentication.
        source_language (str): Source language for Text-to-Speech conversion.
        sampling_rate (int): Sampling rate for audio (default is 44100).
        gender (str): Gender for Text-to-Speech conversion (default is "male").
        """
        self.source_language = source_language
        self.sampling_rate = sampling_rate
        self.gender = gender

        self.headers = {
            "userID": user_id,
            "ulcaApiKey": ulca_api_key
        }

        self.model_pipeline_response = self.get_tts_models_pipeline()
        self.params = self.get_params_from_pipeline()

    def get_tts_models_pipeline(self):
        """
        Retrieve Text-to-Speech models pipeline from ULCA API.

        Returns:
        requests.Response: Response object containing pipeline information.
        """
        supported_languages = ["as", "bn", "brx", "gu", "hi", "kn", "ml", "mni",
                               "mr", "or", "pa", "raj", "ta", "te", "en"]
        
        if self.source_language not in supported_languages:
            print("Specified language is not supported")
            return None
        
        url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": self.source_language
                        }
                    }
                }
            ],
            "pipelineRequestConfig": {
                "pipelineId": "64392f96daac500b55c543cd"
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
        except Exception as e:
            print(response.status_code)
            print(f"Error: {e}")
            print(f"Error message: {response.text}")
            return None

        return response

    def get_params_from_pipeline(self):
        """
        Extract parameters required for Text-to-Speech conversion from API response.

        Args:
        response (requests.Response): Response object containing API response.

        Returns:
        dict: Dictionary containing parameters for Text-to-Speech conversion.
        """
        if not self.model_pipeline_response:
            print("Error: No response received from the API")
            return None
        
        if self.gender not in ["male", "female"]:
            print("Specified gender is not supported")
            return None
        
        inference_api_key = self.model_pipeline_response.json()['pipelineInferenceAPIEndPoint']['inferenceApiKey']
        callback_url = self.model_pipeline_response.json()['pipelineInferenceAPIEndPoint']['callbackUrl']
        name = inference_api_key['name']
        value = inference_api_key['value']
        service_id = self.model_pipeline_response.json()['pipelineResponseConfig'][0]['config'][0]['serviceId']

        params = {
            "inferenceApiKey": inference_api_key, "callbackUrl": callback_url, "name": name, "value": value,
            "serviceId": service_id, "sourceLanguage": self.source_language, "samplingRate": self.sampling_rate,
            "gender": self.gender
        }

        return params

    @staticmethod
    def generate_audio(source, params):
        """
        Generate audio from the given text using Text-to-Speech API.

        Args:
        source (str): Text to be converted to speech.

        Returns:
        str: Base64 encoded audio content.
        """
        if not params:
            print("Error: No parameters received for Text-to-Speech conversion")
            return None
        
        headers = {
            "Accept": "*/*",
            "User-Agent": "Thunder Client (https://www.thunderclient.com)",
            params["name"]: params["value"]
        }

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": params["sourceLanguage"]
                        },
                        "serviceId": params["serviceId"],
                        "gender": params["gender"],
                        "samplingRate": params["samplingRate"]
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": source
                    }
                ]
            }
        }

        try:
            # Send a POST request to the callback URL with the payload
            response = requests.post(params["callbackUrl"], headers=headers,
                                     data=json.dumps(payload))
        except Exception as e:
            print(response.status_code)
            print(f"Error: {e}")
            print(f"Error message: {response.text}")
            return None

        audio_base = response.json()['pipelineResponse'][0]['audio'][0]['audioContent']

        return audio_base

    def read_file(self, file_path):
        """
        Read the input JSON file and return the data as a list of JSON objects.

        Args:
        file_path (str): Path to the input JSON file.

        Returns:
        list: List of JSON objects.
        """
        if not os.path.exists(file_path):
            print(f"Error: File does not exist at {file_path}")
            return None
        
        with open(file_path, "r") as file:
            data = json.load(file)
        
        return data

    @staticmethod
    def adjust_audio_speed(base_audio, target_duration, index, output_dir):
        """
        Speed up the given audio segment by the provided speed factor.

        Args:
        audio_segment (AudioSegment): Input audio segment to be modified.
        speed_factor (float): Speed factor by which the audio will be adjusted.

        Returns:
        AudioSegment: Adjusted audio segment.
        """
        input_audio_file = "input_audio_temp.wav"

        # Decode the base64 encoded audio content
        decoded_audio = base64.b64decode(base_audio)

        # Convert the decoded audio to an AudioSegment object
        audio_segment = AudioSegment.from_file(io.BytesIO(decoded_audio), format="wav")
        
        # Save the audio to a temporary file
        audio_segment.export(input_audio_file, format="wav")

        output_audio_file = f"{output_dir}/adjusted_audio_{index}.wav"

        # Get the duration of the audio segment
        audio_duration = audio_segment.duration_seconds

        # Calculate the speed factor by which the audio will be adjusted
        speed_factor = audio_duration/target_duration

        if 0.99 < speed_factor < 1.01:
            print("No adjustment needed. Speed factor:", speed_factor)
            # Write the audio to output_audio_file
            audio_segment.export(output_audio_file, format="wav")
    
        # Speed up the audio using ffmpeg's atempo filter
        ffmpeg.input(input_audio_file)\
            .filter('atempo', speed_factor)\
            .output(output_audio_file, loglevel="quiet")\
            .run(overwrite_output=True)
        print(f"Audio speed adjusted by {round(speed_factor, 2)}x")

        # Remove the temporary audio files
        os.remove(input_audio_file)
        # os.remove(output_audio_file)
    
    def process(self, json_data, output_dir):
        """
        Read and process the entire list of json objects for Text-to-Speech conversion.

        Args:
        file_path (str) : Path to the input JSON file.
        output_dir (str): Output directory for saving the audio files.
        """
        start = time.time()
        data = json_data
        
        params = self.get_params_from_pipeline()

        text_list = []
        durations = []
        for obj in data:
            text_list.append(obj["text"])
            durations.append(obj["duration"])

        # Multiprocessing
        number_processes = os.cpu_count()
        
        with Pool(processes=number_processes) as pool:
            tts_output = pool.starmap(self.generate_audio, [(text, params) for text in text_list])
        
        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        errors = []
        for index, audio in enumerate(tts_output):
            print(f"Row number: {index}")
            target_duration = durations[index]
            try: 
                self.adjust_audio_speed(audio, target_duration, index, output_dir)
            except Exception as e:
                print(f"Error: {e}")
                print("Error while adjusting audio. Skipping this audio file.")
                errors.append(index)
                continue

        end = time.time()
        print(f"Time taken: {end - start} seconds")
        print("Errors:", errors)
        return errors

# Example usage 
# TO DO : Update for usage for multiprocessing
if __name__ == "__main__":
    
    # User credentials
    user_id = "a2d33d9a5f9441859090b7f2e96b09f3"
    ulca_api_key = "21c903f6b4-0030-4a2e-9d78-cfcb9ec5c19a"

    # Language parameters
    source_language = "ta"
    sampling_rate = 44100 # Default is 44100
    gender = "male" # Default is "male"
    
    # Path to the input JSON file
    file_path = f'../Transcripts/transcript_ta copy.json'
    # Output directory for saving the audio files
    output_dir = f"../Output_ASR/TTS/{source_language}/Test2"
    
    tts = TextToSpeech(user_id, ulca_api_key, source_language, sampling_rate, gender)
    tts.process(file_path, output_dir)