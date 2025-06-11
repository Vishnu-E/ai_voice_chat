import whisper
import speech_recognition as sr
import io
import tempfile
import os
from pydub import AudioSegment
import numpy as np
from config import Config

class SpeechToText:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def transcribe_audio_file(self, audio_file_path):
        """Transcribe audio file using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio file: {e}")
            return None
    
    def transcribe_audio_data(self, audio_data):
        """Transcribe audio data from bytes using Whisper"""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe the temporary file
            result = self.whisper_model.transcribe(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio data: {e}")
            return None
    
    def listen_from_microphone(self, timeout=5, phrase_time_limit=15):
        """Listen to microphone and return transcribed text"""
        try:
            print("Listening...")
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("Transcribing...")
            # Convert audio to text using Whisper
            audio_data = audio.get_wav_data()
            return self.transcribe_audio_data(audio_data)
            
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            return None
    
    def transcribe_webm_to_text(self, webm_data):
        """Convert WebM audio data to text"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_file:
                wav_path = wav_file.name
            
            # Convert WebM to WAV using pydub
            audio = AudioSegment.from_file(webm_path, format="webm")
            audio.export(wav_path, format="wav")
            
            # Transcribe the WAV file
            result = self.whisper_model.transcribe(wav_path)
            
            # Clean up temporary files
            os.unlink(webm_path)
            os.unlink(wav_path)
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing WebM audio: {e}")
            return None
    
    def is_speech_detected(self, audio_data, threshold=0.01):
        """Check if audio data contains speech above threshold"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) for volume detection
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Normalize RMS to 0-1 range
            normalized_rms = rms / 32767.0
            
            return normalized_rms > threshold
            
        except Exception as e:
            print(f"Error detecting speech: {e}")
            return False