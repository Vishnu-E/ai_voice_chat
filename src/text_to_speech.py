import os
import tempfile
import pygame
from elevenlabs import generate, save, set_api_key, voices
from gtts import gTTS
import io
from config import Config

class TextToSpeech:
    def __init__(self):
        self.use_elevenlabs = bool(Config.ELEVENLABS_API_KEY)
        
        if self.use_elevenlabs:
            set_api_key(Config.ELEVENLABS_API_KEY)
            self.voice_id = Config.VOICE_ID
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=Config.SAMPLE_RATE, 
                          size=-16, channels=2, 
                          buffer=Config.CHUNK_SIZE)
    
    def generate_speech_elevenlabs(self, text):
        """Generate speech using ElevenLabs API"""
        try:
            audio = generate(
                text=text,
                voice=self.voice_id,
                model="eleven_monolingual_v1"
            )
            return audio
        except Exception as e:
            print(f"Error generating speech with ElevenLabs: {e}")
            return None
    
    def generate_speech_gtts(self, text, language='en'):
        """Generate speech using Google Text-to-Speech"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            print(f"Error generating speech with gTTS: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech and return audio data"""
        if self.use_elevenlabs:
            return self.generate_speech_elevenlabs(text)
        else:
            return self.generate_speech_gtts(text)
    
    def speak_text(self, text):
        """Convert text to speech and play it immediately"""
        try:
            audio_data = self.text_to_speech(text)
            if audio_data:
                self.play_audio(audio_data)
                return True
            return False
        except Exception as e:
            print(f"Error speaking text: {e}")
            return False
    
    def play_audio(self, audio_data):
        """Play audio data using pygame"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Load and play audio
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def save_audio_file(self, text, filename):
        """Generate speech and save to file"""
        try:
            audio_data = self.text_to_speech(text)
            if audio_data:
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                return True
            return False
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False
    
    def get_available_voices(self):
        """Get list of available voices (ElevenLabs only)"""
        if self.use_elevenlabs:
            try:
                voice_list = voices()
                return [(voice.voice_id, voice.name) for voice in voice_list]
            except Exception as e:
                print(f"Error getting voices: {e}")
                return []
        else:
            return [("gtts", "Google TTS")]
    
    def set_voice(self, voice_id):
        """Set the voice to use for speech generation"""
        if self.use_elevenlabs:
            self.voice_id = voice_id
            return True
        return False