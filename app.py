# app.py

import streamlit as st
from src.speech_to_text import SpeechToText
from src.text_to_speech import TextToSpeech
from src.llm_handler_groq import LLMHandler
from src.rag_engine import RAGEngine
from src.memory_manager import MemoryManager
from src.intent_recognizer import IntentRecognizer
import tempfile
import os

# Streamlit page config
st.set_page_config(page_title="Voice Assistant", layout="centered")
st.title("🗣️ Voice Assistant")

# Initialize modules
llm = LLMHandler()
memory = MemoryManager()
stt = SpeechToText()
tts = TextToSpeech()
rag = RAGEngine()
intent_recognizer = IntentRecognizer()

# Upload audio input
audio_file = st.file_uploader("🎤 Upload your voice (WAV or MP3)", type=["wav", "mp3"])

if audio_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_audio_path = tmp_file.name

    try:
        # 1. Transcribe audio using SpeechToText instance
        user_input = stt.transcribe_audio_file(tmp_audio_path)
        st.write("📝 Transcribed Text:", user_input)

        # 2. Recognize intent using IntentRecognizer instance
        intent = intent_recognizer.recognize_intent(user_input)
        st.write("🔍 Intent Detected:", intent)

        # 3. Get contextual answer from RAG + LLM
        response = rag.get_contextual_answer(user_input, llm, memory)
        st.success("🤖 Assistant Response:")
        st.markdown(response)

        # 4. Convert assistant response to speech using TextToSpeech instance
        audio_path = tts.speak_text(response)

        # 5. Play audio
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

    finally:
        os.remove(tmp_audio_path)
