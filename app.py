# app.py

import streamlit as st
import tempfile
import os
from src.speech_to_text import SpeechToText
from src.text_to_speech import TextToSpeech
from src.llm_handler_groq import LLMHandler
from src.rag_engine import RAGEngine
from src.memory_manager import MemoryManager
from src.intent_recognizer import IntentRecognizer

st.set_page_config(page_title="🎙️ Voice Assistant", layout="centered")
st.title("🧠 AI Voice Assistant")

# Initialize core modules
llm = LLMHandler()
memory = MemoryManager()
stt = SpeechToText()
tts = TextToSpeech()
rag = RAGEngine()
intent_recognizer = IntentRecognizer()

# Step 1: Record audio using mic
st.info("Click below to record your voice:")
audio_bytes = st.audio_recorder(label="🎤 Record your voice", format="audio/wav")

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    try:
        # Step 2: Transcribe
        user_input = stt.transcribe_audio_file(tmp_audio_path)
        st.write("📝 Transcribed Text:", user_input)

        # Step 3: Intent Recognition
        intent = intent_recognizer.recognize_intent(user_input)
        st.write("🔍 Detected Intent:", intent)

        # Step 4: Get Contextual Answer
        response = rag.get_contextual_answer(user_input, llm, memory)
        st.success("🤖 Assistant Response:")
        st.markdown(response)

        # Step 5: Text-to-Speech Response
        response_audio_path = tts.speak_text(response)

        with open(response_audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

    finally:
        os.remove(tmp_audio_path)
