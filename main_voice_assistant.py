import uuid
from typing import Dict, Optional
from src.speech_to_text import SpeechToText
from src.text_to_speech import TextToSpeech
#from src.llm_handler_openAI import LLMHandler
from src.llm_handler_groq import LLMHandler
from src.rag_engine import RAGEngine
from src.memory_manager import MemoryManager
from src.intent_recognizer import IntentRecognizer
from config import Config

class VoiceAssistant:
    def __init__(self):
        print("Initializing Voice Assistant...")
        
        # Initialize components
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.llm = LLMHandler()
        self.rag = RAGEngine()
        self.memory = MemoryManager()
        self.intent_recognizer = IntentRecognizer()
        
        print("Voice Assistant initialized successfully!")
    
    def setup_knowledge_base(self, urls: list = None, pdf_paths: list = None):
        """Set up the knowledge base with URLs and PDFs"""
        print("Setting up knowledge base...")
        
        if urls:
            for url in urls:
                self.rag.add_url(url)
        
        if pdf_paths:
            for pdf_path in pdf_paths:
                self.rag.add_pdf(pdf_path)
        
        if self.rag.documents:
            self.rag.build_index()
            print(f"Knowledge base ready with {len(self.rag.documents)} documents")
        else:
            print("Warning: No documents added to knowledge base")
    
    def process_voice_input(self, session_id: str, audio_data: bytes = None) -> Dict:
        """Process voice input and return response"""
        try:
            # Convert speech to text
            if audio_data:
                user_text = self.stt.transcribe_audio_data(audio_data)
            else:
                user_text = self.stt.listen_from_microphone()
            
            if not user_text:
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'response_text': "I didn't hear anything. Could you please try again?",
                    'audio_response': None
                }
            
            print(f"User said: {user_text}")
            
            # Process the text input
            return self.process_text_input(session_id, user_text)
            
        except Exception as e:
            error_message = "I'm sorry, I'm having trouble with the audio. Could you try again?"
            return {
                'success': False,
                'error': str(e),
                'response_text': error_message,
                'audio_response': self.tts.text_to_speech(error_message)
            }
    
    def process_text_input(self, session_id: str, user_text: str) -> Dict:
        """Process text input and return response"""
        try:
            # Recognize intent
            intent, confidence, entities = self.intent_recognizer.recognize_intent(user_text)
            
            # Add to memory
            self.memory.add_message(session_id, 'user', user_text)
            self.memory.add_intent(session_id, intent, confidence, entities)
            
            # Update user profile with extracted entities
            if entities:
                self.memory.update_user_profile(session_id, entities)
            
            # Get conversation context
            context = self.memory.get_context_for_llm(session_id)
            conversation_history = context['conversation_history']
            
            # Check if we should escalate to human
            should_escalate = self.intent_recognizer.should_escalate_to_human(
                intent, confidence, conversation_history
            )
            
            if should_escalate:
                response_text = self.intent_recognizer.generate_escalation_message(intent, entities)
            else:
                # Get relevant context from RAG
                rag_context = self.rag.get_context(user_text) if self.rag.documents else ""
                
                # Generate response using LLM
                response_text = self.llm.generate_response(
                    user_message=user_text,
                    context=rag_context,
                    conversation_history=conversation_history
                )
            
            # Add response to memory
            self.memory.add_message(session_id, 'assistant', response_text)
            
            # Generate audio response
            audio_response = self.tts.text_to_speech(response_text)
            
            print(f"Assistant: {response_text}")
            
            return {
                'success': True,
                'user_text': user_text,
                'response_text': response_text,
                'audio_response': audio_response,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'should_escalate': should_escalate,
                'session_id': session_id
            }
            
        except Exception as e:
            error_message = "I apologize, but I'm experiencing some technical difficulties. Please try again."
            return {
                'success': False,
                'error': str(e),
                'response_text': error_message,
                'audio_response': self.tts.text_to_speech(error_message)
            }
    
    def start_new_session(self) -> str:
        """Start a new conversation session"""
        session_id = str(uuid.uuid4())
        self.memory.create_session(session_id)
        return session_id
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        return {
            'session_context': self.memory.get_context_for_llm(session_id),
            'user_profile': self.memory.get_user_profile(session_id),
            'conversation_history': self.memory.get_formatted_history(session_id)
        }
    
    def end_session(self, session_id: str):
        """End a conversation session"""
        self.memory.clear_session(session_id)
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return self.rag.get_statistics()
    
    def save_knowledge_base(self, path: str = None):
        """Save the knowledge base index"""
        if path is None:
            path = f"{Config.EMBEDDINGS_DIR}/knowledge_base"
        self.rag.save_index(path)
    
    def load_knowledge_base(self, path: str = None):
        """Load a previously saved knowledge base"""
        if path is None:
            path = f"{Config.EMBEDDINGS_DIR}/knowledge_base"
        return self.rag.load_index(path)