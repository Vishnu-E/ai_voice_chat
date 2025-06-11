import uuid
import time
from typing import Dict, Optional, Tuple
from src.speech_to_text import SpeechToText
from src.text_to_speech import TextToSpeech
from src.llm_handler_groq import LLMHandler
from src.rag_engine import RAGEngine
from src.memory_manager import MemoryManager
from src.intent_recognizer import IntentRecognizer
from config import Config

class VoiceAssistant:
    def __init__(self):
        # Initialize all components
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.llm = LLMHandler()
        self.rag = RAGEngine()
        self.memory = MemoryManager()
        self.intent_recognizer = IntentRecognizer()
        
        # Assistant state
        self.is_listening = False
        self.knowledge_base_loaded = False
        
        print("Voice Assistant initialized successfully!")
    
    def load_knowledge_base(self, sources: list):
        """Load knowledge base from URLs or PDF files"""
        print("Loading knowledge base...")
        
        for source in sources:
            if source.startswith('http'):
                self.rag.add_url(source)
            elif source.endswith('.pdf'):
                self.rag.add_pdf(source)
            else:
                print(f"Unsupported source format: {source}")
        
        if self.rag.documents:
            self.rag.build_index()
            self.knowledge_base_loaded = True
            print(f"Knowledge base loaded with {len(self.rag.documents)} documents")
            
            # Print statistics
            stats = self.rag.get_statistics()
            print(f"Total sources: {stats['unique_sources']}")
            print(f"Total words: {stats['total_words']}")
        else:
            print("No documents loaded into knowledge base")
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.memory.create_session(session_id)
        return session_id
    
    def process_voice_input(self, session_id: str, audio_data: bytes = None) -> Dict:
        """Process voice input and return response"""
        try:
            # Step 1: Convert speech to text
            if audio_data:
                user_text = self.stt.transcribe_audio_data(audio_data)
            else:
                user_text = self.stt.listen_from_microphone()
            
            if not user_text:
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'response_text': 'I didn\'t catch that. Could you please repeat?',
                    'audio_data': None
                }
            
            print(f"User said: {user_text}")
            
            # Process the text input
            result = self.process_text_input(session_id, user_text)
            
            # Generate audio response
            if result['success'] and result['response_text']:
                audio_data = self.tts.text_to_speech(result['response_text'])
                result['audio_data'] = audio_data
            
            return result
            
        except Exception as e:
            print(f"Error processing voice input: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_text': 'I encountered an error processing your request.',
                'audio_data': None
            }
    
    def process_text_input(self, session_id: str, user_text: str) -> Dict:
        """Process text input and return response"""
        try:
            # Step 1: Store user message
            self.memory.add_message(session_id, 'user', user_text)
            
            # Step 2: Recognize intent
            intent, confidence, entities = self.intent_recognizer.recognize_intent(user_text)
            print(f"Detected intent: {intent} (confidence: {confidence:.2f})")
            
            # Step 3: Store intent
            self.memory.add_intent(session_id, intent, confidence, entities)
            
            # Step 4: Check for special handling
            if self.intent_recognizer.needs_human_handoff(intent, confidence):
                response = self._handle_human_handoff(session_id, intent, entities)
            elif self.intent_recognizer.is_scheduling_intent(intent, confidence, entities):
                response = self._handle_scheduling(session_id, entities)
            else:
                response = self._generate_contextual_response(session_id, user_text, intent, entities)
            
            # Step 5: Store assistant response
            self.memory.add_message(session_id, 'assistant', response)
            
            return {
                'success': True,
                'response_text': response,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'session_id': session_id
            }
            
        except Exception as e:
            print(f"Error processing text input: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_text': 'I encountered an error processing your request.',
                'intent': 'error',
                'confidence': 0.0,
                'entities': {}
            }
    
    def _generate_contextual_response(self, session_id: str, user_text: str, intent: str, entities: Dict) -> str:
        """Generate contextual response using RAG and conversation history"""
        
        # Get relevant context from knowledge base
        context = ""
        if self.knowledge_base_loaded:
            context = self.rag.get_context(user_text)
        
        # Get conversation context
        memory_context = self.memory.get_context_for_llm(session_id)
        
        # Generate response
        response = self.llm.generate_response(
            user_message=user_text,
            context=context,
            conversation_history=memory_context['conversation_history'],
            user_profile=memory_context['user_profile']
        )
        
        # Update user profile if entities contain personal info
        if entities.get('name') or entities.get('email') or entities.get('phone'):
            self.memory.update_user_profile(session_id, entities)
        
        return response
    
    def _handle_scheduling(self, session_id: str, entities: Dict) -> str:
        """Handle scheduling requests"""
        user_profile = self.memory.get_user_profile(session_id)
        
        missing_info = []
        if not entities.get('name') and not user_profile.get('name'):
            missing_info.append('your name')
        if not entities.get('email') and not user_profile.get('email'):
            missing_info.append('your email address')
        
        if missing_info:
            return f"I'd be happy to help you schedule a call! To proceed, I'll need {' and '.join(missing_info)}. Could you please provide that information?"
        
        # If we have all info, confirm scheduling
        name = entities.get('name') or user_profile.get('name')
        email = entities.get('email') or user_profile.get('email')
        
        return f"Perfect! I have your information - {name} ({email}). I'll arrange for someone from our team to contact you within 24 hours to schedule a convenient time for your call. Is there anything specific you'd like to discuss during the call?"
    
    def _handle_human_handoff(self, session_id: str, intent: str, entities: Dict) -> str:
        """Handle requests for human assistance"""
        
        if intent == 'complaint':
            return "I understand your concern and want to make sure you get the best assistance. I'm connecting you with a human representative who can help resolve this issue properly. Please hold on while I transfer you."
        
        elif intent == 'contact_human':
            return "Of course! I'm connecting you with a human representative right now. They'll be able to provide more personalized assistance. Please wait just a moment."
        
        else:
            return "I want to make sure you get the best help possible. Let me connect you with one of our human team members who can assist you further. They'll be with you shortly."
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of conversation session"""
        context = self.memory.get_context_for_llm(session_id)
        recent_intents = self.memory.get_recent_intents(session_id)
        
        return {
            'session_id': session_id,
            'message_count': context['message_count'],
            'session_duration': context['session_duration'],
            'user_profile': context['user_profile'],
            'recent_intents': [intent['intent'] for intent in recent_intents],
            'conversation_summary': self.llm.summarize_conversation(context['conversation_history'])

        }
    
