import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    #OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    
    # Voice Settings
    VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
    
    # Model Settings
    #LLM_MODEL = 'gpt-4'
    LLM_MODEL = 'mixtral-8x7b' 
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Audio Settings
    AUDIO_FORMAT = 'mp3'
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024
    
    # RAG Settings
    CHUNK_SIZE_RAG = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_LENGTH = 4000
    TOP_K_RESULTS = 5
    
    # Memory Settings
    MAX_CONVERSATION_HISTORY = 10
    SESSION_TIMEOUT = 1800  # 30 minutes
    
    # Intent Recognition
    INTENT_CONFIDENCE_THRESHOLD = 0.7
    
    # File Paths
    DATA_DIR = 'data'
    EMBEDDINGS_DIR = 'data/embeddings'
    DOCUMENTS_DIR = 'data/documents'
    PROMPTS_DIR = 'prompts'
    
    # Flask Settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    
    @classmethod
    def validate_config(cls):
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not cls.GROQ_API_KEY_API_KEY:
            missing_keys.append('GROQ_API_KEY')
        if not cls.ELEVENLABS_API_KEY:
            missing_keys.append('ELEVENLABS_API_KEY')
            
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True