import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import Config

class MemoryManager:
    def __init__(self):
        self.sessions = {}
        self.max_history = Config.MAX_CONVERSATION_HISTORY
        self.session_timeout = Config.SESSION_TIMEOUT
    
    def create_session(self, session_id: str) -> Dict:
        """Create a new conversation session"""
        session = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'conversation_history': [],
            'user_profile': {},
            'context_summary': "",
            'intent_history': []
        }
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        last_activity = datetime.fromisoformat(session['last_activity'])
        if datetime.now() - last_activity > timedelta(seconds=self.session_timeout):
            # Session expired, create new one
            return self.create_session(session_id)
        
        return session
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp"""
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = datetime.now().isoformat()
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history"""
        session = self.get_session(session_id)
        
        message = {
            'role': role,  # 'user' or 'assistant'
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        session['conversation_history'].append(message)
        
        # Limit conversation history length
        if len(session['conversation_history']) > self.max_history * 2:  # *2 for user+assistant pairs
            # Remove oldest messages but keep context summary
            removed_messages = session['conversation_history'][:2]  # Remove oldest pair
            session['conversation_history'] = session['conversation_history'][2:]
            
            # Update context summary with removed messages
            self._update_context_summary(session, removed_messages)
        
        self.update_session_activity(session_id)
    
    def get_conversation_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        history = session['conversation_history']
        
        if limit:
            return history[-limit:]
        return history
    
    def get_formatted_history(self, session_id: str, limit: int = None) -> str:
        """Get formatted conversation history as string"""
        history = self.get_conversation_history(session_id, limit)
        
        formatted_messages = []
        for msg in history:
            role = msg['role'].capitalize()
            content = msg['content']
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M')
            formatted_messages.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(formatted_messages)
    
    def update_user_profile(self, session_id: str, profile_data: Dict):
        """Update user profile information"""
        session = self.get_session(session_id)
        session['user_profile'].update(profile_data)
        self.update_session_activity(session_id)
    
    def get_user_profile(self, session_id: str) -> Dict:
        """Get user profile information"""
        session = self.get_session(session_id)
        return session.get('user_profile', {})
    
    def add_intent(self, session_id: str, intent: str, confidence: float, entities: Dict = None):
        """Add detected intent to history"""
        session = self.get_session(session_id)
        
        intent_record = {
            'intent': intent,
            'confidence': confidence,
            'entities': entities or {},
            'timestamp': datetime.now().isoformat()
        }
        
        session['intent_history'].append(intent_record)
        
        # Keep only last 20 intents
        if len(session['intent_history']) > 20:
            session['intent_history'] = session['intent_history'][-20:]
        
        self.update_session_activity(session_id)
    
    def get_recent_intents(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent intents for a session"""
        session = self.get_session(session_id)
        return session['intent_history'][-limit:]
    
    def _update_context_summary(self, session: Dict, removed_messages: List[Dict]):
        """Update context summary when removing old messages"""
        # Simple summarization - in production, you might use an LLM for this
        summary_parts = []
        
        for msg in removed_messages:
            if msg['role'] == 'user':
                summary_parts.append(f"User asked about: {msg['content'][:100]}...")
            else:
                summary_parts.append(f"Assistant responded about: {msg['content'][:100]}...")
        
        new_summary = " | ".join(summary_parts)
        
        if session['context_summary']:
            session['context_summary'] += f" | {new_summary}"
        else:
            session['context_summary'] = new_summary
        
        # Limit summary length
        if len(session['context_summary']) > 1000:
            session['context_summary'] = session['context_summary'][-800:]
    
    def get_context_for_llm(self, session_id: str) -> Dict:
        """Get context information formatted for LLM"""
        session = self.get_session(session_id)
        
        # Get recent conversation
        recent_history = self.get_conversation_history(session_id, limit=10)
        user_profile = self.get_user_profile(session_id)
        recent_intents = self.get_recent_intents(session_id, limit=3)
        
        return {
            'conversation_history': recent_history,
            'user_profile': user_profile,
            'context_summary': session.get('context_summary', ''),
            'recent_intents': recent_intents,
            'session_duration': self._get_session_duration(session)
        }
    
    def _get_session_duration(self, session: Dict) -> str:
        """Calculate session duration"""
        start_time = datetime.fromisoformat(session['created_at'])
        duration = datetime.now() - start_time
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def clear_session(self, session_id: str):
        """Clear/delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_all_sessions(self) -> Dict:
        """Get all active sessions (for admin purposes)"""
        return {
            session_id: {
                'created_at': session['created_at'],
                'last_activity': session['last_activity'],
                'message_count': len(session['conversation_history']),
                'user_profile': session['user_profile']
            }
            for session_id, session in self.sessions.items()
        }