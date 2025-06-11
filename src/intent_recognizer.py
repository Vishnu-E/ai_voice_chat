import re
from typing import Dict, List, Tuple
from config import Config

class IntentRecognizer:
    def __init__(self):
        self.confidence_threshold = Config.INTENT_CONFIDENCE_THRESHOLD
        
        # Define intent patterns and keywords
        self.intent_patterns = {
            'schedule_call': {
                'keywords': [
                    'schedule', 'book', 'appointment', 'call', 'meeting', 'demo',
                    'talk to someone', 'speak with', 'consultation', 'set up',
                    'arrange', 'plan', 'calendar', 'available', 'free time'
                ],
                'phrases': [
                    'schedule a call', 'book a meeting', 'set up a demo',
                    'talk to sales', 'speak with someone', 'human agent',
                    'schedule demo', 'book appointment', 'arrange call'
                ]
            },
            'product_inquiry': {
                'keywords': [
                    'product', 'service', 'feature', 'pricing', 'cost', 'price',
                    'how much', 'what does', 'tell me about', 'information',
                    'details', 'specification', 'compare', 'difference'
                ],
                'phrases': [
                    'tell me about', 'how much does', 'what is the price',
                    'product information', 'service details', 'feature list'
                ]
            },
            'support_request': {
                'keywords': [
                    'help', 'problem', 'issue', 'error', 'bug', 'trouble',
                    'not working', 'broken', 'fix', 'support', 'assistance',
                    'troubleshoot', 'resolve', 'solution'
                ],
                'phrases': [
                    'need help', 'having trouble', 'not working', 'technical issue',
                    'customer support', 'help me with', 'problem with'
                ]
            },
            'general_question': {
                'keywords': [
                    'what', 'how', 'when', 'where', 'why', 'who', 'which',
                    'explain', 'describe', 'tell', 'show', 'guide', 'tutorial'
                ],
                'phrases': [
                    'how do i', 'what is', 'can you explain', 'tell me how',
                    'show me', 'guide me', 'how to'
                ]
            },
            'complaint': {
                'keywords': [
                    'disappointed', 'frustrated', 'angry', 'upset', 'complaint',
                    'terrible', 'awful', 'bad', 'worst', 'horrible', 'useless',
                    'waste', 'refund', 'cancel', 'unsatisfied'
                ],
                'phrases': [
                    'not happy', 'want refund', 'cancel subscription', 'poor service',
                    'bad experience', 'very disappointed'
                ]
            },
            'greeting': {
                'keywords': [
                    'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                    'greetings', 'howdy', 'what\'s up'
                ],
                'phrases': [
                    'good morning', 'good afternoon', 'good evening', 'how are you',
                    'nice to meet', 'hello there'
                ]
            },
            'goodbye': {
                'keywords': [
                    'bye', 'goodbye', 'see you', 'farewell', 'talk later',
                    'have a good', 'take care', 'thanks', 'thank you'
                ],
                'phrases': [
                    'talk to you later', 'have a good day', 'see you later',
                    'thanks for help', 'goodbye for now'
                ]
            }
        }
        
        # Contact information patterns
        self.contact_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'name': r'\b(?:my name is|i\'m|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        }
    
    def recognize_intent(self, text: str) -> Tuple[str, float, Dict]:
        """Recognize intent from text and return intent, confidence, and entities"""
        text_lower = text.lower()
        
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
            
            # Check phrases (higher weight)
            for phrase in patterns['phrases']:
                if phrase.lower() in text_lower:
                    score += 2
            
            # Normalize score by total possible matches
            total_possible = len(patterns['keywords']) + (len(patterns['phrases']) * 2)
            normalized_score = score / total_possible if total_possible > 0 else 0
            
            intent_scores[intent] = normalized_score
        
        # Find best intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_name, confidence = best_intent
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Return unknown intent if confidence is too low
        if confidence < 0.1:
            intent_name = "unknown"
            confidence = 0.0
        
        return intent_name, confidence, entities
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities like email, phone, name from text"""
        entities = {}
        
        # Extract email
        email_match = re.search(self.contact_patterns['email'], text)
        if email_match:
            entities['email'] = email_match.group()
        
        # Extract phone
        phone_match = re.search(self.contact_patterns['phone'], text)
        if phone_match:
            entities['phone'] = phone_match.group()
        
        # Extract name
        name_match = re.search(self.contact_patterns['name'], text, re.IGNORECASE)
        if name_match:
            entities['name'] = name_match.group(1)
        
        return entities
    
    def should_escalate_to_human(self, intent: str, confidence: float, conversation_history: List[Dict] = None) -> bool:
        """Determine if conversation should be escalated to human agent"""
        
        # Always escalate complaints
        if intent == 'complaint':
            return True
        
        # Escalate if user explicitly asks for human
        human_keywords = ['human', 'person', 'agent', 'representative', 'someone', 'staff', 'employee']
        last_message = conversation_history[-1]['content'].lower() if conversation_history else ""
        
        if any(keyword in last_message for keyword in human_keywords):
            return True
        
        # Escalate if intent recognition confidence is consistently low
        if conversation_history and len(conversation_history) >= 6:
            recent_messages = conversation_history[-3:]  # Last 3 user messages only
            user_messages = [msg for msg in recent_messages if msg['role'] == 'user']
            
            if len(user_messages) >= 2:
                # If we've had multiple low-confidence interactions, escalate
                return True
        
        # Escalate for scheduling requests with contact info
        if intent == 'schedule_call':
            return True
        
        return False
    
    def generate_escalation_message(self, intent: str, entities: Dict) -> str:
        """Generate appropriate message for human escalation"""
        
        if intent == 'complaint':
            return "I understand you're having concerns. Let me connect you with one of our specialists who can better assist you. Could you please provide your name and email so they can follow up with you?"
        
        elif intent == 'schedule_call':
            if entities.get('email') or entities.get('phone'):
                return f"I'd be happy to help you schedule a call. I have your {'email' if entities.get('email') else 'phone number'}. What's the best time for you, and would you like to speak with someone from sales or support?"
            else:
                return "I'd be happy to help you schedule a call with our team. Could you please provide your name and email address so we can set that up for you?"
        
        elif intent == 'support_request':
            return "For technical issues, our support team can provide the best assistance. Would you like me to connect you with a support specialist? Please share your contact information so they can help you directly."
        
        else:
            return "It sounds like you might benefit from speaking with one of our team members directly. Could you provide your name and email, and I'll have someone reach out to you?"
    
    def extract_scheduling_preferences(self, text: str) -> Dict:
        """Extract scheduling preferences from text"""
        preferences = {}
        text_lower = text.lower()
        
        # Time preferences
        time_patterns = {
            'morning': r'\b(morning|am|9|10|11)\b',
            'afternoon': r'\b(afternoon|pm|1|2|3|4)\b',
            'evening': r'\b(evening|5|6|7)\b'
        }
        
        for time_pref, pattern in time_patterns.items():
            if re.search(pattern, text_lower):
                preferences['time_preference'] = time_pref
                break
        
        # Day preferences
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in text_lower:
                preferences['day_preference'] = day
                break
        
        # Urgency
        urgent_keywords = ['urgent', 'asap', 'soon', 'immediately', 'today', 'tomorrow']
        if any(keyword in text_lower for keyword in urgent_keywords):
            preferences['urgency'] = 'high'
        
        return preferences