import openai
from typing import Dict, List, Optional
from config import Config

class LLMHandler:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.model = Config.LLM_MODEL
        self.max_tokens = 500
        self.temperature = 0.7
        
    def load_system_prompt(self, prompt_file: str = None) -> str:
        """Load system prompt from file"""
        if prompt_file is None:
            prompt_file = f"{Config.PROMPTS_DIR}/assistant_prompt.txt"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return self.get_default_prompt()
    
    def get_default_prompt(self) -> str:
        """Default system prompt if file is not found"""
        return """You are a helpful AI voice assistant. You have access to company documentation and can answer questions about products, services, and general topics. 

Key guidelines:
- Be conversational and natural in your responses
- Keep responses concise but informative
- If you don't know something from the documentation, say so
- If someone seems ready to schedule a call or needs human help, offer to connect them
- Remember the conversation context
- Be friendly and professional

When responding:
- Use natural speech patterns
- Avoid overly technical jargon unless appropriate
- Ask clarifying questions when needed
- Provide actionable information when possible"""
    
    def generate_response(self, user_message: str, context: str = "", conversation_history: List[Dict] = None, system_prompt: str = None) -> str:
        """Generate response using OpenAI"""
        try:
            if system_prompt is None:
                system_prompt = self.load_system_prompt()
            
            # Build messages array
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add context if available
            if context:
                context_message = f"Relevant context from documentation:\n{context}"
                messages.append({"role": "system", "content": context_message})
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
    
    def summarize_conversation(self, conversation_history: List[Dict]) -> str:
        """Summarize conversation history"""
        try:
            # Prepare conversation text
            conv_text = ""
            for msg in conversation_history:
                role = msg["role"].capitalize()
                conv_text += f"{role}: {msg['content']}\n"
            
            messages = [
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences, focusing on key topics and user needs."},
                {"role": "user", "content": conv_text}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return "Conversation summary unavailable."
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities like names, emails, phone numbers from text"""
        try:
            prompt = """Extract the following information from the text if present:
- Name (person's name)
- Email (email address)
- Phone (phone number)
- Company (company name)
- Intent (what the person wants to do)

Return as JSON format. If information is not found, use null.

Text: {text}"""
            
            messages = [
                {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt.format(text=text)}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return {}
                
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}
    
    def check_response_appropriateness(self, response: str) -> bool:
        """Check if the response is appropriate and safe"""
        try:
            messages = [
                {"role": "system", "content": "Analyze if this response is appropriate, helpful, and safe for a customer service context. Reply with 'YES' or 'NO' only."},
                {"role": "user", "content": response}
            ]
            
            check_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )
            
            return check_response.choices[0].message.content.strip().upper() == "YES"
            
        except Exception as e:
            print(f"Error checking response: {e}")
            return True  # Default to allowing response if check fails