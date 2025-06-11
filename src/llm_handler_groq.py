from groq import Groq
from typing import Dict, List, Optional
from config import Config
import json

class LLMHandler:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.LLM_MODEL
        self.max_tokens = 500
        self.temperature = 0.7

    def load_system_prompt(self, prompt_file: str = None) -> str:
        if prompt_file is None:
            prompt_file = f"{Config.PROMPTS_DIR}/assistant_prompt.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return self.get_default_prompt()

    def get_default_prompt(self) -> str:
        return ("""You are a helpful AI voice assistant. You have access to company documentation and can answer questions about products, services, and general topics. 

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
- Provide actionable information when possible""")

    def generate_response(self, user_message: str, context: str = "", conversation_history: List[Dict] = None, system_prompt: str = None) -> str:
        try:
            if system_prompt is None:
                system_prompt = self.load_system_prompt()

            messages = [{"role": "system", "content": system_prompt}]

            if context:
                context_message = f"Relevant context from documentation:\n{context}"
                messages.append({"role": "system", "content": context_message})

            if conversation_history:
                messages.extend(conversation_history[-6:])

            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
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
        try:
            conv_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history)

            messages = [
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences, focusing on key topics and user needs."},
                {"role": "user", "content": conv_text}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return "Conversation summary unavailable."

    def extract_entities(self, text: str) -> Dict:
        try:
            prompt = f"""Extract the following information from the text if present:
- Name (person's name)
- Email (email address)
- Phone (phone number)
- Company (company name)
- Intent (what the person wants to do)

Return as JSON format. If information is not found, use null.

Text: {text}"""

            messages = [
                {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )

            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return {}

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}

    def check_response_appropriateness(self, response: str) -> bool:
        try:
            messages = [
                {"role": "system", "content": "Analyze if this response is appropriate, helpful, and safe for a customer service context. Reply with 'YES' or 'NO' only."},
                {"role": "user", "content": response}
            ]

            check_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )

            return check_response.choices[0].message.content.strip().upper() == "YES"

        except Exception as e:
            print(f"Error checking response: {e}")
            return True
