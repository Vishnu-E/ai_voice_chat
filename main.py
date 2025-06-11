#!/usr/bin/env python3
"""
Voice Assistant CLI Application
Run this for command-line voice interaction
"""

import os
import sys
from config import Config
from src.voice_assistant import VoiceAssistant

def main():
    print("🎤 Voice Assistant CLI")
    print("=" * 40)
    
    try:
        # Validate configuration
        Config.validate_config()
        print("✅ Configuration validated")
        
        # Initialize voice assistant
        assistant = VoiceAssistant()
        
        # Create session
        session_id = assistant.create_session()
        print(f"📱 Session created: {session_id[:8]}...")
        
        # Setup knowledge base (optional)
        setup_knowledge_base(assistant)
        
        print("\n🎯 Voice Assistant Ready!")
        print("Commands:")
        print("  - Speak naturally after the prompt")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'info' to see session information")
        print("=" * 40)
        
        # Main conversation loop
        while True:
            try:
                print("\n🎤 Listening... (or type your message)")
                
                # Get user input (voice or text)
                user_input = input("You (or press Enter to use microphone): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'info':
                    show_session_info(assistant, session_id)
                    continue
                
                # Process input
                if user_input:
                    # Text input
                    result = assistant.process_text_input(session_id, user_input)
                else:
                    # Voice input
                    print("🎤 Speak now...")
                    result = assistant.process_voice_input(session_id)
                
                # Display result
                if result['success']:
                    print(f"\n🤖 Assistant: {result['response_text']}")
                    print(f"🎯 Intent: {result.get('intent', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
                    
                    # Play audio response
                    if result['response_audio']:
                        print("🔊 Playing audio response...")
                        assistant.tts.play_audio(result['response_audio'])
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                    print(f"🤖 Assistant: {result['response_text']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Failed to start voice assistant: {e}")
        print("\n💡 Make sure you have:")
        print("  1. Created a .env file with your API keys")
        print("  2. Installed all requirements: pip install -r requirements.txt")
        print("  3. Set up your microphone permissions")
        sys.exit(1)

def setup_knowledge_base(assistant):
    """Setup knowledge base with sample data"""
    print("\n📚 Setting up knowledge base...")
    
    # Ask user for documents
    while True:
        doc_type = input("Add document? (url/pdf/skip): ").strip().lower()
        
        if doc_type == 'skip':
            break
        elif doc_type == 'url':
            url = input("Enter URL: ").strip()
            if url:
                assistant.setup_knowledge_base(urls=[url])
        elif doc_type == 'pdf':
            pdf_path = input("Enter PDF path: ").strip()
            if pdf_path and os.path.exists(pdf_path):
                assistant.setup_knowledge_base(pdf_paths=[pdf_path])
            else:
                print("❌ PDF file not found")
        else:
            print("Invalid option. Use 'url', 'pdf', or 'skip'")

def show_session_info(assistant, session_id):
    """Display session information"""
    info = assistant.get_session_info(session_id)
    print("\n📊 Session Information:")
    print(f"  Session ID: {info['session_id'][:8]}...")
    print(f"  Messages: {info['conversation_length']}")
    print(f"  Duration: {info['session_duration']}")
    print(f"  User Profile: {info['user_profile']}")
    print(f"  Recent Intents: {[intent['intent'] for intent in info['recent_intents']]}")

if __name__ == "__main__":
    main()