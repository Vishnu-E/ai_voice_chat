# Voice Assistant Project

A comprehensive voice assistant application with speech-to-text, text-to-speech, LLM integration, and RAG (Retrieval-Augmented Generation) capabilities. The project features a modern Streamlit frontend for easy interaction and deployment.

## 🚀 Features

- **Speech Recognition**: Convert speech to text using advanced STT models
- **Text-to-Speech**: Natural voice synthesis for assistant responses
- **LLM Integration**: Powered by large language models for intelligent conversations
- **RAG Engine**: Document retrieval and augmented generation for context-aware responses
- **Memory Management**: Conversation history and context retention
- **Intent Recognition**: Smart understanding of user intentions
- **Streamlit Frontend**: Modern, responsive web interface
- **Flask API**: RESTful backend services

## 📁 Project Structure

```
voice_assistant_project/
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
├── main.py                   # Main application entry point
├── app.py                    # Flask web server
├── templates/
│   └── index.html           # HTML templates
├── src/
│   ├── __init__.py
│   ├── speech_to_text.py    # STT functionality
│   ├── text_to_speech.py    # TTS functionality
│   ├── llm_handler.py       # LLM integration
│   ├── rag_engine.py        # RAG implementation
│   ├── memory_manager.py    # Conversation memory
│   ├── intent_recognizer.py # Intent classification
│   └── voice_assistant.py   # Core assistant logic
├── data/
│   ├── documents/           # Knowledge base documents
│   └── embeddings/          # Vector embeddings storage
├── prompts/
│   └── assistant_prompt.txt # System prompts
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice_assistant_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**
   - Update `config.py` with your API keys and settings
   - Add your documents to the `data/documents/` folder
   - Customize prompts in `prompts/assistant_prompt.txt`

## 🚀 Usage

### Running with Streamlit (Recommended)

```bash
streamlit run main.py
```

The Streamlit interface will be available at `http://localhost:8501`

### Running with Flask

```bash
python app.py
```

The Flask API will be available at `http://localhost:5000`

## 📋 Configuration

Edit `config.py` to configure:

- **API Keys**: Set your LLM provider API keys
- **Speech Models**: Configure STT/TTS model preferences
- **RAG Settings**: Adjust embedding models and retrieval parameters
- **Memory Settings**: Configure conversation history limits
- **Server Settings**: Set ports and host configurations

## 🎯 Core Components

### Speech-to-Text (`speech_to_text.py`)
Handles audio input processing and conversion to text using various STT engines.

### Text-to-Speech (`text_to_speech.py`)
Converts assistant responses to natural-sounding speech output.

### LLM Handler (`llm_handler.py`)
Manages integration with large language models for generating intelligent responses.

### RAG Engine (`rag_engine.py`)
Implements retrieval-augmented generation for context-aware responses using document knowledge base.

### Memory Manager (`memory_manager.py`)
Handles conversation history, context retention, and memory optimization.

### Intent Recognizer (`intent_recognizer.py`)
Classifies user inputs and determines appropriate response strategies.

### Voice Assistant (`voice_assistant.py`)
Core orchestration logic that coordinates all components.

## 📚 Adding Documents

1. Place your documents in the `data/documents/` folder
2. Supported formats: PDF, TXT, DOCX, MD
3. Run the embedding generation process to update the knowledge base
4. Documents will be automatically indexed for RAG functionality

## 🔧 API Endpoints (Flask)

- `POST /api/chat` - Send text message to assistant
- `POST /api/voice` - Send audio for speech processing
- `GET /api/status` - Check system status
- `POST /api/reset` - Reset conversation memory

## 🎨 Streamlit Interface

The Streamlit frontend provides:

- **Voice Input**: Click-to-record functionality
- **Text Chat**: Traditional text-based interaction
- **Settings Panel**: Adjust voice and model parameters
- **Document Upload**: Add new documents to knowledge base
- **Conversation History**: View and manage chat history

## 🔍 Troubleshooting

### Common Issues

1. **Audio not working**: Check microphone permissions and audio device settings
2. **Slow responses**: Verify API keys and internet connection
3. **Import errors**: Ensure all dependencies are installed correctly
4. **Memory issues**: Adjust memory limits in configuration

### Logs

Check application logs for detailed error information:
- Streamlit logs appear in the terminal
- Flask logs are available in the console output

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for LLM capabilities
- Streamlit for the excellent web framework
- Various open-source STT/TTS libraries
- The Python AI/ML community

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration documentation

---

**Happy Coding! 🎉**
