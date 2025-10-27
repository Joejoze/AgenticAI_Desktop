# 🤖 JARVIS AI Assistant - Configuration Guide

## 🔧 Environment Setup

Create a `.env` file in your project root with the following configuration:

```env
# JARVIS AI Assistant Configuration
# =================================

# AI Model Configuration (REQUIRED)
GROQ_API_KEY=your_groq_api_key_here

# Google Services Configuration (REQUIRED for search functionality)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=91c4eabf9ed3b4411

# Optional: Telegram Bot (if you want Telegram integration)
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional: WhatsApp Integration (if you want WhatsApp integration)
# WHATSAPP_SESSION_PATH=./whatsapp_session

# Optional: Voice Components (if you want voice recognition)
# VOICE_ENABLED=true

# Optional: System Monitoring
# SYSTEM_MONITORING_ENABLED=true

# Optional: File Operations
# FILE_OPERATIONS_ENABLED=true

# Optional: Email Operations
# EMAIL_OPERATIONS_ENABLED=true
```

## 🔑 API Keys Setup

### 1. GROQ API Key (Required)
- Go to [https://console.groq.com/](https://console.groq.com/)
- Create an account and get your API key
- Add it to your `.env` file as `GROQ_API_KEY`

### 2. Google API Key (Required for Search)
- Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
- Create a new project or select existing one
- Enable the "Custom Search API"
- Create credentials (API Key)
- Add it to your `.env` file as `GOOGLE_API_KEY`

### 3. Google Custom Search Engine ID
- You already have this: `91c4eabf9ed3b4411`
- Add it to your `.env` file as `GOOGLE_CSE_ID`

## 🚀 Quick Start

1. **Create your `.env` file:**
   ```bash
   # Copy the template above into a new .env file
   ```

2. **Add your API keys:**
   - Get GROQ API key from [console.groq.com](https://console.groq.com/)
   - Get Google API key from [console.cloud.google.com](https://console.cloud.google.com/)
   - Use the CSE ID: `91c4eabf9ed3b4411`

3. **Launch JARVIS:**
   ```bash
   # Activate virtual environment
   venv\Scripts\activate
   
   # Launch JARVIS
   python run_jarvis_streamlit.py
   ```

## 🎯 Features Available

### ✅ Core Features (Always Available)
- 💬 **Chat Interface** - Natural language conversation
- 🧠 **AI Memory** - Remembers conversation context
- 📧 **Gmail Integration** - Check and send emails
- 🔍 **Google Search** - Real-time web search
- 💻 **System Monitoring** - CPU, memory, processes
- 📁 **File Operations** - Read, write, search files

### ⚠️ Optional Features (Install if needed)
- 🎤 **Voice Recognition** - Install: `pip install SpeechRecognition pyttsx3 pyaudio`
- 📱 **Telegram Bot** - Install: `pip install pyTelegramBotAPI`
- 💬 **WhatsApp Integration** - Install: `pip install selenium`

## 🔧 Troubleshooting

### "GROQ_API_KEY not set"
- Make sure you have a `.env` file in the project root
- Add your GROQ API key to the `.env` file

### "Google search not working"
- Make sure you have `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in your `.env` file
- Verify your Google API key has Custom Search API enabled

### "Gmail not connecting"
- Make sure you have `credentials.json` in the project root
- Follow the Gmail setup instructions in the main README

## 🎉 Ready to Use!

Once you have your `.env` file configured with the required API keys, JARVIS will:

1. **Auto-initialize** when you open the app
2. **Connect to Gmail** automatically (if credentials are available)
3. **Enable Google search** with your custom search engine
4. **Start chatting immediately** - no manual setup required!

**Your JARVIS AI Assistant is ready! 🤖✨**
