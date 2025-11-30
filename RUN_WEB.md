# How to Run JARVIS Web Interface

## Quick Start

### Option 1: Using the Launcher Script (Recommended)

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run the launcher:**
   ```bash
   python run_jarvis_streamlit.py
   ```

### Option 2: Direct Streamlit Command

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run Streamlit directly:**
   ```bash
   streamlit run jarvis_streamlit.py
   ```

   Or with custom port:
   ```bash
   streamlit run jarvis_streamlit.py --server.port 8501
   ```

### Option 3: Sessions Version (Multi-user support)

If you want the sessions version with chat history:

```bash
streamlit run jarvis_streamlit_sessions.py
```

## Prerequisites

Before running, make sure you have:

1. **Virtual environment activated** (see above)
2. **Environment variables set:**
   - Create a `.env` file in the project root
   - Add: `GROQ_API_KEY=your_api_key_here`
3. **Gmail credentials** (optional, for email features):
   - `credentials.json` file in project root
   - See `JARVIS_CONFIG_GUIDE.md` for setup instructions

## Access the Web Interface

Once running, the web interface will automatically open in your browser at:
- **URL:** http://localhost:8501
- If it doesn't open automatically, manually navigate to the URL

## Troubleshooting

### Port Already in Use
If port 8501 is busy, use a different port:
```bash
streamlit run jarvis_streamlit.py --server.port 8502
```

### Missing Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
Make sure you see `(venv)` in your terminal prompt before running.

## Features Available in Web Interface

- 🤖 Chat with JARVIS
- 📧 Gmail integration (read, send, reply to emails)
- 💻 System commands
- 📁 File operations
- 🎤 Voice mode (if enabled)
- 💾 Chat history (sessions version)

## Stop the Server

Press `Ctrl+C` in the terminal to stop the Streamlit server.

