@echo off
echo Setting JARVIS to use Groq...
set JARVIS_MODEL=groq
echo.
echo [SUCCESS] JARVIS configured to use Groq Llama 3.1 8B
echo [INFO] Run: venv\Scripts\python jarvis_streamlit_sessions.py
echo.
echo Starting JARVIS with Groq...
venv\Scripts\python jarvis_streamlit_sessions.py
