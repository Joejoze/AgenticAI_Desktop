@echo off
echo Setting JARVIS to use Gemini...
set JARVIS_MODEL=gemini
set GEMINI_API_KEY=AIzaSyCvwrnSNHFONWicTswcE3ZzGX-9fw3E26o
echo.
echo [SUCCESS] JARVIS configured to use Gemini 2.5 Flash
echo [INFO] Run: venv\Scripts\python jarvis_streamlit_sessions.py
echo.
echo Starting JARVIS with Gemini...
venv\Scripts\python jarvis_streamlit_sessions.py
