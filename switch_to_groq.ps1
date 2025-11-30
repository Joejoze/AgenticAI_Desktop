# PowerShell script to switch to Groq
Write-Host "Setting JARVIS to use Groq..." -ForegroundColor Green

$env:JARVIS_MODEL = "groq"

Write-Host ""
Write-Host "[SUCCESS] JARVIS configured to use Groq Llama 3.1 8B" -ForegroundColor Green
Write-Host "[INFO] Starting JARVIS with Groq..." -ForegroundColor Yellow
Write-Host ""

# Start JARVIS
venv\Scripts\python jarvis_streamlit_sessions.py

