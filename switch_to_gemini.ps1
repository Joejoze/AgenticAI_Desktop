# PowerShell script to switch to Gemini 2.5 Flash
Write-Host "Setting JARVIS to use Gemini 2.5 Flash..." -ForegroundColor Green

$env:JARVIS_MODEL = "gemini"
$env:GEMINI_API_KEY = "AIzaSyCvwrnSNHFONWicTswcE3ZzGX-9fw3E26o"

Write-Host ""
Write-Host "[SUCCESS] JARVIS configured to use Gemini 2.5 Flash" -ForegroundColor Green
Write-Host "[INFO] Starting JARVIS with Gemini 2.5 Flash..." -ForegroundColor Yellow
Write-Host ""

# Start JARVIS
venv\Scripts\python jarvis_streamlit_sessions.py

