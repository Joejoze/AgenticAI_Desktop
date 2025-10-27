## Email Agent (Local + Streamlit UI)

AI assistant that reads your Gmail, classifies emails, drafts replies with Groq LLM, and can send responses. Works via CLI or Streamlit UI.

### Features
- Gmail OAuth (local desktop flow)
- Email triage: spam, important, normal, reply_needed
- Draft replies using Groq LLM via LangChain
- Send replies from the UI
- Persistent episodic memory (`episodic_memory.json`)

### Requirements
- Python 3.9.10
- A Google Cloud project with Gmail API enabled
- OAuth client (Desktop) JSON saved as `credentials.json` in the project root
- A Groq API key (`GROQ_API_KEY`)

### Clone
```bash
git clone <your-repo-url> EmailAgentTest
cd EmailAgentTest
```

### 1) Create venv and install
```bash
# Optional if you use pyenv
pyenv local 3.9.10

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure environment and secrets
Create `.env` in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Place your Google OAuth desktop client JSON as `credentials.json` in the project root.

These files are intentionally git-ignored:
- `.env`
- `credentials.json`
- `token.json` (created automatically after first OAuth)
- `episodic_memory.json` (runtime memory)

### 3) First-time Gmail auth
On first run (CLI or UI), your browser will open for Google consent. After you approve, `token.json` will be saved for future runs.

### 4) Run (CLI)
```bash
python agent.py
```

### 5) Run (Streamlit UI)
```bash
streamlit run app_streamlit.py
```
In the UI:
- Sidebar → "Connect to Gmail" to complete OAuth if needed
- Click "Load Emails"
- Use "Classify" to triage, and optionally send replies when needed

### Troubleshooting
- Blank UI page: Hard refresh the browser. Ensure you clicked "Connect to Gmail" then "Load Emails".
- Module not found (e.g., blinker): install missing dep
  ```bash
  pip install blinker watchdog
  ```
- SSL errors on macOS with newer Python: use Python 3.9.10 venv (recommended).
- Large wheel stalls (pyarrow): networks can delay Streamlit deps; wait or pre-install `pyarrow<22`.
  ```bash
  pip install 'pyarrow<22'
  ```

### Security notes
- Keep `credentials.json` and `.env` out of version control (already in `.gitignore`).
- `token.json` grants Gmail access; treat it like a secret.

### Project structure
```
agent.py                # LLM logic, email classification, pipeline
emailapi.py             # Gmail OAuth + API client + send helper
app_streamlit.py        # Streamlit UI
requirements.txt        # Python deps (unpinned)
README.md               # This file
episodic_memory.json    # Runtime memory (created on run, git-ignored)
credentials.json        # Your Google OAuth credentials (git-ignored)
token.json              # OAuth token (created on run, git-ignored)
.env                    # GROQ_API_KEY (git-ignored)
```

### Customization
- Edit `semantic_memory` and `procedural_memory` in `agent.py` to match your style.
- Change the LLM model in `agent.py` (ChatGroq `model` field) if desired.
