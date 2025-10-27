# =============================
# FINAL AI EMAIL ASSISTANT PIPELINE
# =============================

# For local: install dependencies via requirements.txt

import os, json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# =============================
# LANGGRAPH MEMORY SYSTEM
# =============================
from simple_langgraph_memory import SimpleLangGraphMemorySystem, create_simple_memory_workflow

# =============================
# 1. Setup LLM
# =============================
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set. Create a .env with GROQ_API_KEY=... or export it.")
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant"
)

# =============================
# 2. Semantic + Procedural Memory
# =============================
semantic_memory = {
    "user_name": "Rostan Lobo",
    "role": "Student",
    "preferences": ["short replies", "ignore spam"]
}

procedural_memory = """
You are an AI Email Assistant.
- Classify emails into: spam, important, normal, or reply_needed.
- Spam = junk emails such as lotteries, phishing, promotions, or generic unsolicited advertisements.
- Important = work-related or deadlines (mentions of boss, projects, or tasks) but no direct reply required.
- Normal = casual or friendly messages (coffee invites, greetings, jokes) without urgency or expectation.
- Reply_needed = emails explicitly asking for response or action (e.g., “Are you available?”, “Please confirm”, “Send file”).
- For forwarded emails, analyze the content, not just the subject line.
- If subject or body mentions "urgent", "deadline", "meeting", or "boss", treat it as important or reply_needed depending on action requested.
- If email requests a document/file, do not make one up; notify user instead.
- Emails from known spam sources (Glassdoor, Pinterest, Zolve, etc.) are spam unless context shows action required.
- Always respect user’s preferences from semantic memory.
"""


# =============================
# 3. Episodic Memory (Persistent)
# =============================
MEMORY_FILE = "episodic_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

episodic_memory = load_memory()

# =============================
# LANGGRAPH MEMORY SYSTEM INITIALIZATION
# =============================
langgraph_memory = SimpleLangGraphMemorySystem()
langgraph_workflow = create_simple_memory_workflow(langgraph_memory)

# =============================
# 4. Triage Tool
# =============================
triage_prompt = PromptTemplate(
    input_variables=["from_", "subject", "body", "procedural", "semantic"],
    template="""
{procedural}

User facts: {semantic}

Classify the following email as one of: spam, important, casual, urgent.

Classification criteria:
- SPAM: Unsolicited commercial emails, phishing attempts, suspicious content
- IMPORTANT: Business communications, official notifications, work-related matters
- CASUAL: Personal messages, newsletters, social updates, non-urgent content
- URGENT: Time-sensitive matters, emergency notifications, critical business issues

Email:
From: {from_}
Subject: {subject}
Body: {body}

Answer with only one word.
"""
)

def classify_email(email):
    prompt = triage_prompt.format(
        from_=email["from"], subject=email["subject"], body=email["body"],
        procedural=procedural_memory, semantic=semantic_memory
    )
    result = llm.invoke(prompt).content.strip().lower()
    return result

# =============================
# 5. Writing Tool
# =============================
def writing_tool(email, classification, semantic_profile=None):
    default_profile = {
        "tone": "polite and professional",
        "signature": "Best regards,\nYour AI Email Assistant"
    }
    if semantic_profile is None:
        semantic_profile = default_profile
    else:
        # Merge defaults with provided profile; provided values take precedence if present
        merged = dict(default_profile)
        try:
            merged.update(semantic_profile)
        except Exception:
            # If a non-dict is passed inadvertently, fall back to defaults
            merged = default_profile
        semantic_profile = merged

    email_text = f"From: {email['from']}\nSubject: {email['subject']}\nBody: {email['body']}"

    # Spam
    if classification == "spam":
        return "SPAM: No reply needed. Marked as spam."

    # Urgent emails - high priority
    if classification == "urgent":
        return f"URGENT: {email_text}\n\nThis email requires immediate attention!"

    # Important emails - business/work related
    if classification == "important":
        return f"IMPORTANT: {email_text}\n\nThis email is marked as important for business/work matters."

    # Casual emails - personal/social
    if classification == "casual":
        return f"CASUAL: {email_text}\n\nThis is a casual/personal email."

    # Document requests (legacy support)
    if any(word in email['body'].lower() for word in ["document", "report", "attachment", "file"]):
        return f"DOCUMENT REQUEST: This email is asking for a document.\n\n{email_text}"

    # Normal casual
    if classification == "normal":
        return f"Notification: Casual message received.\n\n{email_text}\n\nSuggestion: You may want to reply, but it's not urgent."

    return "No action taken."

# =============================
# 6. Full Pipeline Function (Enhanced with LangGraph Memory)
# =============================
def process_email(email):
    """
    Enhanced process_email function with LangGraph memory capabilities
    """
    # Process through LangGraph workflow (includes memory context)
    result = langgraph_workflow(email)
    
    # Extract results
    classification = result['classification']
    action = result['action_taken']
    memory_id = result['memory_id']

    # Keep your existing episodic memory for backward compatibility
    entry = {
        "time": datetime.now().isoformat(),
        "from": email["from"],
        "subject": email["subject"],
        "classification": classification,
        "action_taken": action,
        "langgraph_memory_id": memory_id  # Add this for reference
    }

    episodic_memory.append(entry)
    save_memory(episodic_memory)

    return classification, action

def fetch_recent_emails(service, max_results=10):
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])
    fetched = []
    if not messages:
        return fetched
    for m in messages:
        msg = service.users().messages().get(userId="me", id=m["id"]).execute()
        headers = msg["payload"].get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(Unknown Sender)")
        snippet = msg.get("snippet", "")
        fetched.append({"from": sender, "subject": subject, "body": snippet})
    return fetched

if __name__ == "__main__":
    from emailapi import get_service
    
    print("🤖 AI Email Assistant with LangGraph Memory")
    print("=" * 50)
    
    emails = fetch_recent_emails(get_service(), max_results=10)
    if not emails:
        print("📭 No messages found.")
    else:
        print(f"📧 Found {len(emails)} emails to process...")
        for email in emails:
            classification, action = process_email(email)
            print("="*70)
            print(f"FROM: {email['from']}")
            print(f"SUBJECT: {email['subject']}")
            print(f"BODY: {email['body']}")
            print(f"CLASSIFICATION: {classification}")
            print("AI ACTION:", action)
            print()
    
    # Show memory statistics
    show_memory_stats()

    # print("\n=== Episodic Memory (Persisted) ===")
    # for i, entry in enumerate(episodic_memory, 1):
    #     print(f"{i}. {entry}")

# =============================
# LANGGRAPH MEMORY MANAGEMENT FUNCTIONS
# =============================
def get_memory_insights():
    """Get insights from the LangGraph memory system"""
    from simple_langgraph_memory import get_memory_insights
    return get_memory_insights(langgraph_memory)

def search_memories(query):
    """Search through stored memories"""
    # Simple search through memories
    results = []
    query_lower = query.lower()
    for memory in langgraph_memory.memories:
        if (query_lower in memory.subject.lower() or 
            query_lower in memory.content.lower() or 
            query_lower in memory.sender.lower()):
            results.append(memory)
    return results

def clear_memories():
    """Clear all stored memories (use with caution)"""
    langgraph_memory.memories = []
    langgraph_memory._save_memories()
    print("All memories cleared!")

def show_memory_stats():
    """Show memory statistics"""
    insights = get_memory_insights()
    print("\n🧠 LangGraph Memory System Stats:")
    print("=" * 40)
    print(f"Total emails processed: {insights['total_emails']}")
    print(f"Classification distribution: {insights['classification_distribution']}")
    print(f"Top senders: {insights['top_senders']}")
    print(f"Recent activity: {len(insights['recent_activity'])} recent emails")
