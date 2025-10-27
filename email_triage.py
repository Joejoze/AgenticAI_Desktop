import random
from transformers import pipeline
from langgraph.graph import StateGraph, END

# ------------------ Initialize Hugging Face Pipelines ------------------
print("Loading Hugging Face models... (first run may take time)")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("Models loaded successfully!")

# ------------------ Test Emails ------------------
test_emails = [
    # Urgent
    """🚨 URGENT: Server down in production! Immediate action required.
    Please investigate and resolve ASAP before client impact worsens.""",

    # Normal
    """Hi Alex, 
    Can we reschedule our weekly team meeting to Thursday afternoon?
    Let me know what works best for you.""",

    # Spam
    """🎁 Congratulations! You've won a $1000 Amazon gift card.
    CLICK HERE: http://fake-prizes.example.com to claim your reward now!!!""",

    # Newsletter
    """Hello subscriber, 
    Here’s our October Newsletter featuring product updates, 
    tips & tricks, and upcoming webinar schedules. 
    Thank you for staying with us!""",

    # Follow-up
    """Hi Sarah, 
    Just following up on my last email about the project proposal. 
    Please let me know if you had a chance to review it."""
]

# ------------------ Email State for LangGraph ------------------
class EmailState(dict):
    email_content: str
    category: str
    response: str

# ------------------ Helper Functions ------------------
def triage_email(state: EmailState):
    labels = [
        "urgent or critical email",
        "normal professional email",
        "spam or phishing email",
        "newsletter or marketing update",
        "follow-up or reminder email"
    ]
    result = classifier(state['email_content'], labels)
    raw_label = result['labels'][0].lower()
    print(f"[DEBUG] Classified as: {raw_label}")

    # Map descriptive labels to LangGraph keys
    label_map = {
        "urgent or critical email": "urgent",
        "normal professional email": "normal",
        "spam or phishing email": "spam",
        "newsletter or marketing update": "newsletter",
        "follow-up or reminder email": "follow-up",
    }
    state['category'] = label_map.get(raw_label, "normal")  # Default to normal if unknown
    return state

def reply_email(state: EmailState):
    state['response'] = f"Thank you for your email regarding: '{state['email_content'][:50]}...'.\nWe will get back to you shortly."
    print("[DEBUG] Reply generated.")
    return state

def archive_email(state: EmailState):
    state['response'] = "Email archived (spam or irrelevant)."
    print("[DEBUG] Email archived.")
    return state

def summarize_email(state: EmailState):
    result = summarizer(state['email_content'], max_length=30, min_length=10, do_sample=False)
    state['response'] = result[0]['summary_text']
    print("[DEBUG] Email summarized.")
    return state

# ------------------ Build LangGraph Workflow ------------------
graph = StateGraph(EmailState)

graph.add_node("triage", triage_email)
graph.add_node("reply", reply_email)
graph.add_node("archive", archive_email)
graph.add_node("summarize", summarize_email)

graph.add_conditional_edges(
    "triage",
    lambda state: state['category'],
    {
        "urgent": "reply",
        "normal": "reply",
        "follow-up": "reply",
        "spam": "archive",
        "newsletter": "summarize",
    },
)

graph.add_edge("reply", END)
graph.add_edge("archive", END)
graph.add_edge("summarize", END)
graph.set_entry_point("triage")

workflow = graph.compile()

# ------------------ Process Test Emails ------------------
for idx, email in enumerate(test_emails, 1):
    print(f"\n=== Processing Email {idx} ===")
    email_state = EmailState(email_content=email)
    result = workflow.invoke(email_state)
    print("Email Snippet:", email.strip()[:80], "...")
    print("Category:", result['category'])
    print("Action Result:", result['response'])
