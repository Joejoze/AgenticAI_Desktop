# =============================
# SIMPLIFIED LANGGRAPH MEMORY SYSTEM
# =============================

import os
import json
from datetime import datetime
import concurrent.futures
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass

# Basic imports (no heavy dependencies)
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# State definition for LangGraph
class MemoryState(TypedDict):
    email_data: Dict[str, Any]
    memory_context: str
    classification: str
    action_taken: str
    memory_stored: bool
    memory_id: str

@dataclass
class EmailMemory:
    """Simple memory structure for email interactions"""
    id: str
    timestamp: str
    sender: str
    subject: str
    content: str
    classification: str
    action_taken: str
    importance: float

class SimpleLangGraphMemorySystem:
    """Simplified long-term memory system using basic LangGraph concepts"""
    
    def __init__(self, persist_directory: str = "./simple_memory_db"):
        self.persist_directory = persist_directory
        self.timeout_seconds = 25
        
        # In-memory storage for quick access
        self.memories: List[EmailMemory] = []
        self.memory_file = os.path.join(persist_directory, "memories.json")
        
        # Setup LLM
        gemini_key = os.getenv("GEMINI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        # Prefer Gemini if available; fallback to Groq; else raise clear error
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=gemini_key,
                    temperature=0.7,
                )
                print("[Memory LLM] Using Gemini 2.5 Flash via LangChain")
            except Exception as e:
                print(f"[Memory LLM WARNING] Gemini setup failed: {e}. Falling back to Groq if available.")
                if not groq_key:
                    raise RuntimeError("No valid LLM configured: set GEMINI_API_KEY or GROQ_API_KEY in .env")
                self.llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
                print("[Memory LLM] Using Groq Llama 3.1 8B Instant")
        elif groq_key:
            self.llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
            print("[Memory LLM] Using Groq Llama 3.1 8B Instant")
        else:
            raise RuntimeError("No valid LLM configured for memory system. Please set GEMINI_API_KEY or GROQ_API_KEY in .env")
        
        # Load existing memories
        self._load_memories()
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM with a timeout to avoid hangs."""
        def _invoke():
            return self.llm.invoke(prompt).content
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_invoke)
            return fut.result(timeout=self.timeout_seconds)
    
    def _load_memories(self):
        """Load memories from disk"""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memories = [EmailMemory(**mem) for mem in data.get('memories', [])]
            except Exception as e:
                print(f"Error loading memories: {e}")
                self.memories = []
    
    def _save_memories(self):
        """Save memories to disk"""
        try:
            data = {'memories': [mem.__dict__ for mem in self.memories]}
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def _calculate_importance(self, email_data: Dict[str, Any]) -> float:
        """Calculate importance score for email"""
        score = 0.0
        content = (email_data['subject'] + ' ' + email_data['body']).lower()
        
        # Check for urgent keywords
        urgent_words = ['urgent', 'asap', 'deadline', 'meeting', 'important', 'boss']
        for word in urgent_words:
            if word in content:
                score += 0.2
        
        # Check for action requests
        action_words = ['please', 'need', 'request', 'send', 'reply', 'confirm']
        for word in action_words:
            if word in content:
                score += 0.1
        
        return min(score, 1.0)
    
    def get_context_for_email(self, email_data: Dict[str, Any]) -> str:
        """Get relevant context from memory for current email"""
        context_parts = []
        
        # Check if this is actually email data
        if not isinstance(email_data, dict) or 'from' not in email_data:
            # This is not email data, return general context
            return self.get_general_context(email_data.get('content', ''))
        
        # Add sender history
        sender_memories = [mem for mem in self.memories 
                          if mem.sender == email_data['from']]
        if sender_memories:
            recent_sender = max(sender_memories, key=lambda x: x.timestamp)
            context_parts.append(f"Previous email from {email_data['from']}: {recent_sender.subject}")
        
        # Add similar content (simple keyword matching)
        current_text = (email_data['subject'] + ' ' + email_data['body']).lower()
        for memory in self.memories[-5:]:  # Check last 5 memories
            if memory.sender != email_data['from']:  # Avoid duplicates
                memory_text = (memory.subject + ' ' + memory.content).lower()
                # Simple similarity check
                common_words = set(current_text.split()) & set(memory_text.split())
                if len(common_words) > 2:  # If more than 2 common words
                    context_parts.append(f"Similar email: {memory.subject} ({memory.classification})")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found"
    
    def get_general_context(self, content):
        """Get general context for non-email content"""
        context_parts = []
        
        # Add recent memories
        if self.memories:
            recent_memories = self.memories[-3:]  # Last 3 memories
            for mem in recent_memories:
                context_parts.append(f"Recent: {mem.subject} ({mem.classification})")
        
        return "\n".join(context_parts) if context_parts else "No recent context found"
    
    def store_memory(self, email_data: Dict[str, Any], classification: str, action_taken: str) -> str:
        """Store email interaction in memory"""
        # Check if this is email data or general content
        if isinstance(email_data, dict) and 'subject' in email_data:
            # This is email data
            memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(email_data['subject'])}"
            
            memory = EmailMemory(
                id=memory_id,
                timestamp=datetime.now().isoformat(),
                sender=email_data['from'],
                subject=email_data['subject'],
                content=email_data['body'],
                classification=classification,
                action_taken=action_taken,
                importance=self._calculate_importance(email_data)
            )
        else:
            # This is general content
            content = email_data.get('content', '')
            memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(content)}"
            
            memory = EmailMemory(
                id=memory_id,
                timestamp=datetime.now().isoformat(),
                sender=email_data.get('source', 'unknown'),
                subject=content[:50] + "..." if len(content) > 50 else content,
                content=content,
                classification=classification,
                action_taken=action_taken,
                importance=0.5  # Default importance for general content
            )
        
        # Store in memory list
        self.memories.append(memory)
        
        # Save to disk
        self._save_memories()
        
        return memory_id

def create_simple_memory_workflow(memory_system: SimpleLangGraphMemorySystem):
    """Create a simple workflow for email processing with memory"""
    
    def get_memory_context(state: MemoryState) -> MemoryState:
        """Node to get relevant memory context"""
        email_data = state["email_data"]
        context = memory_system.get_context_for_email(email_data)
        
        return {
            **state,
            "memory_context": context
        }
    
    def classify_with_memory(state: MemoryState) -> MemoryState:
        """Node to classify email using memory context"""
        email_data = state["email_data"]
        memory_context = state["memory_context"]
        
        # Check if this is email data or general content
        if isinstance(email_data, dict) and 'from' in email_data:
            # This is email data
            prompt = f"""
            You are an AI Email Assistant with access to email history.
            
            Previous Context:
            {memory_context}
            
            Classify the following email as one of: spam, important, normal, reply_needed.
            
            Email:
            From: {email_data['from']}
            Subject: {email_data['subject']}
            Body: {email_data['body']}
            
            Consider the context from previous emails when classifying.
            Answer with only one word.
            """
        else:
            # This is general content
            prompt = f"""
            You are an AI Assistant with access to conversation history.
            
            Previous Context:
            {memory_context}
            
            Classify the following content as one of: question, command, information, greeting.
            
            Content: {email_data.get('content', '')}
            Source: {email_data.get('source', 'unknown')}
            
            Consider the context from previous interactions when classifying.
            Answer with only one word.
            """
        
        try:
            result = memory_system._call_llm(prompt).strip().lower()
        except Exception:
            # Safe fallback classification
            text = email_data.get('body') if isinstance(email_data, dict) else email_data.get('content', '')
            if text and any(k in text.lower() for k in ["urgent", "asap", "deadline", "meeting"]):
                result = "important"
            else:
                result = "normal"
        
        return {
            **state,
            "classification": result
        }
    
    def generate_response_with_memory(state: MemoryState) -> MemoryState:
        """Node to generate response using memory context"""
        email_data = state["email_data"]
        classification = state["classification"]
        memory_context = state["memory_context"]
        
        # Check if this is email data or general content
        if isinstance(email_data, dict) and 'from' in email_data:
            # This is email data
            prompt = f"""
            You are an AI writing assistant with access to email history.
            
            Memory Context: {memory_context}
            
            Write a response to the following email based on its classification: {classification}
            
            Email:
            From: {email_data['from']}
            Subject: {email_data['subject']}
            Body: {email_data['body']}
            
            Consider the context and previous interactions when crafting your response.
            """
        else:
            # This is general content
            prompt = f"""
            You are an AI assistant with access to conversation history.
            
            Memory Context: {memory_context}
            
            Respond to the following content based on its classification: {classification}
            
            Content: {email_data.get('content', '')}
            Source: {email_data.get('source', 'unknown')}
            
            Consider the context and previous interactions when crafting your response.
            """
        
        try:
            response = memory_system._call_llm(prompt)
        except Exception:
            # Fallback short response
            if isinstance(email_data, dict) and 'from' in email_data:
                response = "Thank you for your email. I'll get back to you shortly."
            else:
                response = "Got it. How else can I help?"
        
        return {
            **state,
            "action_taken": response
        }
    
    def store_memory(state: MemoryState) -> MemoryState:
        """Node to store email in long-term memory"""
        email_data = state["email_data"]
        classification = state["classification"]
        action_taken = state["action_taken"]
        
        # Store in long-term memory
        memory_id = memory_system.store_memory(email_data, classification, action_taken)
        
        return {
            **state,
            "memory_id": memory_id,
            "memory_stored": True
        }
    
    # Simple workflow execution (simulating LangGraph)
    def execute_workflow(email_data: Dict[str, Any]) -> MemoryState:
        """Execute the workflow steps"""
        state = {"email_data": email_data}
        
        # Step 1: Get memory context
        state = get_memory_context(state)
        
        # Step 2: Classify with memory
        state = classify_with_memory(state)
        
        # Step 3: Generate response
        state = generate_response_with_memory(state)
        
        # Step 4: Store memory
        state = store_memory(state)
        
        return state
    
    return execute_workflow

def get_memory_insights(memory_system: SimpleLangGraphMemorySystem) -> Dict[str, Any]:
    """Get insights from stored memories"""
    if not memory_system.memories:
        return {"message": "No memories stored yet"}
    
    insights = {
        'total_emails': len(memory_system.memories),
        'classification_distribution': {},
        'top_senders': {},
        'recent_activity': []
    }
    
    # Classification distribution
    for memory in memory_system.memories:
        classification = memory.classification
        insights['classification_distribution'][classification] = \
            insights['classification_distribution'].get(classification, 0) + 1
    
    # Top senders
    for memory in memory_system.memories:
        sender = memory.sender
        insights['top_senders'][sender] = insights['top_senders'].get(sender, 0) + 1
    
    # Sort top senders
    insights['top_senders'] = dict(sorted(
        insights['top_senders'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5])
    
    # Recent activity (last 5 emails)
    recent_memories = sorted(
        memory_system.memories, 
        key=lambda x: x.timestamp, 
        reverse=True
    )[:5]
    
    insights['recent_activity'] = [
        {
            'timestamp': mem.timestamp,
            'sender': mem.sender,
            'subject': mem.subject,
            'classification': mem.classification,
            'importance': mem.importance
        }
        for mem in recent_memories
    ]
    
    return insights

# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory_system = SimpleLangGraphMemorySystem()
    workflow = create_simple_memory_workflow(memory_system)
    
    # Example email processing
    email_data = {
        'from': 'test@example.com',
        'subject': 'Test Email with Simple Memory',
        'body': 'This is a test email for the simplified LangGraph memory system.'
    }
    
    # Process email through workflow
    print("Processing email with simplified LangGraph memory system...")
    result = workflow(email_data)
    
    print("\nProcessing result:")
    print(f"Classification: {result['classification']}")
    print(f"Action taken: {result['action_taken']}")
    print(f"Memory ID: {result['memory_id']}")
    
    # Get memory insights
    print("\nMemory insights:")
    insights = get_memory_insights(memory_system)
    print(json.dumps(insights, indent=2))
