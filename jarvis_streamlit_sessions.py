import os
import streamlit as st
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

# Import JARVIS components
from jarvis import JARVISAssistant, Command, CommandType
from agent import fetch_recent_emails, process_email, get_memory_insights
from emailapi import get_service, send_email
from system_manager import JARVISSystemIntegration
from chat_database import chat_db
from smart_nlp import SmartNLP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="JARVIS AI Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "jarvis" not in st.session_state:
    st.session_state.jarvis = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "system_integration" not in st.session_state:
    st.session_state.system_integration = None
if "auto_initialized" not in st.session_state:
    st.session_state.auto_initialized = False
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False
if "nlp_processor" not in st.session_state:
    st.session_state.nlp_processor = None

def initialize_jarvis():
    """Initialize JARVIS system"""
    try:
        if st.session_state.jarvis is None:
            with st.spinner("Initializing JARVIS..."):
                st.session_state.jarvis = JARVISAssistant()
                st.session_state.system_integration = JARVISSystemIntegration()
                st.session_state.nlp_processor = SmartNLP()
            st.success("[SUCCESS] JARVIS initialized successfully!")
        return True
    except Exception as e:
        st.error(f"[ERROR] Failed to initialize JARVIS: {e}")
        return False

def get_or_create_session():
    """Get current session or create a new one"""
    if not st.session_state.current_session_id:
        # Create a new session
        st.session_state.current_session_id = chat_db.create_session()
    
    return st.session_state.current_session_id

def load_messages(session_id: str) -> List[Dict[str, Any]]:
    """Load messages for a session"""
    return chat_db.get_messages(session_id)

def save_message(session_id: str, role: str, content: str, metadata: Dict = None):
    """Save a message to the database"""
    return chat_db.add_message(session_id, role, content, metadata)

def process_chat_message(message: str) -> str:
    """Process chat message through JARVIS (exact copy from jarvis_streamlit.py)"""
    try:
        if st.session_state.jarvis is None:
            return "❌ JARVIS not initialized. Please initialize first."
        
        # Validate message
        if not message or not message.strip():
            return "❌ Please enter a valid message."
        
        # Create command object
        command = Command(
            type=CommandType.TEXT,
            content=message.strip(),
            source="streamlit_chat",
            timestamp=datetime.now()
        )
        
        # Process through JARVIS with auto-learning
        response = st.session_state.jarvis.process_command(command)
        
        # Learn from this interaction
        st.session_state.jarvis.learn_from_interaction(command, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return f"❌ Error processing message: {e}"

def process_with_ai_routing(user_message: str, nlp_context: dict, jarvis, timeout_seconds: int = 60) -> str:
    """Decide and execute actions (list files, read files, etc.) using NLP context.

    Minimal routing to unblock UI: handles common file ops directly, otherwise falls back
    to the standard chat processing.
    """
    try:
        import os
        import subprocess
        from system_manager import JARVISSystemIntegration

        action = None
        path = None
        if nlp_context:
            action = (nlp_context.get("action") or "").lower()
            intent = (nlp_context.get("intent") or "").lower()
            path = nlp_context.get("path")

            # LIST FILES
            if action in ["list", "list_files"] or intent == "list_files":
                if path:
                    try:
                        system = JARVISSystemIntegration()
                        results = system.file_manager.list_files(path)
                        if isinstance(results, list) and results and (not isinstance(results[0], dict) or "error" not in results[0]):
                            out = f"📁 Files in {path}:\n\n"
                            for f in results[:100]:
                                icon = "📁" if f.get("is_directory") else "📄"
                                name = f.get("name", "Unknown")
                                size = f.get("size", 0)
                                size_str = f" ({size} bytes)" if f.get("is_file") else ""
                                out += f"{icon} {name}{size_str}\n"
                            if len(results) > 100:
                                out += f"\n... and {len(results)-100} more items"
                            return out
                    except Exception:
                        pass

                    # Fallback to PowerShell listing
                    try:
                        ps = f"powershell -Command \"Get-ChildItem -Path '{path}' | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize\""
                        r = subprocess.run(ps, shell=True, capture_output=True, text=True, timeout=30)
                        if r.stdout.strip():
                            return f"```\n{r.stdout.strip()}\n```"
                        if r.stderr.strip():
                            return f"❌ {r.stderr.strip()}"
                    except Exception as e:
                        return f"❌ Error listing files: {e}"
                else:
                    return "❌ No directory path detected. Please specify which folder to list."

            # READ FILE
            if action in ["read", "read_file"] or intent == "read_file":
                if path:
                    try:
                        ps = f"powershell -Command \"Get-Content -Raw -Path '{path}'\""
                        r = subprocess.run(ps, shell=True, capture_output=True, text=True, timeout=30)
                        if r.returncode == 0:
                            content = r.stdout or ""
                            max_len = 10000
                            if len(content) > max_len:
                                content = content[:max_len] + "\n...\n(truncated)"
                            return f"```\n{content.strip()}\n```"
                        else:
                            err = r.stderr.strip() or "Error reading file"
                            return f"❌ {err}"
                    except Exception as e:
                        return f"❌ Error reading file: {e}"
                else:
                    return "❌ No file path detected. Please specify which file to read."

        # Fallback to regular processing
        return process_chat_message(user_message)
    except Exception as e:
        return f"❌ Error: {e}"

def render_sidebar():
    """Render the sidebar with chat sessions"""
    with st.sidebar:
                # Session management
        st.subheader("⚙️ Session Settings")
        
        current_session = chat_db.get_active_session()
        if current_session:
            new_name = st.text_input(
                "Rename current session:",
                value=current_session['name'],
                key="rename_input"
            )
            
            if st.button("✏️ Rename", use_container_width=True):
                if new_name and new_name != current_session['name']:
                    if chat_db.rename_session(current_session['id'], new_name):
                        st.success("Session renamed!")
                        st.rerun()
                    else:
                        st.error("Failed to rename session")

        st.title("💬 Chat Sessions")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            new_session_id = chat_db.create_session()
            st.session_state.current_session_id = new_session_id
            st.rerun()
        
        st.divider()
        
        # Get all sessions
        sessions = chat_db.get_all_sessions()
        
        if not sessions:
            st.info("No chat sessions yet. Start a new chat!")
            return
        
        # Display sessions
        for session in sessions:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Session name with click to switch
                if st.button(
                    session['name'], 
                    key=f"session_{session['id']}",
                    use_container_width=True,
                    type="primary" if session['is_active'] else "secondary"
                ):
                    st.session_state.current_session_id = session['id']
                    chat_db.switch_to_session(session['id'])
                    st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                    if chat_db.delete_session(session['id']):
                        if st.session_state.current_session_id == session['id']:
                            # If we deleted the current session, create a new one
                            st.session_state.current_session_id = chat_db.create_session()
                        st.rerun()
                    else:
                        st.error("Failed to delete session")
            
            # Session stats
            stats = chat_db.get_session_stats(session['id'])
            if stats['message_count'] > 0:
                st.caption(f"📊 {stats['message_count']} messages")
                if stats['last_message']:
                    last_msg_time = datetime.fromisoformat(stats['last_message'].replace('Z', '+00:00'))
                    st.caption(f"🕒 {last_msg_time.strftime('%H:%M')}")
        
        st.divider()
        


def render_chat_interface():
    """Render the main chat interface"""
    # Get current session
    session_id = get_or_create_session()
    current_session = chat_db.get_active_session()
    
    if not current_session:
        st.error("No active session found. Please create a new chat.")
        return
    
    # Display session name
    st.title(f"💬 {current_session['name']}")
    
    # Load messages
    messages = load_messages(session_id)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in messages:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask JARVIS anything..."):
        # Add user message to database
        save_message(session_id, "user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Process with SmartNLP pipeline
        nlp_result = None
        resolved_path = None
        if st.session_state.nlp_processor:
            nlp_result = st.session_state.nlp_processor.process(prompt)
            resolved_path = nlp_result.path_intent
            
            # Show NLP insights in expander
            with st.expander("NLP Insights", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Original:** {nlp_result.original_text}")
                    st.write(f"**Normalized:** {nlp_result.normalized_text}")
                    if resolved_path:
                        st.write(f"**Detected Path:** `{resolved_path}`")
                with col2:
                    st.write(f"**Intent:** {nlp_result.intent} ({nlp_result.confidence:.0%})")
                    st.write(f"**Action:** {nlp_result.action}")
                    st.write(f"**Context:** {nlp_result.context_summary}")
                
                if nlp_result.entities:
                    with st.expander("Entities", expanded=False):
                        st.json({k: v for k, v in nlp_result.entities.items() if v})
        
        # Get JARVIS response
        if st.session_state.jarvis:
            try:
                # Use AI-driven routing so the model can decide to run commands, read files, etc.
                with st.spinner("JARVIS is thinking..."):
                    nlp_context = None
                    if nlp_result:
                        nlp_context = {
                            "original": nlp_result.original_text,
                            "normalized": nlp_result.normalized_text,
                            "intent": nlp_result.intent,
                            "confidence": nlp_result.confidence,
                            "action": nlp_result.action,
                            "entities": nlp_result.entities,
                            "path": resolved_path,
                            "context_summary": nlp_result.context_summary,
                        }
                    response = process_with_ai_routing(prompt, nlp_context, st.session_state.jarvis)

                # Remember last command for retry
                st.session_state.last_command = prompt

                # Update NLP context
                if st.session_state.nlp_processor and nlp_result:
                    st.session_state.nlp_processor.update_context(prompt, nlp_result, response)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)
                
                # Save assistant response to database
                save_message(session_id, "assistant", response)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(error_msg)
                save_message(session_id, "assistant", error_msg)
        else:
            error_msg = "JARVIS not initialized. Please check the connection."
            with st.chat_message("assistant"):
                st.error(error_msg)
            save_message(session_id, "assistant", error_msg)


def main():
    """Main application"""
    # Auto-initialize JARVIS
    if not st.session_state.auto_initialized:
        if initialize_jarvis():
            st.session_state.auto_initialized = True
            st.session_state.connected = True
    
    # Render sidebar
    render_sidebar()
    
    
    # Main chat interface
    render_chat_interface()
    
    # Status bar
    with st.container():
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.connected:
                st.success("🟢 JARVIS Connected")
            else:
                st.error("🔴 JARVIS Disconnected")
        
        with col2:
            if st.session_state.voice_mode:
                st.info("🎤 Voice Mode ON")
            else:
                st.info("🔇 Voice Mode OFF")
        
        with col3:
            current_session = chat_db.get_active_session()
            if current_session:
                stats = chat_db.get_session_stats(current_session['id'])
                st.info(f"💬 {stats['message_count']} messages")

if __name__ == "__main__":
    main()
