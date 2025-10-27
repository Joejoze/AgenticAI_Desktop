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

def initialize_jarvis():
    """Initialize JARVIS system"""
    try:
        if st.session_state.jarvis is None:
            with st.spinner("Initializing JARVIS..."):
                st.session_state.jarvis = JARVISAssistant()
                st.session_state.system_integration = JARVISSystemIntegration()
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
        
        # Get JARVIS response
        if st.session_state.jarvis:
            try:
                # Create command object
                command = Command(
                    type=CommandType.TEXT,
                    content=prompt,
                    source="streamlit",
                    timestamp=datetime.now()
                )
                
                # Get response
                with st.spinner("JARVIS is thinking..."):
                    response = st.session_state.jarvis.process_command(command)
                
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
