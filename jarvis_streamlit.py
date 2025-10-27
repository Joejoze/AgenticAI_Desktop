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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="JARVIS AI Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "jarvis" not in st.session_state:
    st.session_state.jarvis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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
            st.success("✅ JARVIS initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize JARVIS: {e}")
        return False

def auto_initialize():
    """Auto-initialize JARVIS on first load"""
    if not st.session_state.auto_initialized:
        try:
            # Check if GROQ_API_KEY is available
            if os.getenv("GROQ_API_KEY"):
                st.session_state.jarvis = JARVISAssistant()
                st.session_state.system_integration = JARVISSystemIntegration()
                st.session_state.auto_initialized = True
                
                # Try to connect Gmail automatically
                try:
                    service = get_service()
                    st.session_state.connected = True
                except Exception as e:
                    # Gmail connection failed, but that's okay
                    st.session_state.connected = False
                    
            else:
                st.session_state.auto_initialized = True  # Mark as attempted
        except Exception as e:
            st.session_state.auto_initialized = True  # Mark as attempted
            logger.error(f"Auto-initialization failed: {e}")

def process_chat_message(message: str) -> str:
    """Process chat message through JARVIS"""
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
        
        # Add to chat history
        st.session_state.chat_history.append({
            "user": message.strip(),
            "jarvis": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return f"❌ Error processing message: {e}"

def handle_special_commands(user_input: str) -> str:
    """Handle special commands that don't go through JARVIS"""
    try:
        command_lower = user_input.lower().strip()
        
        if command_lower == "clear":
            st.session_state.chat_history = []
            return "Chat history cleared!"
        
        elif command_lower == "status":
            status_info = []
            status_info.append(f"JARVIS: {'✅ Ready' if st.session_state.jarvis else '❌ Not initialized'}")
            status_info.append(f"Gmail: {'✅ Connected' if st.session_state.connected else '❌ Not connected'}")
            status_info.append(f"Voice: {'✅ Available' if st.session_state.jarvis and st.session_state.jarvis.voice_enabled else '❌ Not available'}")
            status_info.append(f"Chat History: {len(st.session_state.chat_history)} messages")
            return "\n".join(status_info)
        
        return None
        
    except Exception as e:
        logger.error(f"Error in handle_special_commands: {e}")
        return f"❌ Error processing command: {e}"

# Auto-initialize JARVIS
auto_initialize()

# Main UI - ChatGPT-like interface
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🤖 JARVIS AI Assistant")
    st.caption("Your complete AI assistant with voice, file management, and system control")

with col2:
    # Compact status display
    if st.session_state.jarvis:
        st.success("✅ JARVIS Ready")
    else:
        st.error("❌ JARVIS Offline")
    
    if st.session_state.connected:
        st.success("📧 Gmail Connected")
    else:
        st.info("📧 Gmail Disconnected")
    
    st.info(f"💬 {len(st.session_state.chat_history)} Messages")

# Quick actions
st.markdown("### 🚀 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎤 Test Voice", help="Test voice synthesis"):
        if st.session_state.jarvis and st.session_state.jarvis.voice_enabled:
            st.session_state.jarvis.speak("Hello! JARVIS voice test successful.")
            st.success("🗣️ Voice test completed")
        else:
            st.error("❌ Voice not available")

with col2:
    if st.button("📧 Check Emails", help="Check recent emails"):
        if st.session_state.connected:
            try:
                service = get_service()
                emails = fetch_recent_emails(service, max_results=3)
                if emails:
                    st.success(f"📧 Found {len(emails)} recent emails")
                else:
                    st.info("📭 No recent emails")
            except Exception as e:
                st.error(f"❌ Email error: {e}")
        else:
            st.error("❌ Gmail not connected")

with col3:
    if st.button("💻 System Status", help="Check system status"):
        if st.session_state.system_integration:
            try:
                status = st.session_state.system_integration.handle_system_command("system status")
                st.success("💻 System status checked")
            except Exception as e:
                st.error(f"❌ System error: {e}")
        else:
            st.error("❌ System integration not available")

with col4:
    if st.button("🗑️ Clear Chat", help="Clear chat history"):
        st.session_state.chat_history = []
        st.success("🗑️ Chat cleared")

# Main chat interface
st.markdown("### 💬 Chat with JARVIS")

# Chat history container with better scrolling
chat_container = st.container()

with chat_container:
    if st.session_state.chat_history:
        # Show last 20 messages to reduce scrolling issues
        recent_messages = st.session_state.chat_history[-20:]
        
        for i, chat in enumerate(recent_messages):
            # User message
            with st.chat_message("user"):
                st.write(chat['user'])
            
            # JARVIS response
            with st.chat_message("assistant"):
                st.write(chat['jarvis'])
            
            # Timestamp (smaller, less intrusive)
            st.caption(f"*{chat['timestamp'][:19]}*")
            
            # Add some spacing but not too much
            if i < len(recent_messages) - 1:
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation with JARVIS! Try asking: 'list files in downloads' or 'check system status'")

# Chat input
user_input = st.chat_input("Ask JARVIS anything... (e.g., 'list files in downloads', 'check system status', 'search for AI news')")

if user_input:
    # Check for special commands first
    special_response = handle_special_commands(user_input)
    
    if special_response:
        st.session_state.chat_history.append({
            "user": user_input,
            "jarvis": special_response,
            "timestamp": datetime.now().isoformat()
        })
        st.rerun()
    else:
        # Process through JARVIS
        if st.session_state.jarvis is not None:
            response = process_chat_message(user_input)
            st.rerun()
        else:
            st.session_state.chat_history.append({
                "user": user_input,
                "jarvis": "❌ JARVIS not initialized. Please check your GROQ_API_KEY and try initializing manually.",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()

# Sidebar for advanced features
with st.sidebar:
    st.header("⚙️ Advanced Features")
    
    # Gmail connection
    st.subheader("📧 Gmail")
    if st.button("Connect Gmail"):
        try:
            service = get_service()
            st.session_state.connected = True
            st.success("✅ Gmail connected!")
        except Exception as e:
            st.error(f"❌ Gmail connection failed: {e}")
    
    if st.button("Disconnect Gmail"):
        st.session_state.connected = False
        st.success("📧 Gmail disconnected")
    
    # Voice controls
    st.subheader("🎤 Voice")
    if st.session_state.jarvis and st.session_state.jarvis.voice_enabled:
        if st.button("🎤 Start Voice Mode"):
            st.session_state.voice_mode = True
            st.success("🎤 Voice mode activated")
        
        if st.button("🔇 Stop Voice Mode"):
            st.session_state.voice_mode = False
            st.success("🔇 Voice mode deactivated")
        
        # Voice test
        test_text = st.text_input("Test voice:", value="Hello! This is JARVIS.")
        if st.button("🗣️ Test Voice"):
            st.session_state.jarvis.speak(test_text)
            st.success("🗣️ Voice test completed")
    else:
        st.error("❌ Voice not available")
    
    # System commands
    st.subheader("💻 System")
    system_command = st.text_input("System command:", placeholder="e.g., list files in downloads")
    if st.button("🔧 Execute"):
        if st.session_state.system_integration:
            try:
                result = st.session_state.system_integration.handle_system_command(system_command)
                st.text_area("Result:", result, height=200)
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.error("❌ System integration not available")
    
    # File operations
    st.subheader("📁 Files")
    file_path = st.text_input("File path:", placeholder="e.g., C:/Users/Username/Downloads")
    if st.button("📋 List Files"):
        if st.session_state.system_integration:
            try:
                result = st.session_state.system_integration.handle_system_command(f"list files in {file_path}")
                st.text_area("Files:", result, height=200)
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.error("❌ System integration not available")
    
    # Settings
    st.subheader("⚙️ Settings")
    if st.button("🔄 Reinitialize JARVIS"):
        st.session_state.jarvis = None
        st.session_state.auto_initialized = False
        auto_initialize()
        st.success("🔄 JARVIS reinitialized")
    
    if st.button("🗑️ Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.connected = False
        st.session_state.voice_mode = False
        st.success("🗑️ All data cleared")

# Footer
st.markdown("---")
st.markdown("🤖 **JARVIS AI Assistant** - Complete system control with voice, file management, and Gmail integration")