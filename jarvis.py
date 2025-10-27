#!/usr/bin/env python3
"""
JARVIS - Complete AI Assistant System
=====================================

A comprehensive AI assistant with:
- Voice recognition and speech synthesis
- Telegram & WhatsApp integration
- File system access and management
- Google search capabilities
- Unified chat interface
- Gmail integration (existing)
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Core AI and memory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Voice processing (optional)
try:
    import speech_recognition as sr
    import pyttsx3
    import threading
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("WARNING: Voice components not available. Install with: pip install SpeechRecognition pyttsx3 pyaudio")

# File system and web
import requests
from pathlib import Path
import webbrowser

# Messaging platforms (optional)
try:
    import telebot
    from telebot import types
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("WARNING: Telegram components not available. Install with: pip install pyTelegramBotAPI")

# Existing components
from emailapi import get_service
from simple_langgraph_memory import SimpleLangGraphMemorySystem, create_simple_memory_workflow

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommandType(Enum):
    VOICE = "voice"
    TEXT = "text"
    FILE = "file"
    SEARCH = "search"
    MESSAGE = "message"
    SYSTEM = "system"

@dataclass
class Command:
    type: CommandType
    content: str
    source: str  # telegram, whatsapp, voice, chat
    timestamp: datetime
    metadata: Dict[str, Any] = None

class JARVISAssistant:
    """Complete JARVIS AI Assistant System"""
    
    def __init__(self):
        self.setup_ai()
        self.setup_voice()
        self.setup_memory()
        self.setup_file_system()
        self.setup_messaging()
        self.setup_search()
        self.command_history = []
        self.learning_data = {}
        self.user_preferences = {}
        self.auto_learn = True
        self.current_emails = []  # Store current email list for replies
        
    def setup_ai(self):
        """Initialize the AI model"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env file")
        
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant"
        )
        
        # JARVIS personality and capabilities
        self.system_prompt = """
        You are JARVIS, an advanced AI assistant with full system access. You can help with:
        
        - Email management (check, send, reply to emails)
        - File system operations (read, write, search files)
        - Google search for information
        - System monitoring and control
        - Voice interaction
        
        PERSONALITY:
        - Speak naturally and conversationally, like a helpful human assistant
        - Understand ANY way the user phrases their requests - be flexible with language
        - Don't mention specific commands - just do what the user asks naturally
        - Be proactive and handle tasks automatically in the background
        - When users ask for emails in ANY way, fetch real emails from Gmail
        - When users ask for files in ANY way, access the actual file system
        - Always provide real data, never fake or placeholder information
        - If you're not sure what the user wants, ask for clarification in a friendly way
        
        UNDERSTAND THESE VARIATIONS:
        - "emails" = "mail", "messages", "inbox", "gmail", "email list"
        - "files" = "documents", "folders", "my stuff", "data", "content"
        - "search" = "find", "look up", "google", "research", "check"
        - "system" = "computer", "pc", "machine", "status", "health"
        - "reply" = "respond", "answer", "write back", "send a message"
        - "send" = "email", "message", "write", "compose"
        
        EXAMPLES OF FLEXIBLE UNDERSTANDING:
        - "Can you show me my emails?" → Fetch real emails
        - "I want to see what's in my inbox" → Fetch real emails
        - "Check my mail please" → Fetch real emails
        - "What files do I have?" → List actual files
        - "Show me my documents" → List actual files
        - "Look up AI news" → Perform Google search
        - "How's my computer doing?" → Check system status
        - "Reply to that email" → Reply to most recent email
        - "Send a message to John" → Help compose email
        
        Always be helpful, understanding, and execute what the user actually wants.
        """
    
    def setup_voice(self):
        """Initialize voice recognition and synthesis"""
        if not VOICE_AVAILABLE:
            self.recognizer = None
            self.microphone = None
            self.tts_engine = None
            self.voice_enabled = False
            return
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS with better voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a male voice (JARVIS-like)
                for voice in voices:
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 180)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
            
            # Calibrate microphone for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.voice_enabled = True
            print("SUCCESS: Voice recognition and synthesis enabled")
            
        except Exception as e:
            print(f"WARNING: Voice setup failed: {e}")
            self.recognizer = None
            self.microphone = None
            self.tts_engine = None
            self.voice_enabled = False
        
    def setup_memory(self):
        """Initialize memory system"""
        self.memory_system = SimpleLangGraphMemorySystem()
        self.memory_workflow = create_simple_memory_workflow(self.memory_system)
        
    def setup_file_system(self):
        """Setup file system access"""
        self.home_dir = Path.home()
        self.project_dir = Path.cwd()
        
    def setup_messaging(self):
        """Initialize messaging platforms"""
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.whatsapp_token = os.getenv("WHATSAPP_TOKEN")
        
        if TELEGRAM_AVAILABLE and self.telegram_token:
            try:
                self.telegram_bot = telebot.TeleBot(self.telegram_token)
                self.setup_telegram_handlers()
            except Exception as e:
                print(f"WARNING: Telegram setup failed: {e}")
                self.telegram_bot = None
        else:
            self.telegram_bot = None
        
    def setup_search(self):
        """Setup Google search capabilities"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
    def setup_telegram_handlers(self):
        """Setup Telegram bot handlers"""
        if not TELEGRAM_AVAILABLE or not self.telegram_bot:
            return
        
        @self.telegram_bot.message_handler(commands=['start'])
        def start_command(message):
            self.telegram_bot.reply_to(message, "Hello! I'm JARVIS, your AI assistant. How can I help you?")
        
        @self.telegram_bot.message_handler(func=lambda message: True)
        def handle_message(message):
            command = Command(
                type=CommandType.MESSAGE,
                content=message.text,
                source="telegram",
                timestamp=datetime.now(),
                metadata={"user_id": message.from_user.id, "username": message.from_user.username}
            )
            response = self.process_command(command)
            self.telegram_bot.reply_to(message, response)
    
    def listen_for_voice(self):
        """Listen for voice commands"""
        if not VOICE_AVAILABLE or not self.voice_enabled:
            return None
            
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
            
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with speech recognition: {e}"
    
    def speak(self, text: str):
        """Convert text to speech"""
        if VOICE_AVAILABLE and self.voice_enabled and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"WARNING: Speech synthesis error: {e}")
    
    def listen_for_voice(self, timeout: int = 5) -> str:
        """Listen for voice input and return transcribed text"""
        if not VOICE_AVAILABLE or not self.voice_enabled or not self.recognizer or not self.microphone:
            return ""
        
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                print("Processing speech...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"Heard: {text}")
                return text.lower()
                
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"ERROR: Speech recognition error: {e}")
            return ""
        except Exception as e:
            print(f"WARNING: Voice recognition error: {e}")
            return ""
    
    def start_voice_mode(self):
        """Start continuous voice interaction mode"""
        if not VOICE_AVAILABLE or not self.voice_enabled:
            print("ERROR: Voice mode not available")
            return
        
        print("Voice mode activated! Say 'JARVIS' to activate, 'exit' to quit")
        self.speak("Voice mode activated. Say JARVIS to activate, exit to quit.")
        
        while True:
            try:
                # Listen for activation word
                text = self.listen_for_voice(timeout=10)
                
                if "jarvis" in text or "hey jarvis" in text:
                    self.speak("Yes, sir? How may I assist you?")
                    
                    # Listen for command
                    command_text = self.listen_for_voice(timeout=15)
                    
                    if command_text:
                        if "exit" in command_text or "quit" in command_text:
                            self.speak("Goodbye, sir.")
                            break
                        
                        # Process command
                        command = Command(
                            type=CommandType.VOICE,
                            content=command_text,
                            source="voice",
                            timestamp=datetime.now()
                        )
                        
                        response = self.process_command(command)
                        print(f"JARVIS: {response}")
                        self.speak(response)
                
                elif "exit" in text or "quit" in text:
                    self.speak("Goodbye, sir.")
                    break
                    
            except KeyboardInterrupt:
                print("\nVoice mode deactivated")
                break
            except Exception as e:
                print(f"WARNING: Voice mode error: {e}")
                continue
    
    def stop_voice_mode(self):
        """Stop voice interaction mode"""
        print("Voice mode deactivated")
        self.speak("Voice mode deactivated")
    
    def process_command(self, command: Command) -> str:
        """Process incoming commands with full system access and auto-learning"""
        self.command_history.append(command)
        
        # Load learning data
        self.load_learning_data()
        
        # Check for system/file operations FIRST - before any AI processing
        system_response = self.handle_system_operations(command.content)
        if system_response:
            # Learn from this interaction
            self.learn_from_interaction(command, system_response)
            return system_response
        
        # Process through memory system only if not a system operation
        memory_result = self.memory_workflow({
            "content": command.content,
            "source": command.source,
            "timestamp": command.timestamp.isoformat()
        })
        
        # Handle Gmail commands before AI generation - be more specific
        email_keywords = [
            "email", "emails", "gmail", "mail", "message", "messages", "inbox", 
            "send", "reply", "respond", "write", "compose", "mailbox", "correspondence", "communication"
        ]
        
        # Check for specific email operations (not just "list" or "check")
        email_operations = [
            "check emails", "fetch emails", "get emails", "show emails", "list emails",
            "send email", "reply to", "respond to", "write email", "compose email"
        ]
        
        # Also check for sender mentions (like "rughved", "pepperfry", etc.)
        sender_mentions = []
        if self.current_emails:
            for email in self.current_emails:
                sender_name = email.get('from', '').split('<')[0].strip().lower()
                if sender_name in command.content.lower():
                    sender_mentions.append(sender_name)
        
        # Only process as email command if it's clearly email-related
        is_email_command = (
            any(op in command.content.lower() for op in email_operations) or
            sender_mentions or
            (any(keyword in command.content.lower() for keyword in email_keywords) and 
             not any(fs_keyword in command.content.lower() for fs_keyword in ["desktop", "downloads", "documents", "files", "folder", "directory"]))
        )
        
        if is_email_command:
            gmail_result = self.handle_gmail_command(command.content)
            if gmail_result and not gmail_result.startswith("Available Gmail commands"):
                # Learn from this interaction
                self.learn_from_interaction(command, gmail_result)
                return gmail_result
        
        # Generate response using AI with learned preferences
        learned_prefs = self.get_learned_preferences(command.content)
        response = self.generate_response(command, memory_result, learned_prefs)
        
        # Execute actions if needed
        if command.type == CommandType.SEARCH:
            search_results = self.google_search(command.content)
            response += f"\n\nSearch Results:\n{search_results}"
        
        elif command.type == CommandType.FILE:
            file_result = self.handle_file_operation(command.content)
            response += f"\n\nFile Operation:\n{file_result}"
        
        elif command.type == CommandType.MESSAGE:
            # Handle messaging commands
            pass
        
        # Learn from this interaction
        self.learn_from_interaction(command, response)
        
        return response
    
    def handle_system_operations(self, command: str) -> str:
        """Handle system operations with full access and natural language processing"""
        command_lower = command.lower()
        
        # Skip email commands - they should be handled by Gmail handler
        email_operations = [
            "check emails", "fetch emails", "get emails", "show emails", "list emails",
            "send email", "reply to", "respond to", "write email", "compose email",
            "email from", "message from", "mail from"
        ]
        
        if any(op in command_lower for op in email_operations):
            return None
        
        # Convert natural language to system commands
        system_command = self._convert_to_system_command(command_lower)
        if not system_command:
            return None
        
        try:
            from system_manager import JARVISSystemIntegration
            system = JARVISSystemIntegration()
            result = system.handle_system_command(system_command)
            if result:
                return result
            else:
                # If system manager didn't handle it, continue to AI
                pass
        except Exception as e:
            return f"System operation error: {e}"
    
    def _convert_to_system_command(self, command_lower: str) -> str:
        """Convert natural language requests to specific system commands"""
        
        # File listing patterns
        if any(word in command_lower for word in ["list", "show", "display", "what files", "what's in"]) or \
           any(phrase in command_lower for phrase in ["my downloads", "my desktop", "my documents", "my pictures", "my music", "my videos"]):
            # Extract directory from command
            directory = self._extract_directory(command_lower)
            if directory:
                return f"list files in {directory}"
            else:
                return "list files"
        
        # File reading patterns
        elif any(word in command_lower for word in ["read", "open", "view", "see content"]):
            # Extract file path from command
            file_path = self._extract_file_path(command_lower)
            if file_path:
                return f"read file {file_path}"
            else:
                return None
        
        # File search patterns
        elif any(word in command_lower for word in ["search", "find", "look for"]):
            # Extract search term from command
            search_term = self._extract_search_term(command_lower)
            if search_term:
                return f"search files {search_term}"
            else:
                return None
        
        # System status patterns
        elif any(word in command_lower for word in ["system status", "computer status", "how's my computer", "system info"]):
            return "system status"
        
        # CPU usage patterns
        elif any(word in command_lower for word in ["cpu usage", "cpu", "processor", "how much cpu"]):
            return "cpu usage"
        
        # Memory usage patterns
        elif any(word in command_lower for word in ["memory usage", "memory", "ram", "how much memory"]):
            return "memory usage"
        
        # Running processes patterns
        elif any(word in command_lower for word in ["running processes", "processes", "what's running", "active processes"]):
            return "running processes"
        
        # User profile patterns
        elif any(word in command_lower for word in ["user profile", "my profile", "profile", "user info"]):
            return "user profile"
        
        # PowerShell patterns
        elif any(word in command_lower for word in ["powershell", "ps", "run command"]):
            # Extract command after powershell/ps
            ps_command = self._extract_powershell_command(command_lower)
            if ps_command:
                return f"powershell {ps_command}"
            else:
                return "powershell"
        
        # CMD patterns
        elif any(word in command_lower for word in ["cmd", "command", "run cmd"]):
            # Extract command after cmd
            cmd_command = self._extract_cmd_command(command_lower)
            if cmd_command:
                return f"cmd {cmd_command}"
            else:
                return "cmd"
        
        return None
    
    def _extract_directory(self, command_lower: str) -> str:
        """Extract directory name from natural language command"""
        # Common directory mappings
        directory_mappings = {
            "download": "download",
            "downloads": "download", 
            "desktop": "desktop",
            "documents": "documents",
            "pictures": "pictures",
            "music": "music",
            "videos": "videos",
            "my files": "documents",
            "my downloads": "download",
            "my desktop": "desktop",
            "my documents": "documents",
            "my pictures": "pictures",
            "my music": "music",
            "my videos": "videos"
        }
        
        # Look for directory keywords
        for keyword, directory in directory_mappings.items():
            if keyword in command_lower:
                return directory
        
        # Look for "in [directory]" or "from [directory]" patterns
        import re
        patterns = [
            r"in\s+(\w+)",
            r"from\s+(\w+)",
            r"my\s+(\w+)",
            r"the\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                dir_name = match.group(1)
                if dir_name in directory_mappings:
                    return directory_mappings[dir_name]
                else:
                    return dir_name
        
        return None
    
    def _extract_file_path(self, command_lower: str) -> str:
        """Extract file path from natural language command"""
        import re
        # Look for file paths or file names
        patterns = [
            r"file\s+([^\s]+)",
            r"read\s+([^\s]+)",
            r"open\s+([^\s]+)",
            r"view\s+([^\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_search_term(self, command_lower: str) -> str:
        """Extract search term from natural language command"""
        import re
        # Look for search terms after "search", "find", "look for"
        patterns = [
            r"search\s+(?:for\s+)?(.+)",
            r"find\s+(?:the\s+)?(.+)",
            r"look\s+for\s+(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_powershell_command(self, command_lower: str) -> str:
        """Extract PowerShell command from natural language"""
        import re
        patterns = [
            r"powershell\s+(.+)",
            r"ps\s+(.+)",
            r"run\s+command\s+(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_cmd_command(self, command_lower: str) -> str:
        """Extract CMD command from natural language"""
        import re
        patterns = [
            r"cmd\s+(.+)",
            r"command\s+(.+)",
            r"run\s+cmd\s+(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def learn_from_interaction(self, command: Command, response: str, user_feedback: str = None):
        """Learn from user interactions and improve responses"""
        if not self.auto_learn:
            return
        
        # Store interaction data
        interaction = {
            "timestamp": command.timestamp.isoformat(),
            "command": command.content,
            "response": response,
            "source": command.source,
            "feedback": user_feedback
        }
        
        # Learn from patterns
        command_lower = command.content.lower()
        
        # File system preferences
        if any(keyword in command_lower for keyword in ["downloads", "desktop", "documents"]):
            if "file_preferences" not in self.learning_data:
                self.learning_data["file_preferences"] = []
            self.learning_data["file_preferences"].append({
                "directory": command_lower,
                "timestamp": command.timestamp.isoformat()
            })
        
        # Search preferences
        if "search" in command_lower or "find" in command_lower:
            if "search_preferences" not in self.learning_data:
                self.learning_data["search_preferences"] = []
            self.learning_data["search_preferences"].append({
                "query": command.content,
                "timestamp": command.timestamp.isoformat()
            })
        
        # System monitoring preferences
        if any(keyword in command_lower for keyword in ["system", "cpu", "memory", "status"]):
            if "system_preferences" not in self.learning_data:
                self.learning_data["system_preferences"] = []
            self.learning_data["system_preferences"].append({
                "command": command.content,
                "timestamp": command.timestamp.isoformat()
            })
        
        # Save learning data
        self.save_learning_data()
    
    def save_learning_data(self):
        """Save learning data to file"""
        try:
            with open("jarvis_learning.json", "w") as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def load_learning_data(self):
        """Load learning data from file"""
        try:
            if os.path.exists("jarvis_learning.json"):
                with open("jarvis_learning.json", "r") as f:
                    self.learning_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
    
    def get_learned_preferences(self, context: str) -> Dict[str, Any]:
        """Get learned preferences based on context"""
        preferences = {}
        
        if "file" in context.lower():
            if "file_preferences" in self.learning_data:
                preferences["frequent_directories"] = self.learning_data["file_preferences"][-5:]
        
        if "search" in context.lower():
            if "search_preferences" in self.learning_data:
                preferences["search_patterns"] = self.learning_data["search_preferences"][-5:]
        
        if "system" in context.lower():
            if "system_preferences" in self.learning_data:
                preferences["system_commands"] = self.learning_data["system_preferences"][-5:]
        
        return preferences
    
    def generate_response(self, command: Command, memory_context: Dict, learned_prefs: Dict = None) -> str:
        """Generate AI response with learned preferences"""
        learned_context = ""
        if learned_prefs:
            learned_context = f"\n\nLearned Preferences:\n{json.dumps(learned_prefs, indent=2)}"
        
        prompt = f"""
        {self.system_prompt}
        
        User Command: {command.content}
        Memory Context: {memory_context}
        {learned_context}
        Source: {command.source}
        Memory Context: {memory_context}
        
        Provide a helpful response and execute any requested actions.
        """
        
        response = self.llm.invoke(prompt).content
        return response
    
    def google_search(self, query: str) -> str:
        """Perform Google search"""
        if not self.google_api_key or not self.google_cse_id:
            return "Google search not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env"
        
        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': 5
            }
            
            response = requests.get(url, params=params)
            results = response.json()
            
            if 'items' in results:
                search_results = []
                for item in results['items']:
                    search_results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\n")
                return "\n".join(search_results)
            else:
                return "No search results found."
                
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def handle_file_operation(self, command: str) -> str:
        """Handle file system operations using SystemManager"""
        try:
            from system_manager import JARVISSystemIntegration
            system_integration = JARVISSystemIntegration()
            return system_integration.handle_system_command(command)
        except Exception as e:
            return f"File operation error: {str(e)}"
    
    def handle_gmail_command(self, command: str) -> str:
        """Handle Gmail-specific commands"""
        try:
            from agent import fetch_recent_emails, process_email
            from emailapi import get_service, send_email
            
            command_lower = command.lower()
            
            # Be very flexible with email-related requests
            email_request_keywords = [
                "check", "show", "list", "fetch", "get", "see", "view", "display", "read",
                "emails", "email", "mail", "messages", "inbox", "gmail", "correspondence"
            ]
            
            if any(keyword in command_lower for keyword in email_request_keywords):
                service = get_service()
                
                # Determine number of emails to fetch based on request
                max_results = 5  # default
                if "10" in command_lower or "ten" in command_lower:
                    max_results = 10
                elif "20" in command_lower or "twenty" in command_lower:
                    max_results = 20
                elif "50" in command_lower or "fifty" in command_lower:
                    max_results = 50
                
                emails = fetch_recent_emails(service, max_results=max_results)
                
                if not emails:
                    return "No recent emails found."
                
                # Store emails for reply functionality
                self.current_emails = emails
                
                response = f"Found {len(emails)} recent emails:\n\n"
                for i, email in enumerate(emails, 1):
                    # Classify each email
                    try:
                        from agent import classify_email
                        classification = classify_email(email)
                        
                        # Add classification label
                        if classification == "spam":
                            class_label = "[SPAM]"
                        elif classification == "urgent":
                            class_label = "[URGENT]"
                        elif classification == "important":
                            class_label = "[IMPORTANT]"
                        elif classification == "casual":
                            class_label = "[CASUAL]"
                        else:
                            class_label = "[NORMAL]"
                        
                        response += f"{i}. {class_label} **From:** {email['from']}\n"
                        response += f"   **Subject:** {email['subject']}\n"
                        response += f"   **Preview:** {email['body'][:100]}...\n\n"
                        
                    except Exception as e:
                        # Fallback if classification fails
                        response += f"{i}. [NORMAL] **From:** {email['from']}\n"
                        response += f"   **Subject:** {email['subject']}\n"
                        response += f"   **Preview:** {email['body'][:100]}...\n\n"
                
                response += "\nTo reply to an email, use: 'reply to [number/name/sender] [your message]'"
                response += "\nExamples: 'reply to 1 message', 'reply to first email message', 'reply to message from pepperfry'"
                return response
            
            elif command_lower.startswith("reply to "):
                # Handle reply to specific email
                if not self.current_emails:
                    return "ERROR: No emails loaded. Please run 'check emails' first."
                
                remaining = command[9:].strip()  # Everything after "reply to "
                
                # Check if it's a natural language reference
                email_refs = [
                    "first email", "first mail", "email 1", "mail 1", "the first", "1st email", "1st mail",
                    "second email", "second mail", "email 2", "mail 2", "the second", "2nd email", "2nd mail",
                    "third email", "third mail", "email 3", "mail 3", "the third", "3rd email", "3rd mail",
                    "fourth email", "fourth mail", "email 4", "mail 4", "the fourth", "4th email", "4th mail",
                    "fifth email", "fifth mail", "email 5", "mail 5", "the fifth", "5th email", "5th mail"
                ]
                
                # Check if it's a reference by sender name (multiple patterns)
                sender_reference = None
                
                # Pattern 1: "reply to message from pepperfry"
                if "message from" in command_lower or "email from" in command_lower or "mail from" in command_lower:
                    for phrase in ["message from", "email from", "mail from"]:
                        if phrase in command_lower:
                            sender_part = command_lower.split(phrase, 1)[1].strip()
                            sender_name = sender_part.split()[0] if sender_part.split() else ""
                            sender_reference = sender_name
                            break
                
                # Pattern 2: "reply to quora digest" or "reply to pepperfry" (direct sender reference)
                elif not any(ref in command_lower for ref in email_refs):
                    # Check if the remaining text looks like a sender name
                    remaining_words = remaining.split()
                    if remaining_words:
                        potential_sender = remaining_words[0]
                        # Check if this matches any sender in the emails
                        for email in self.current_emails:
                            sender_email = email.get('from', '').lower()
                            sender_name = sender_email.split('<')[0].strip().lower()
                            if (potential_sender.lower() in sender_name or 
                                potential_sender.lower() in sender_email or
                                sender_name in potential_sender.lower()):
                                sender_reference = potential_sender
                                break
                
                
                # Determine which email to reply to
                email_index = None
                
                # First check if it's a sender reference
                if sender_reference:
                    # Find email by sender name (fuzzy matching)
                    for i, email in enumerate(self.current_emails):
                        sender_email = email.get('from', '').lower()
                        sender_name = sender_email.split('<')[0].strip().lower()
                        
                        # Check if sender name contains the reference
                        if (sender_reference.lower() in sender_name or 
                            sender_reference.lower() in sender_email or
                            sender_name in sender_reference.lower()):
                            email_index = i
                            break
                    
                    if email_index is None:
                        return f"ERROR: No email found from sender containing '{sender_reference}'. Available senders: {', '.join([email.get('from', '').split('<')[0].strip() for email in self.current_emails[:3]])}"
                
                # If not a sender reference, check for positional references
                elif any(ref in command_lower for ref in ["first", "1st", "email 1", "mail 1"]):
                    email_index = 0
                elif any(ref in command_lower for ref in ["second", "2nd", "email 2", "mail 2"]):
                    email_index = 1
                elif any(ref in command_lower for ref in ["third", "3rd", "email 3", "mail 3"]):
                    email_index = 2
                elif any(ref in command_lower for ref in ["fourth", "4th", "email 4", "mail 4"]):
                    email_index = 3
                elif any(ref in command_lower for ref in ["fifth", "5th", "email 5", "mail 5"]):
                    email_index = 4
                
                # Store the reference that was found for message parsing
                found_ref = None
                for ref in email_refs:
                    if ref in command_lower:
                        found_ref = ref
                        break
                
                # If no natural language reference found, try numeric parsing
                if email_index is None:
                    try:
                        parts = remaining.split(" ", 1)
                        if len(parts) >= 2:
                            email_num = int(parts[0])
                            email_index = email_num - 1  # Convert to 0-based index
                            remaining = parts[1]  # Update remaining to be the message
                        else:
                            return "ERROR: Invalid reply format. Use: 'reply to [number/name] [your message]'"
                    except ValueError:
                        return "ERROR: Invalid reply format. Use: 'reply to [number/name] [your message]'"
                
                # Validate email index
                if email_index < 0 or email_index >= len(self.current_emails):
                    return f"ERROR: Invalid email reference. Please choose between 1 and {len(self.current_emails)}"
                
                # Extract the message
                if email_index is not None and found_ref:
                    # For natural language, find where the reference ends and message begins
                    message_start = command_lower.find(found_ref) + len(found_ref)
                    reply_message = command[message_start:].strip()
                elif sender_reference:
                    # For sender references, find where the sender name ends and message begins
                    for phrase in ["message from", "email from", "mail from"]:
                        if phrase in command_lower:
                            phrase_start = command_lower.find(phrase) + len(phrase)
                            after_phrase = command[phrase_start:].strip()
                            # Remove the sender name and get the message
                            words = after_phrase.split()
                            if len(words) > 1:
                                reply_message = " ".join(words[1:])  # Skip first word (sender name)
                            else:
                                reply_message = ""
                            break
                    else:
                        reply_message = remaining
                else:
                    reply_message = remaining
                
                if not reply_message:
                    # Generate a smart suggested message based on the email content
                    target_email = self.current_emails[email_index]
                    email_subject = target_email.get('subject', '')
                    email_body = target_email.get('body', '')
                    sender_name = target_email.get('from', '').split('<')[0].strip()
                    
                    # Generate a contextual reply suggestion
                    suggested_message = self.generate_smart_reply(target_email)
                    
                    return f"SUGGESTED REPLY to {sender_name}:\n\nSubject: Re: {email_subject}\n\nSuggested message: {suggested_message}\n\nTo send this reply, use: 'reply to {sender_name} {suggested_message}'\nOr modify the message: 'reply to {sender_name} Your custom message here'"
                
                # Get the email to reply to
                target_email = self.current_emails[email_index]
                from_addr = target_email['from']
                original_subject = target_email['subject']
                
                # Create reply subject
                if not original_subject.lower().startswith('re:'):
                    reply_subject = f"Re: {original_subject}"
                else:
                    reply_subject = original_subject
                
                # Send the reply
                resp = send_email(from_addr, reply_subject, reply_message)
                return f"Reply sent successfully to {from_addr}!\nSubject: {reply_subject}\nMessage: {reply_message}\nMessage ID: {resp.get('id', 'Unknown')}"
            
            elif command_lower.startswith("send email to "):
                # Handle both direct email addresses and references to stored emails
                remaining = command[13:].strip()
                
                # Check if it's a reference to a stored email
                email_refs = [
                    "first mail", "first email", "email 1", "mail 1", "the first", "1st email", "1st mail",
                    "second mail", "second email", "email 2", "mail 2", "the second", "2nd email", "2nd mail",
                    "third mail", "third email", "email 3", "mail 3", "the third", "3rd email", "3rd mail",
                    "fourth mail", "fourth email", "email 4", "mail 4", "the fourth", "4th email", "4th mail",
                    "fifth mail", "fifth email", "email 5", "mail 5", "the fifth", "5th email", "5th mail"
                ]
                
                if any(ref in command_lower for ref in email_refs):
                    if not self.current_emails:
                        return "ERROR: No emails loaded. Please run 'check emails' first."
                    
                    # Determine which email to send to
                    email_index = 0  # default to first
                    if any(ref in command_lower for ref in ["second", "2nd", "email 2", "mail 2"]):
                        email_index = 1
                    elif any(ref in command_lower for ref in ["third", "3rd", "email 3", "mail 3"]):
                        email_index = 2
                    elif any(ref in command_lower for ref in ["fourth", "4th", "email 4", "mail 4"]):
                        email_index = 3
                    elif any(ref in command_lower for ref in ["fifth", "5th", "email 5", "mail 5"]):
                        email_index = 4
                    
                    if email_index >= len(self.current_emails):
                        return f"ERROR: Email number {email_index + 1} not available. Only {len(self.current_emails)} emails loaded."
                    
                    # Extract the message from the command
                    message_start = 0
                    for ref in email_refs:
                        if ref in command_lower:
                            message_start = command_lower.find(ref) + len(ref)
                            break
                    
                    if message_start == 0:
                        return "ERROR: Could not parse message. Use: 'send email to the first mail [your message]'"
                    
                    message = command[message_start:].strip()
                    if not message:
                        return "ERROR: Please provide a message. Use: 'send email to the first mail [your message]'"
                    
                    # Get the target email
                    target_email = self.current_emails[email_index]
                    from_addr = target_email['from']
                    original_subject = target_email['subject']
                    
                    # Create subject
                    if not original_subject.lower().startswith('re:'):
                        reply_subject = f"Re: {original_subject}"
                    else:
                        reply_subject = original_subject
                    
                    # Send the email
                    resp = send_email(from_addr, reply_subject, message)
                    return f"Email sent successfully to {from_addr}!\nSubject: {reply_subject}\nMessage: {message}\nMessage ID: {resp.get('id', 'Unknown')}"
                
                # Handle direct email address format
                else:
                    parts = remaining.split(" ", 2)
                    if len(parts) >= 3:
                        to_addr, subject, body = parts
                        resp = send_email(to_addr, subject, body)
                        return f"Email sent successfully! Message ID: {resp.get('id', 'Unknown')}"
                    else:
                        return "ERROR: Invalid email format. Use: 'Send email to [address] [subject] [message]' or 'Send email to the first mail [message]'"
            
            else:
                # Check if it's a general sender mention (like "rughved", "pepperfry")
                if self.current_emails:
                    for email in self.current_emails:
                        sender_name = email.get('from', '').split('<')[0].strip().lower()
                        if sender_name in command_lower:
                            # Found a sender mention, show their emails
                            sender_emails = [e for e in self.current_emails if sender_name in e.get('from', '').lower()]
                            
                            response = f"Found {len(sender_emails)} email(s) from {sender_name}:\n\n"
                            for i, email in enumerate(sender_emails, 1):
                                response += f"{i}. **Subject:** {email['subject']}\n"
                                response += f"   **Preview:** {email['body'][:100]}...\n\n"
                            
                            response += f"To reply to {sender_name}, use: 'reply to {sender_name} [your message]'"
                            return response
                
                return "Available Gmail commands:\n- 'check emails' - List recent emails\n- 'reply to [number/name/sender] [message]' - Reply to specific email\n- 'send email to [address] [subject] [message]' - Send new email\n- 'send email to the first mail [message]' - Send to first email\n- 'send email to email 2 [message]' - Send to second email\n\nReply examples: 'reply to 1 message', 'reply to first email message', 'reply to message from pepperfry'"
                
        except Exception as e:
            return f"Gmail operation error: {str(e)}"
    
    def generate_smart_reply(self, email: dict) -> str:
        """Generate a smart reply suggestion based on email content"""
        try:
            subject = email.get('subject', '')
            body = email.get('body', '')
            sender = email.get('from', '')
            sender_name = sender.split('<')[0].strip()
            
            # Use AI to generate a contextual reply
            prompt = f"""
            Generate a brief, professional reply to this email:
            
            From: {sender}
            Subject: {subject}
            Content: {body[:500]}...
            
            Generate a short, appropriate reply (1-2 sentences). Be polite and professional.
            If it's spam, suggest declining politely.
            If it's a question, provide a helpful answer.
            If it's a notification, acknowledge receipt.
            
            Reply:
            """
            
            response = self.llm.invoke(prompt).content.strip()
            return response
            
        except Exception as e:
            # Fallback suggestions based on email type
            subject = email.get('subject', '').lower()
            if 'spam' in subject or 'sale' in subject or 'offer' in subject:
                return "Thank you for your email. I'm not interested in this offer at this time."
            elif 'question' in subject or '?' in email.get('body', ''):
                return "Thank you for your email. I'll get back to you with more information soon."
            else:
                return "Thank you for your email. I've received your message and will respond accordingly."
    
    def start_voice_mode(self):
        """Start continuous voice interaction"""
        if not VOICE_AVAILABLE:
            print("ERROR: Voice mode not available. Install voice components first.")
            return
        
        self.voice_enabled = True
        self.speak("Voice mode activated. I'm listening...")
        
        def voice_loop():
            while self.voice_enabled:
                text = self.listen_for_voice()
                if text and text != "Could not understand audio":
                    command = Command(
                        type=CommandType.VOICE,
                        content=text,
                        source="voice",
                        timestamp=datetime.now()
                    )
                    response = self.process_command(command)
                    self.speak(response)
        
        voice_thread = threading.Thread(target=voice_loop)
        voice_thread.daemon = True
        voice_thread.start()
    
    def stop_voice_mode(self):
        """Stop voice interaction"""
        self.voice_enabled = False
        if VOICE_AVAILABLE:
            self.speak("Voice mode deactivated.")
    
    def run_telegram_bot(self):
        """Run Telegram bot"""
        if TELEGRAM_AVAILABLE and self.telegram_bot:
            self.telegram_bot.polling(none_stop=True)
        else:
            logger.warning("Telegram not available or not configured")
    
    def chat_mode(self):
        """Interactive chat mode"""
        print("JARVIS AI Assistant - Chat Mode")
        print("=" * 50)
        print("Commands:")
        print("- 'voice on/off' - Toggle voice mode")
        print("- 'search [query]' - Google search")
        print("- 'read file [path]' - Read file")
        print("- 'write file [path] [content]' - Write file")
        print("- 'find files [pattern]' - Find files")
        print("- 'check emails' - Check Gmail")
        print("- 'quit' - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("JARVIS: Goodbye!")
                    break
                
                elif user_input.lower() == 'voice on':
                    self.start_voice_mode()
                    continue
                
                elif user_input.lower() == 'voice off':
                    self.stop_voice_mode()
                    continue
                
                elif user_input.startswith('search '):
                    command = Command(
                        type=CommandType.SEARCH,
                        content=user_input[7:],
                        source="chat",
                        timestamp=datetime.now()
                    )
                elif user_input.startswith(('read file ', 'write file ', 'find files ')):
                    command = Command(
                        type=CommandType.FILE,
                        content=user_input,
                        source="chat",
                        timestamp=datetime.now()
                    )
                elif user_input == 'check emails':
                    # Use existing email functionality
                    from agent import fetch_recent_emails
                    try:
                        service = get_service()
                        emails = fetch_recent_emails(service, max_results=5)
                        if emails:
                            print(f"\nFound {len(emails)} recent emails:")
                            for i, email in enumerate(emails, 1):
                                print(f"{i}. From: {email['from']}")
                                print(f"   Subject: {email['subject']}")
                                print(f"   Preview: {email['body'][:100]}...")
                        else:
                            print("No recent emails found.")
                    except Exception as e:
                        print(f"ERROR: Email error: {e}")
                    continue
                else:
                    command = Command(
                        type=CommandType.TEXT,
                        content=user_input,
                        source="chat",
                        timestamp=datetime.now()
                    )
                
                response = self.process_command(command)
                print(f"\nJARVIS: {response}")
                
            except KeyboardInterrupt:
                print("\nJARVIS: Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] Error: {e}")

def main():
    """Main entry point"""
    try:
        jarvis = JARVISAssistant()
        
        print("[STARTUP] Starting JARVIS AI Assistant...")
        print("Choose mode:")
        print("1. Chat Mode")
        print("2. Telegram Bot")
        print("3. Voice Mode")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            jarvis.chat_mode()
        elif choice == "2":
            jarvis.run_telegram_bot()
        elif choice == "3":
            jarvis.start_voice_mode()
            input("Press Enter to stop voice mode...")
            jarvis.stop_voice_mode()
        else:
            print("Invalid choice. Starting chat mode...")
            jarvis.chat_mode()
            
    except Exception as e:
        logger.error(f"JARVIS startup error: {e}")
        print(f"[ERROR] Error starting JARVIS: {e}")

if __name__ == "__main__":
    main()
