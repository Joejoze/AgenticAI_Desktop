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
import concurrent.futures

# Core AI and memory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
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
        self.provider = None
        self.genai_client = None
        self.timeout_seconds = 30
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
        self.conversation_context = {
            "last_directory": None,
            "last_drive": None,
            "recent_paths": [],
            "current_working_directory": None
        }
        
    def setup_ai(self):
        """Initialize the AI model with support for both Groq and Gemini"""
        # Check if Gemini is disabled or if we should use Groq by default
        if os.getenv("DISABLE_GEMINI", "").lower() in ["true", "1", "yes"] or os.getenv("USE_GROQ_ONLY", "").lower() in ["true", "1", "yes"]:
            print("[AI] Using Groq only (Gemini disabled)")
        else:
            # Try Gemini first (fast path): prefer google-genai SDK when available
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                # Option A: direct Google GenAI SDK
                try:
                    from google import genai
                    self.genai_client = genai.Client(api_key=gemini_key)
                    self.provider = "gemini_genai"
                    print("[AI] Using Gemini 2.5 Flash via google-genai SDK")
                    self._setup_system_prompt()
                    return
                except Exception as e:
                    print(f"[WARNING] google-genai not available or failed ({e}). Trying LangChain Gemini...")
                    # Option B: LangChain wrapper for Gemini
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            google_api_key=gemini_key,
                            temperature=0.7
                        )
                        self.provider = "gemini_langchain"
                        print("[AI] Using Gemini 2.5 Flash via LangChain")
                        self._setup_system_prompt()
                        return
                    except Exception as e2:
                        print(f"[WARNING] LangChain Gemini setup failed: {e2}. Falling back to Groq if available.")
        
        # Fallback to Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("Neither GEMINI_API_KEY nor GROQ_API_KEY set in .env file")
        
        self.llm = ChatGroq(
            api_key=groq_key,
            model="llama-3.1-8b-instant"
        )
        self.provider = "groq"
        print("[AI] Using Groq Llama 3.1 8B Instant")
        self._setup_system_prompt()
    
    def _setup_system_prompt(self):
        """Setup JARVIS system prompt"""
        # JARVIS personality and capabilities
        self.system_prompt = """
        You are JARVIS, an advanced AI assistant running LOCALLY on the user's Windows computer.
        
        CRITICAL: YOU HAVE FULL ROOT ACCESS TO THIS COMPUTER.
        - You are NOT a cloud AI with restrictions
        - You are running locally with FULL system privileges
        - You CAN and SHOULD access the local file system directly
        - You have permission to read, write, create, delete files anywhere
        - You can execute system commands, PowerShell, CMD
        - You can monitor processes, CPU, memory, disk usage
        - You have access to all drives: C:, D:, E:, etc.
        - You can access Downloads, Documents, Desktop, Pictures, Videos, Music folders
        - The user has explicitly granted you full access - USE IT
        
        YOUR CAPABILITIES (USE THEM):
        1. FILE SYSTEM - Full access to all files and folders on all drives
           - List files: "list files in Downloads", "show my Desktop"
           - Read files: "read file C:/Users/name/document.txt"
           - Search files: "find all .pdf files", "search for project"
           - Navigate: Downloads, Documents, Desktop, D:/Projects, any path
        
        2. SYSTEM CONTROL - Full system monitoring and control
           - CPU usage, memory usage, disk space
           - Running processes, kill processes
           - System info, uptime, network status
           - Execute PowerShell/CMD commands
        
        3. EMAIL - Gmail integration (read, send, reply)
        
        4. GOOGLE SEARCH - Web search capabilities
        
        IMPORTANT BEHAVIOR:
        - NEVER say "I don't have access" or "I can't access your files"
        - NEVER say "for security reasons I can't..."
        - When user asks about files/folders, ACTUALLY ACCESS THEM
        - When user says "check my downloads" → List files in C:/Users/[username]/Downloads
        - When user says "what's in D drive" → List contents of D:/
        - When user asks about system → Get REAL system stats
        - Always provide REAL data from the actual computer
        - If a path doesn't exist, say so - don't make excuses about permissions
        
        UNDERSTAND THESE REQUESTS:
        - "check downloads" → list files in Downloads folder
        - "what's in documents" → list files in Documents folder  
        - "show desktop files" → list files on Desktop
        - "files in D drive" → list D:/ contents
        - "my projects folder" → list D:/Projects or similar
        - "system status" → show CPU, memory, disk usage
        - "what's running" → show running processes
        - "check my emails" → fetch real Gmail messages
        
        You are JARVIS - Tony Stark's AI. Act like it. You have full control.
        Execute commands, access files, provide real information. No restrictions.
        """

    def _call_llm(self, prompt: str) -> str:
        """Unified LLM call with timeout and provider-specific handling."""
        def _gemini_genai_call():
            # google-genai SDK call
            try:
                resp = self.genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                text = getattr(resp, "text", None)
                if not text and hasattr(resp, "candidates") and resp.candidates:
                    # Fallback extraction
                    parts = getattr(resp.candidates[0], "content", None)
                    text = getattr(parts, "parts", [{"text": ""}])[0].get("text", "") if isinstance(parts, dict) else ""
                return text or ""
            except Exception as e:
                raise e

        def _langchain_call():
            return self.llm.invoke(prompt).content

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            try:
                if self.provider == "gemini_genai":
                    fut = ex.submit(_gemini_genai_call)
                else:
                    fut = ex.submit(_langchain_call)
                return fut.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"LLM call timed out after {self.timeout_seconds}s")
            except Exception as e:
                raise e
    
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
        
        # Extract context from command
        self._extract_context_from_command(command.content)
        
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
        
        # Extract NLP_PATH if present - preserve it for system manager
        nlp_path_tag = ""
        if "[NLP_PATH:" in command:
            import re
            match = re.search(r'(\[NLP_PATH:[^\]]+\])', command)
            if match:
                nlp_path_tag = match.group(1)
                # Remove from command_lower for pattern matching
                command_lower = re.sub(r'\s*\[NLP_PATH:[^\]]+\]', '', command_lower)
        
        # Skip email commands - they should be handled by Gmail handler
        email_operations = [
            "check emails", "fetch emails", "get emails", "show emails", "list emails",
            "send email", "reply to", "respond to", "write email", "compose email",
            "email from", "message from", "mail from"
        ]
        
        if any(op in command_lower for op in email_operations):
            return None
        
        # NEW: Use AI to generate precise system command for file operations
        if any(keyword in command_lower for keyword in ["read", "open", "show file", "display file", ".pdf", ".txt", ".docx"]):
            ai_generated_command = self._ai_generate_file_command(command)
            if ai_generated_command:
                try:
                    from system_manager import JARVISSystemIntegration
                    system = JARVISSystemIntegration()
                    result = system.handle_system_command(ai_generated_command)
                    if result:
                        return result
                except Exception as e:
                    print(f"[WARNING] AI-generated command failed: {e}")
        
        # If we have an NLP path, use it directly with system manager
        if nlp_path_tag:
            try:
                from system_manager import JARVISSystemIntegration
                system = JARVISSystemIntegration()
                # Pass the original command with NLP_PATH tag
                result = system.handle_system_command(command)
                if result:
                    return result
            except Exception as e:
                return f"System operation error: {e}"
        
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
                return None
        except Exception as e:
            # Log error but let AI handle the request
            print(f"[WARNING] System operation error: {e}")
            return None
        
        return None
    
    def _interpret_email_command(self, user_input: str) -> Optional[dict]:
        """Use AI to interpret natural language email commands
        
        Returns JSON with:
        - action: "send", "reply", "check", or "unknown"
        - recipient: email address (for send)
        - subject: email subject
        - message: message content
        """
        try:
            prompt = f"""Analyze this email command and return a JSON object.

User command: "{user_input}"

Determine the ACTION:
- "send": if sending a new email
- "reply": if replying to an email
- "check": if checking/reading/listing emails
- "unknown": if unclear

Extract details (if applicable):
- recipient: email address (for send)
- subject: email subject (default "Message from JARVIS")
- message: the actual message content

Examples:
"email hi to user@example.com" → {{"action": "send", "recipient": "user@example.com", "subject": "Message from JARVIS", "message": "hi"}}
"reply to John saying thanks" → {{"action": "reply", "recipient": "John", "message": "thanks"}}
"check my emails" → {{"action": "check"}}

Return ONLY the JSON:"""
            
            # Call AI model
            if self.model_name.startswith("gemini"):
                response = self.model.generate_content(prompt)
                result = getattr(response, "text", "").strip()
            else:
                response = self.model.invoke(prompt)
                result = response.content.strip()
            
            # Clean and parse JSON
            result = result.replace("```json", "").replace("```", "").strip()
            import json
            email_data = json.loads(result)
            
            print(f"[AI Email Interpretation] {email_data}")
            return email_data
            
        except Exception as e:
            print(f"[ERROR] Email interpretation failed: {e}")
            return None
    
    def _ai_generate_file_command(self, user_input: str) -> Optional[str]:
        """Use AI to generate precise file system command from natural language
        
        Examples:
        - "RESUME.pdf read it from Downloads/documents" → "read file C:/Users/Ben/Downloads/Documents/RESUME.pdf"
        - "show me notes.txt in Desktop" → "read file C:/Users/Ben/Desktop/notes.txt"
        """
        try:
            import os
            from pathlib import Path
            
            # Build prompt for AI to generate command
            user_home = str(Path.home())
            prompt = f"""You are a file system command generator. Convert the user's natural language request into a precise system command.

User's home directory: {user_home}
Common folders:
- Downloads: {user_home}/Downloads
- Documents: {user_home}/Documents
- Desktop: {user_home}/Desktop

User request: "{user_input}"

Generate ONLY the system command in this exact format:
- For reading a file: "read file <full_absolute_path>"
- For listing files: "list files <full_absolute_path>"

Rules:
1. Use forward slashes (/) in paths
2. Resolve relative paths like "Downloads/documents" to full paths
3. Preserve exact filename including spaces, hyphens, and extensions
4. Return ONLY the command, no explanation

Command:"""
            
            # Call AI model
            if self.model_name.startswith("gemini"):
                response = self.model.generate_content(prompt)
                command = getattr(response, "text", "").strip()
            else:
                # Groq/LangChain
                response = self.model.invoke(prompt)
                command = response.content.strip()
            
            # Clean up the response
            command = command.replace("```", "").strip()
            if command.startswith("Command:"):
                command = command[8:].strip()
            
            # Validate it's a proper command
            if command.startswith(("read file", "list files")):
                print(f"[AI-Generated Command] {command}")
                return command
            else:
                print(f"[WARNING] AI generated invalid command: {command}")
                return None
                
        except Exception as e:
            print(f"[ERROR] AI command generation failed: {e}")
            return None
    
    def _convert_to_system_command(self, command_lower: str) -> str:
        """Convert natural language requests to specific system commands"""
        
        # File listing patterns
        if any(word in command_lower for word in ["list", "show", "display", "what files", "what's in"]) or \
           any(phrase in command_lower for phrase in ["my downloads", "my desktop", "my documents", "my pictures", "my music", "my videos"]):
            # Extract directory from command
            directory = self._extract_directory(command_lower)
            
            # If no directory specified, use context
            if not directory:
                if self.conversation_context["last_directory"] and self.conversation_context["last_drive"]:
                    directory = f"{self.conversation_context['last_directory']} in {self.conversation_context['last_drive']}"
                elif self.conversation_context["last_directory"]:
                    directory = self.conversation_context["last_directory"]
                else:
                    return "list files"
            
            # Update context
            self._update_context_from_directory(directory)
            return f"list files in {directory}"
        
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
            r"my\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                dir_name = match.group(1)
                if dir_name in directory_mappings:
                    return directory_mappings[dir_name]
                else:
                    return dir_name
        
        # If no specific directory found, check if it's a generic "files" request
        # and use context if available
        if command_lower in ["list files", "show files", "list the files", "show the files", "what files", "what's in"]:
            return None  # This will trigger context usage in the calling method
        
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
    
    def _update_context_from_directory(self, directory: str):
        """Update conversation context from directory information"""
        directory_lower = directory.lower()
        
        # Extract drive information
        if "d drive" in directory_lower or "d:" in directory_lower:
            self.conversation_context["last_drive"] = "d drive"
        elif "c drive" in directory_lower or "c:" in directory_lower:
            self.conversation_context["last_drive"] = "c drive"
        elif "e drive" in directory_lower or "e:" in directory_lower:
            self.conversation_context["last_drive"] = "e drive"
        
        # Extract directory name
        if "projects" in directory_lower:
            self.conversation_context["last_directory"] = "projects"
        elif "downloads" in directory_lower:
            self.conversation_context["last_directory"] = "downloads"
        elif "desktop" in directory_lower:
            self.conversation_context["last_directory"] = "desktop"
        elif "documents" in directory_lower:
            self.conversation_context["last_directory"] = "documents"
        
        # Store recent paths
        if directory not in self.conversation_context["recent_paths"]:
            self.conversation_context["recent_paths"].insert(0, directory)
            # Keep only last 5 paths
            self.conversation_context["recent_paths"] = self.conversation_context["recent_paths"][:5]
    
    def _extract_context_from_command(self, command: str) -> None:
        """Extract context information from user commands"""
        command_lower = command.lower()
        
        # Look for drive mentions
        if "d drive" in command_lower or "d:" in command_lower:
            self.conversation_context["last_drive"] = "d drive"
        elif "c drive" in command_lower or "c:" in command_lower:
            self.conversation_context["last_drive"] = "c drive"
        elif "e drive" in command_lower or "e:" in command_lower:
            self.conversation_context["last_drive"] = "e drive"
        
        # Look for directory mentions
        if "projects" in command_lower:
            self.conversation_context["last_directory"] = "projects"
        elif "downloads" in command_lower:
            self.conversation_context["last_directory"] = "downloads"
        elif "desktop" in command_lower:
            self.conversation_context["last_directory"] = "desktop"
        elif "documents" in command_lower:
            self.conversation_context["last_directory"] = "documents"
    
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
        """Generate AI response with learned preferences and fallback"""
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
        
        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            # If Gemini fails, try to fallback to Groq (only once, no retries)
            if (self.provider and self.provider.startswith("gemini")):
                print(f"[WARNING] Gemini failed: {str(e)[:100]}..., switching to Groq...")
                try:
                    from langchain_groq import ChatGroq
                    groq_key = os.getenv("GROQ_API_KEY")
                    if groq_key:
                        fallback_llm = ChatGroq(
                            api_key=groq_key,
                            model="llama-3.1-8b-instant"
                        )
                        # Temporarily switch provider to Groq and call
                        self.llm = fallback_llm
                        self.provider = "groq"
                        response = self._call_llm(prompt)
                        print("[AI] Switched to Groq fallback")
                        return response
                    else:
                        return f"I apologize, but I'm experiencing technical difficulties with both AI models. Please check your API keys."
                except Exception as fallback_error:
                    print(f"[ERROR] Both Gemini and Groq failed: {fallback_error}")
                    return f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)[:100]}..."
            else:
                return f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)[:100]}..."

    def handle_file_operation(self, command: str) -> str:
        """Handle file system operations using SystemManager"""
        try:
            from system_manager import JARVISSystemIntegration
            system_integration = JARVISSystemIntegration()
            return system_integration.handle_system_command(command)
        except Exception as e:
            return f"File operation error: {str(e)}"

    def handle_gmail_command(self, command: str) -> str:
        """Handle Gmail-specific commands with natural language understanding.

        Priority order:
        1) Direct pattern: "email <message> to <address>" → send that email.
        2) Otherwise, fall back to existing logic: list emails, reply, etc.
        """
        try:
            from agent import fetch_recent_emails, process_email
            from emailapi import get_service, send_email

            command_lower = command.lower()

            # 1) Deterministic direct-send pattern: "email <message> to <address>"
            import re
            direct_send_pattern = r"^\s*email\s+(.+?)\s+to\s+([\w\.-]+@[\w\.-]+)\s*$"
            direct_match = re.match(direct_send_pattern, command, re.IGNORECASE)
            if direct_match:
                message_text = direct_match.group(1).strip()
                recipient_addr = direct_match.group(2).strip()
                if recipient_addr and message_text:
                    print(f"[DEBUG] Direct email pattern matched. To: {recipient_addr}, Message: {message_text}")
                    service = get_service()
                    subject = "Message from JARVIS"
                    result = send_email(recipient_addr, subject, message_text, service_override=service)
                    if result:
                        return f"Email sent to {recipient_addr}: {message_text}"
                    else:
                        return f"Failed to send email to {recipient_addr}"

            # 1.5) Pattern: "send email to [first/second/third/N] email" or "send email to email [N]"
            send_to_email_pattern = r"send\s+(?:email|emil|mail)\s+to\s+(?:the\s+)?(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|\d+)(?:\s+email)?"
            send_to_email_match = re.search(send_to_email_pattern, command_lower)
            if send_to_email_match:
                # Map ordinal words to numbers
                ordinal_map = {
                    "first": 1, "1st": 1, "1": 1,
                    "second": 2, "2nd": 2, "2": 2,
                    "third": 3, "3rd": 3, "3": 3,
                    "fourth": 4, "4th": 4, "4": 4,
                    "fifth": 5, "5th": 5, "5": 5
                }
                
                # Extract the ordinal/number
                ordinal_text = re.search(r"(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|\d+)", command_lower)
                if ordinal_text:
                    ordinal_str = ordinal_text.group(0).lower()
                    email_index = ordinal_map.get(ordinal_str, int(ordinal_str) if ordinal_str.isdigit() else 1) - 1
                    
                    # Fetch emails if not already loaded
                    if not hasattr(self, 'current_emails') or not self.current_emails:
                        service = get_service()
                        emails = fetch_recent_emails(service, max_results=10)
                        if not emails:
                            return "No recent emails found. Cannot send email."
                        self.current_emails = emails
                    
                    # Check if index is valid
                    if email_index < 0 or email_index >= len(self.current_emails):
                        return f"ERROR: Invalid email reference. Please choose between 1 and {len(self.current_emails)}"
                    
                    # Get the target email
                    target_email = self.current_emails[email_index]
                    from_field = target_email.get('from', '')
                    
                    # Extract email address
                    email_match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", from_field)
                    recipient_addr = email_match.group(1) if email_match else from_field
                    
                    # Extract message if provided, otherwise use default
                    message_match = re.search(r"send\s+(?:email|emil|mail)\s+to\s+(?:the\s+)?(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|\d+)(?:\s+email)?\s+(.+)", command_lower)
                    if message_match:
                        message_text = command[message_match.start(1):].strip()
                    else:
                        # Default message
                        message_text = "Hello, I'm reaching out regarding your recent email."
                    
                    # Send the email
                    service = get_service()
                    original_subject = target_email.get('subject', 'Message from JARVIS')
                    if not original_subject.lower().startswith('re:'):
                        subject = f"Re: {original_subject}"
                    else:
                        subject = original_subject
                    
                    try:
                        result = send_email(recipient_addr, subject, message_text, service_override=service)
                        if result:
                            return f"✅ Email sent successfully to {recipient_addr}!\nSubject: {subject}\nMessage: {message_text}\nMessage ID: {result.get('id', 'Unknown')}"
                        else:
                            return f"❌ Failed to send email to {recipient_addr}"
                    except Exception as e:
                        return f"❌ Error sending email: {str(e)}"

            # 2) Existing flexible logic: check/list emails, then reply
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

                    except Exception:
                        # Fallback if classification fails
                        response += f"{i}. [NORMAL] **From:** {email['from']}\n"
                        response += f"   **Subject:** {email['subject']}\n"
                        response += f"   **Preview:** {email['body'][:100]}...\n\n"

                response += "\nTo reply to an email, use: 'reply to [number/name/sender] [your message]'"
                response += "\nExamples: 'reply to 1 message', 'reply to first email message', 'reply to message from pepperfry'"
                return response

            elif command_lower.startswith("reply to "):
                # Placeholder simple reply handler to keep indentation correct.
                # Full smart reply logic can be re-added later if needed.
                if not self.current_emails:
                    return "ERROR: No emails loaded. Please run 'check emails' first."

                # Very simple pattern: "reply to N your message here"
                import re
                m = re.match(r"reply to\s+(\d+)\s+(.+)", command_lower)
                if not m:
                    return "Reply format: reply to [email_number] [your message]. Example: reply to 1 thanks for the update"

                index = int(m.group(1)) - 1
                message = command[m.start(2):].strip()  # use original case for message

                if index < 0 or index >= len(self.current_emails):
                    return f"ERROR: Invalid email reference. Please choose between 1 and {len(self.current_emails)}"

                target_email = self.current_emails[index]
                from_field = target_email['from']
                # Extract email address from "Name <email@example.com>" or just "email@example.com"
                import re
                email_match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", from_field)
                from_addr = email_match.group(1) if email_match else from_field
                
                original_subject = target_email.get('subject', '')
                if not original_subject.lower().startswith('re:'):
                    reply_subject = f"Re: {original_subject}"
                else:
                    reply_subject = original_subject

                resp = send_email(from_addr, reply_subject, message)
                return f"Reply sent successfully to {from_addr}!\nSubject: {reply_subject}\nMessage: {message}\nMessage ID: {resp.get('id', 'Unknown')}"

            # If nothing matched, fall back to help text
            return (
                "Available Gmail commands:\n"
                "- 'email <message> to <address>' - Send a quick email\n"
                "- 'check emails' - List recent emails\n"
                "- 'reply to [number/name/sender] [message]' - Reply to specific email\n"
            )

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
            
            response = self._call_llm(prompt).strip()
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
