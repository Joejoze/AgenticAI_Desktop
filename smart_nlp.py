"""
Smart NLP Pipeline for JARVIS
Implements:
1. Text Normalization
2. Semantic Chunking
3. Intent Classification
4. Entity Extraction
5. Context Management
6. Query Rewriting
"""

import re
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures

@dataclass
class NLPResult:
    """Result from NLP processing"""
    original_text: str
    normalized_text: str
    rewritten_query: str
    intent: str
    confidence: float
    entities: Dict[str, List[str]]
    path_intent: Optional[str]
    action: str
    context_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextNormalizer:
    """Layer 1: Text Normalization"""
    
    def __init__(self):
        # Slang/abbreviation mappings
        self.slang_map = {
            "btw": "by the way",
            "idk": "i do not know",
            "tbh": "to be honest",
            "imo": "in my opinion",
            "imho": "in my humble opinion",
            "fyi": "for your information",
            "asap": "as soon as possible",
            "aka": "also known as",
            "etc": "et cetera",
            "eg": "for example",
            "ie": "that is",
            "w/": "with",
            "w/o": "without",
            "b/c": "because",
            "pls": "please",
            "plz": "please",
            "thx": "thanks",
            "ty": "thank you",
            "u": "you",
            "ur": "your",
            "r": "are",
            "y": "why",
            "n": "and",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "do not know",
            "ain't": "is not",
            "y'all": "you all",
            "c'mon": "come on",
            "info": "information",
            "docs": "documents",
            "pics": "pictures",
            "vids": "videos",
            "app": "application",
            "apps": "applications",
            "config": "configuration",
            "configs": "configurations",
            "dir": "directory",
            "dirs": "directories",
            "exec": "execute",
            "cmd": "command",
            "sys": "system",
            "mem": "memory",
            "proc": "process",
            "procs": "processes",
        }
        
        # Prototype phrases for embedding-based intent similarity
        # These are used when rule-based confidence is low
        self.intent_prototypes: Dict[str, List[str]] = {
            "list_files": [
                "list files in a folder",
                "show me what is inside this directory",
                "display the files in downloads",
            ],
            "read_file": [
                "read the contents of a file",
                "open this pdf and show what is inside",
                "view this document",
            ],
            "search_files": [
                "search for files by name",
                "find a file matching this pattern",
                "look for a document on disk",
            ],
            "execute_command": [
                "run a terminal command",
                "execute a powershell script",
                "run a cmd command",
            ],
            "system_status": [
                "show system health and status",
                "check cpu and memory usage",
                "tell me how my computer is doing",
            ],
            "question": [
                "ask a general question",
                "user is asking for an explanation",
                "the user wants information, not actions",
            ],
            "help": [
                "user is asking for help or guidance",
                "explain how to do something",
            ],
        }
        
        # Contractions
        self.contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "how's": "how is",
            "there's": "there is",
            "here's": "here is",
            "let's": "let us",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "we'd": "we would",
            "they'd": "they would",
        }
        
        # Common typos/misspellings
        self.typo_fixes = {
            "teh": "the",
            "hte": "the",
            "taht": "that",
            "whcih": "which",
            "wiht": "with",
            "adn": "and",
            "thnk": "think",
            "recieve": "receive",
            "seperate": "separate",
            "occured": "occurred",
            "untill": "until",
            "wich": "which",
            "becuase": "because",
            "beacuse": "because",
            "definately": "definitely",
            "occassion": "occasion",
            "accomodate": "accommodate",
            "acheive": "achieve",
            "accross": "across",
            "agressive": "aggressive",
            "apparant": "apparent",
            "calender": "calendar",
            "collegue": "colleague",
            "concious": "conscious",
            "enviroment": "environment",
            "existance": "existence",
            "goverment": "government",
            "harrass": "harass",
            "immediatly": "immediately",
            "independant": "independent",
            "knowlege": "knowledge",
            "liason": "liaison",
            "millenium": "millennium",
            "neccessary": "necessary",
            "noticable": "noticeable",
            "occurence": "occurrence",
            "persistant": "persistent",
            "posession": "possession",
            "prefered": "preferred",
            "priviledge": "privilege",
            "publically": "publicly",
            "recomend": "recommend",
            "refered": "referred",
            "relevent": "relevant",
            "rythm": "rhythm",
            "succesful": "successful",
            "suprise": "surprise",
            "tommorow": "tomorrow",
            "truely": "truly",
            "wierd": "weird",
        }
    
    def normalize(self, text: str) -> str:
        """Full text normalization pipeline"""
        if not text:
            return ""
        
        # Step 1: Basic cleanup
        text = self._remove_html(text)
        text = self._remove_emojis(text)
        text = self._fix_spacing(text)
        
        # Step 2: Lowercase (preserve paths and proper nouns temporarily)
        text_lower = text.lower()
        
        # Step 3: Fix typos
        text_lower = self._fix_typos(text_lower)
        
        # Step 4: Expand contractions
        text_lower = self._expand_contractions(text_lower)
        
        # Step 5: Replace slang
        text_lower = self._replace_slang(text_lower)
        
        # Step 6: Normalize punctuation
        text_lower = self._normalize_punctuation(text_lower)
        
        return text_lower.strip()
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags"""
        return re.sub(r'<[^>]+>', '', text)
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis but keep text"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues"""
        # Multiple spaces to single
        text = re.sub(r'\s+', ' ', text)
        # Fix broken words (e.g., "down loads" -> "downloads")
        broken_words = {
            r'down\s+loads?': 'downloads',
            r'desk\s+top': 'desktop',
            r'docu\s*ments?': 'documents',
            r'pic\s*tures?': 'pictures',
            r'power\s*shell': 'powershell',
            r'file\s*system': 'filesystem',
            r'data\s*base': 'database',
        }
        for pattern, replacement in broken_words.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _fix_typos(self, text: str) -> str:
        """Fix common typos"""
        words = text.split()
        fixed_words = []
        for word in words:
            # Check if word (without punctuation) is a typo
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.typo_fixes:
                word = word.replace(clean_word, self.typo_fixes[clean_word])
            fixed_words.append(word)
        return ' '.join(fixed_words)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def _replace_slang(self, text: str) -> str:
        """Replace slang with full forms"""
        words = text.split()
        replaced_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if clean_word in self.slang_map:
                word = word.lower().replace(clean_word, self.slang_map[clean_word])
            replaced_words.append(word)
        return ' '.join(replaced_words)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation"""
        # Remove excessive punctuation
        text = re.sub(r'[!?]{2,}', '?', text)
        text = re.sub(r'\.{2,}', '.', text)
        # Ensure space after punctuation
        text = re.sub(r'([.!?,;:])([^\s\d])', r'\1 \2', text)
        return text


class IntentClassifier:
    """Layer 4: Intent Classification"""
    
    def __init__(self):
        # Define intent patterns with keywords and phrases
        self.intent_patterns = {
            "list_files": {
                "keywords": ["list", "show", "display", "what files", "what is in", "contents of", "files in", "folder contents"],
                "patterns": [r"(?:list|show|display|get|check)\s+(?:files|folders|contents)", r"what(?:'s| is) in"],
                "weight": 1.0
            },
            "read_file": {
                "keywords": ["read", "open", "view", "content of", "show content", "cat", "display file"],
                "patterns": [r"(?:read|open|view|show)\s+(?:file|document|content)"],
                "weight": 1.0
            },
            "search_files": {
                "keywords": ["search", "find", "locate", "look for", "where is", "search for"],
                "patterns": [r"(?:search|find|locate|look for)\s+(?:file|folder|document)?"],
                "weight": 1.0
            },
            "create_file": {
                "keywords": ["create", "make", "new file", "write file", "touch"],
                "patterns": [r"(?:create|make|write|new)\s+(?:file|folder|document)"],
                "weight": 1.0
            },
            "delete_file": {
                "keywords": ["delete", "remove", "rm", "erase", "trash"],
                "patterns": [r"(?:delete|remove|erase)\s+(?:file|folder|document)?"],
                "weight": 1.0
            },
            "system_status": {
                "keywords": ["system status", "cpu", "memory", "ram", "disk", "processes", "running", "performance", "health"],
                "patterns": [r"(?:system|computer|pc)\s+(?:status|health|info)", r"(?:cpu|memory|ram|disk)\s+(?:usage|status)?"],
                "weight": 1.0
            },
            "execute_command": {
                "keywords": ["run", "execute", "command", "terminal", "shell", "powershell", "cmd", "ping", "script"],
                "patterns": [r"(?:run|execute)\s+(?:command|script)?", r"(?:in|using|via)\s+(?:terminal|shell|cmd|powershell)"],
                "weight": 1.0
            },
            "email_operation": {
                "keywords": ["email", "mail", "inbox", "send", "reply", "compose", "gmail"],
                "patterns": [r"(?:check|read|send|reply|compose)\s+(?:email|mail)", r"(?:email|mail)\s+(?:from|to)"],
                "weight": 1.0
            },
            "web_search": {
                "keywords": ["search", "google", "look up", "find online", "research", "web search"],
                "patterns": [r"(?:google|search|look up)\s+(?:for|about)?", r"(?:search|find)\s+(?:online|on the web)"],
                "weight": 0.8
            },
            "question": {
                "keywords": ["what", "why", "how", "when", "where", "who", "explain", "tell me", "describe"],
                "patterns": [r"^(?:what|why|how|when|where|who)\s+", r"(?:explain|tell me|describe)\s+"],
                "weight": 0.7
            },
            "help": {
                "keywords": ["help", "assist", "support", "how to", "guide", "tutorial"],
                "patterns": [r"(?:help|assist)\s+(?:me|with)?", r"how\s+(?:do|can|to)"],
                "weight": 0.6
            },
            "navigate": {
                "keywords": ["go to", "navigate", "open folder", "change directory", "cd"],
                "patterns": [r"(?:go|navigate|cd)\s+(?:to|into)?"],
                "weight": 0.9
            },
        }
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the text"""
        text_lower = text.lower()
        scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    score += 0.3 * config["weight"]
            
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    score += 0.5 * config["weight"]
            
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return "general_query", 0.3
        
        # Get highest scoring intent
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent], 1.0)
        
        return best_intent, confidence


class EmbeddingEngine:
    """Lightweight embedding engine using Gemini embeddings when available.

    This is intentionally simple: it provides cosine-like similarity over
    small sets of texts for routing and intent refinement. If embeddings
    are not available, it safely degrades to keyword overlap.
    """

    def __init__(self):
        self.client = None
        self.model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        self.available = False
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=gemini_key)
                self.available = True
                print(f"[EmbeddingEngine] Using Gemini embeddings: {self.model_name}")
            except Exception as e:
                print(f"[EmbeddingEngine WARNING] Failed to init Gemini embeddings: {e}")
                self.client = None
                self.available = False

    def embed(self, text: str) -> Optional[List[float]]:
        """Get an embedding vector for text. Returns None on failure."""
        text = (text or "").strip()
        if not text:
            return None
        if not self.available or not self.client:
            return None
        try:
            resp = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            # google-genai returns an object with .embeddings[0].values
            emb = getattr(resp, "embeddings", None)
            if not emb:
                return None
            values = getattr(emb[0], "values", None)
            return list(values) if values is not None else None
        except Exception as e:
            print(f"[EmbeddingEngine WARNING] embed failed: {e}")
            return None

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    @staticmethod
    def _norm(a: List[float]) -> float:
        return float(sum(x * x for x in a)) ** 0.5

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors, returns 0.0-1.0."""
        if not a or not b:
            return 0.0
        denom = self._norm(a) * self._norm(b)
        if denom == 0:
            return 0.0
        return max(min(self._dot(a, b) / denom, 1.0), -1.0)

    def similarity_texts(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Return candidates with similarity scores for a query.

        Falls back to simple token overlap if embeddings are not available.
        """
        if not candidates:
            return []

        # Fast path: embeddings
        q_emb = self.embed(query)
        if q_emb is not None:
            results: List[Tuple[str, float]] = []
            for c in candidates:
                c_emb = self.embed(c)
                if c_emb is None:
                    continue
                sim = self.similarity(q_emb, c_emb)
                results.append((c, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        # Fallback: simple token overlap similarity
        q_tokens = set(query.lower().split())
        results: List[Tuple[str, float]] = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            if not q_tokens or not c_tokens:
                sim = 0.0
            else:
                inter = len(q_tokens & c_tokens)
                union = len(q_tokens | c_tokens)
                sim = inter / union if union else 0.0
            results.append((c, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class EntityExtractor:
    """Layer 3: Entity Extraction with spaCy support"""
    
    def __init__(self):
        self.user_home = str(Path.home())
        self.username = os.getenv("USERNAME", os.getenv("USER", "user"))
        
        # Try to load spaCy for better NLP
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("[EntityExtractor] spaCy loaded successfully")
            except OSError:
                print("[EntityExtractor] spaCy model not found, using regex fallback")
        except ImportError:
            print("[EntityExtractor] spaCy not installed, using regex fallback")
        
        # Common file extensions
        self.file_extensions = {
            'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt',  # Documents
            'xls', 'xlsx', 'csv', 'ods',  # Spreadsheets
            'ppt', 'pptx', 'odp',  # Presentations
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp', 'ico',  # Images
            'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a',  # Audio
            'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm',  # Video
            'zip', 'rar', '7z', 'tar', 'gz', 'bz2',  # Archives
            'py', 'js', 'ts', 'html', 'css', 'json', 'xml', 'yaml', 'yml',  # Code
            'exe', 'msi', 'bat', 'sh', 'ps1',  # Executables
            'iso', 'img', 'dmg',  # Disk images
        }
        
        # Build dynamic folder knowledge
        self.known_folders = self._discover_folders()
    
    def _discover_folders(self) -> Dict[str, str]:
        """Dynamically discover folders on the system"""
        folders = {}
        
        # Standard user folders
        user_home = Path.home()
        standard = ["Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos", "AppData"]
        for folder in standard:
            path = user_home / folder
            if path.exists():
                folders[folder.lower()] = str(path)
        
        # Scan drives
        for drive in ["C:/", "D:/", "E:/"]:
            drive_path = Path(drive)
            if drive_path.exists():
                try:
                    for item in drive_path.iterdir():
                        if item.is_dir() and not item.name.startswith(('.', '$', 'Windows', 'Program')):
                            folders[item.name.lower()] = str(item)
                except PermissionError:
                    pass
        
        # Scan user home
        try:
            for item in user_home.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    folders[item.name.lower()] = str(item)
        except PermissionError:
            pass
        
        return folders
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            "paths": [],
            "files": [],
            "folders": [],
            "commands": [],
            "urls": [],
            "emails": [],
            "numbers": [],
            "dates": [],
            "spacy_entities": [],
        }
        
        text_lower = text.lower()
        
        # Use spaCy if available for better entity extraction
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities["spacy_entities"].append({"text": ent.text, "label": ent.label_})
        
        # Extract explicit paths (C:/..., D:/..., /home/...)
        path_pattern = r'[A-Za-z]:[/\\][\w\s/\\.-]*|/[\w/.-]+'
        paths = re.findall(path_pattern, text)
        entities["paths"].extend(paths)
        
        # IMPROVED: Extract full file names with extensions (handles spaces and special chars)
        # Pattern: anything ending with .extension
        ext_pattern = '|'.join(self.file_extensions)
        
        # Method 1: Find filename.ext pattern (simple names)
        simple_file_pattern = rf'\b[\w-]+\.(?:{ext_pattern})\b'
        simple_files = re.findall(simple_file_pattern, text, re.IGNORECASE)
        entities["files"].extend(simple_files)
        
        # Method 2: Find "Name With Spaces (1).ext" pattern
        # Look for text ending with .extension
        complex_file_pattern = rf'([A-Za-z][\w\s\-\(\)_.,]+\.(?:{ext_pattern}))'
        complex_files = re.findall(complex_file_pattern, text, re.IGNORECASE)
        for f in complex_files:
            f = f.strip()
            if f and f not in entities["files"]:
                entities["files"].append(f)
        
        # Method 3: If text contains a known extension, try to extract the full filename
        for ext in self.file_extensions:
            if f'.{ext}' in text.lower():
                # Find everything before .ext that looks like a filename
                pattern = rf'([\w][\w\s\-\(\)_.,]*\.{ext})'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for m in matches:
                    m = m.strip()
                    if m and m not in entities["files"]:
                        entities["files"].append(m)
        
        # Extract folder references
        for folder_name, folder_path in self.known_folders.items():
            if folder_name in text_lower:
                entities["folders"].append({"name": folder_name, "path": folder_path})
        
        # Extract command-like patterns
        cmd_pattern = r'\b(?:ping|tracert|ipconfig|netstat|dir|ls|cd|mkdir|rmdir|copy|move|del|rm|cat|grep|find|ps|kill)\b'
        commands = re.findall(cmd_pattern, text_lower)
        entities["commands"].extend(commands)
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        entities["urls"].extend(urls)
        
        # Extract emails
        email_pattern = r'[\w.-]+@[\w.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        entities["emails"].extend(emails)
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        entities["numbers"].extend(numbers)
        
        return entities
    
    def extract_filename(self, text: str) -> Optional[str]:
        """Extract a filename from text, preserving spaces and special characters"""
        # Build extension pattern
        ext_pattern = '|'.join(self.file_extensions)
        
        # Common action words that should NOT be part of filename
        action_words = {'read', 'open', 'view', 'show', 'display', 'check', 'find', 'search', 
                        'delete', 'remove', 'copy', 'move', 'here', 'please', 'can', 'you',
                        'the', 'a', 'an', 'this', 'that', 'my', 'from', 'in', 'at', 'to'}
        
        # Try to find filename with extension
        # Pattern matches: "Name With Spaces (1).pdf" or "simple.txt"
        patterns = [
            rf'([A-Za-z][\w\s\-\(\)_.,\'\"]+\.(?:{ext_pattern}))',  # Complex names
            rf'([\w-]+\.(?:{ext_pattern}))',  # Simple names
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Get the longest match
                filename = max(matches, key=len).strip()
                
                # Clean up: remove leading action words
                words = filename.split()
                cleaned_words = []
                started = False
                for word in words:
                    # Skip leading action words
                    if not started and word.lower() in action_words:
                        continue
                    started = True
                    cleaned_words.append(word)
                
                if cleaned_words:
                    return ' '.join(cleaned_words)
                return filename
        
        return None
    
    def extract_path_intent(self, text: str) -> Optional[str]:
        """Extract the intended file path from natural language
        
        Handles:
        - "filename.pdf from Downloads/documents" → full path to file
        - "filename.pdf from C:/Users/Ben/Documents" → absolute path
        - "documents in downloads" → folder path
        - "C:/path/to/folder" → any absolute path
        - "Downloads/documents/subfolder" → nested relative paths
        """
        text_lower = text.lower()
        
        # NEW Pattern 0: Handle "filename from path" or "filename in path"
        # Supports: relative paths, absolute paths, any depth
        # Match: "RESUME.pdf from Downloads/documents" or "file.txt in C:/Users/Ben/Desktop"
        file_from_path_pattern = r'([\w\s._-]+\.[\w]+)\s+(?:read it |)(?:from|in)\s+([\w:/\\._-]+)'
        match = re.search(file_from_path_pattern, text, re.IGNORECASE)
        if match:
            filename = match.group(1).strip()
            path_str = match.group(2).strip()
            print(f"[DEBUG] Pattern 0 matched: filename='{filename}', path='{path_str}'")
            
            # Check if it's an absolute path (starts with drive letter or /)
            if re.match(r'^[a-zA-Z]:[/\\]', path_str) or path_str.startswith('/'):
                # Absolute path - use as-is
                full_file_path = os.path.join(path_str, filename)
                return full_file_path
            
            # Relative path - parse and resolve
            path_parts = re.split(r'[/\\]+', path_str)
            
            # Resolve first part to known folder
            if path_parts:
                base_folder = self._resolve_folder(path_parts[0])
                if base_folder:
                    # Build full path with all subfolders
                    full_path = base_folder
                    for subfolder in path_parts[1:]:
                        # Try exact case first
                        test_path = os.path.join(full_path, subfolder)
                        if os.path.exists(test_path):
                            full_path = test_path
                        else:
                            # Try title case
                            test_path = os.path.join(full_path, subfolder.title())
                            if os.path.exists(test_path):
                                full_path = test_path
                            else:
                                # Try lowercase
                                test_path = os.path.join(full_path, subfolder.lower())
                                if os.path.exists(test_path):
                                    full_path = test_path
                                else:
                                    # Use original case even if doesn't exist yet
                                    full_path = os.path.join(full_path, subfolder)
                    
                    # Add filename
                    full_file_path = os.path.join(full_path, filename)
                    return full_file_path
                else:
                    # First part not a known folder, treat entire path as-is
                    full_file_path = os.path.join(path_str, filename)
                    return full_file_path
        
        # Pattern 1: Nested paths like "X in Y" or "X folder in Y" or "X which is in Y"
        # Must match folder-like words, not generic words like "files"
        nested_patterns = [
            r'(\w+)\s+folder\s+(?:which\s+is\s+)?in\s+(\w+)',  # "documents folder in downloads"
            r'(\w+)\s+(?:which\s+is\s+)?in\s+(\w+)\s+folder',  # "documents which is in downloads folder"
            r'(\w+)\s+(?:inside|within|under)\s+(\w+)',
        ]
        
        # Skip words that aren't folder names
        skip_inner = {'files', 'file', 'show', 'list', 'check', 'get', 'the', 'my', 'all'}
        
        for pattern in nested_patterns:
            match = re.search(pattern, text_lower)
            if match:
                inner = match.group(1).strip()
                outer = match.group(2).strip()
                
                # Skip if inner is not a folder-like word
                if inner in skip_inner:
                    continue
                
                # Resolve outer folder
                outer_path = self._resolve_folder(outer)
                if outer_path:
                    # Build nested path
                    inner_name = inner.title()
                    full_path = os.path.join(outer_path, inner_name)
                    # Check if it exists
                    if os.path.exists(full_path):
                        return full_path
                    # Try lowercase
                    full_path_lower = os.path.join(outer_path, inner.lower())
                    if os.path.exists(full_path_lower):
                        return full_path_lower
                    # Try original case
                    full_path_orig = os.path.join(outer_path, inner)
                    if os.path.exists(full_path_orig):
                        return full_path_orig
                    # Return title case (might be created)
                    return full_path
        
        # Pattern 1b: Handle "X in Y" without "folder" keyword but both are known folders
        simple_nested = r'(\w+)\s+(?:which\s+is\s+)?in\s+(\w+)'
        match = re.search(simple_nested, text_lower)
        if match:
            inner = match.group(1).strip()
            outer = match.group(2).strip()
            
            # Only process if both look like folder names
            if inner not in skip_inner and outer in self.known_folders:
                outer_path = self._resolve_folder(outer)
                if outer_path:
                    inner_name = inner.title()
                    full_path = os.path.join(outer_path, inner_name)
                    if os.path.exists(full_path):
                        return full_path
                    full_path_lower = os.path.join(outer_path, inner.lower())
                    if os.path.exists(full_path_lower):
                        return full_path_lower
        
        # Pattern 2: Drive references
        drive_pattern = r'\b([cdefCDEF])[\s:]*drive\b|([cdefCDEF]):[\\/]?(\S*)'
        match = re.search(drive_pattern, text)
        if match:
            if match.group(1):
                return f"{match.group(1).upper()}:/"
            elif match.group(2):
                drive = match.group(2).upper()
                subpath = match.group(3) or ""
                return f"{drive}:/{subpath}" if subpath else f"{drive}:/"
        
        # Pattern 3: Direct folder reference
        for folder_name, folder_path in self.known_folders.items():
            if folder_name in text_lower:
                return folder_path
        
        # Pattern 4: Try each word as potential folder
        words = re.findall(r'\b([a-zA-Z][\w-]*)\b', text)
        skip_words = {'the', 'my', 'in', 'show', 'list', 'check', 'get', 'find', 'folder', 
                      'files', 'directory', 'dir', 'a', 'an', 'and', 'or', 'of', 'to', 'from',
                      'is', 'are', 'which', 'that', 'this', 'what', 'where', 'please', 'can', 'you'}
        
        for word in words:
            if word.lower() not in skip_words:
                resolved = self._resolve_folder(word)
                if resolved:
                    return resolved
        
        return None
    
    def _resolve_folder(self, name: str) -> Optional[str]:
        """Resolve a folder name to path"""
        name_lower = name.lower()
        
        # Check known folders
        if name_lower in self.known_folders:
            return self.known_folders[name_lower]
        
        # Search in common locations
        search_locations = [
            Path.home(),
            Path.home() / "Downloads",
            Path.home() / "Documents", 
            Path.home() / "Desktop",
            Path("C:/"),
            Path("D:/"),
            Path("D:/Projects"),
        ]
        
        for location in search_locations:
            if location.exists():
                try:
                    for item in location.iterdir():
                        if item.is_dir() and item.name.lower() == name_lower:
                            return str(item)
                except PermissionError:
                    pass
        
        return None


class ContextManager:
    """Layer 5: Context Management"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.entities_memory: Dict[str, Any] = {}
        self.last_intent: Optional[str] = None
        self.last_path: Optional[str] = None
        self.last_files_listed: List[str] = []  # full paths or names
        self.last_commands: List[str] = []      # raw system/terminal commands
        self.session_start = datetime.now()
    
    def add_turn(self, user_input: str, nlp_result: NLPResult, response: str):
        """Add a conversation turn to history"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "normalized": nlp_result.normalized_text,
            "intent": nlp_result.intent,
            "entities": nlp_result.entities,
            "path": nlp_result.path_intent,
            "response_summary": response[:200] if response else ""
        }
        
        self.conversation_history.append(turn)
        
        # Update memory
        if nlp_result.intent:
            self.last_intent = nlp_result.intent
        if nlp_result.path_intent:
            self.last_path = nlp_result.path_intent
        
        # Merge entities
        for entity_type, values in nlp_result.entities.items():
            if values:
                if entity_type not in self.entities_memory:
                    self.entities_memory[entity_type] = []
                self.entities_memory[entity_type].extend(values)
                # Keep only recent unique values (handle dicts specially)
                try:
                    # Try set for hashable types
                    if values and not isinstance(values[0], dict):
                        self.entities_memory[entity_type] = list(set(self.entities_memory[entity_type]))[-10:]
                    else:
                        # For dicts, dedupe by converting to tuple of items
                        seen = set()
                        unique = []
                        for item in self.entities_memory[entity_type]:
                            if isinstance(item, dict):
                                key = tuple(sorted(item.items()))
                            else:
                                key = item
                            if key not in seen:
                                seen.add(key)
                                unique.append(item)
                        self.entities_memory[entity_type] = unique[-10:]
                except TypeError:
                    # Fallback: just keep last 10
                    self.entities_memory[entity_type] = self.entities_memory[entity_type][-10:]
        
        # Update specialized memories for files and commands
        # Files: track last N mentioned or listed
        files = nlp_result.entities.get("files") or []
        extracted = nlp_result.entities.get("extracted_filename") if isinstance(nlp_result.entities, dict) else None
        if extracted and isinstance(extracted, str):
            files = files + [extracted]
        if files:
            for f in files:
                if isinstance(f, str):
                    self.last_files_listed.append(f)
            # Keep only last 20
            self.last_files_listed = self.last_files_listed[-20:]

        # Commands: look for explicit command-style entities
        cmds = nlp_result.entities.get("commands") if isinstance(nlp_result.entities, dict) else None
        if cmds:
            for c in cmds:
                if isinstance(c, str):
                    self.last_commands.append(c)
            self.last_commands = self.last_commands[-20:]

        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context_summary(self) -> str:
        """Get a summary of current context"""
        summary_parts = []
        
        if self.last_intent:
            summary_parts.append(f"Last action: {self.last_intent}")
        
        if self.last_path:
            summary_parts.append(f"Last path: {self.last_path}")
        
        if self.entities_memory.get("folders"):
            recent_folders = self.entities_memory["folders"][-3:]
            folder_names = [f.get("name", f) if isinstance(f, dict) else f for f in recent_folders]
            summary_parts.append(f"Recent folders: {', '.join(folder_names)}")
        
        return " | ".join(summary_parts) if summary_parts else "No prior context"
    
    def get_relevant_context(self, current_intent: str) -> Dict[str, Any]:
        """Get context relevant to current intent"""
        context = {
            "last_path": self.last_path,
            "last_intent": self.last_intent,
            "recent_folders": [],
            "recent_files": [],
            "last_files_listed": list(self.last_files_listed[-10:]),
            "last_commands": list(self.last_commands[-10:]),
        }
        
        if self.entities_memory.get("folders"):
            context["recent_folders"] = self.entities_memory["folders"][-5:]
        
        if self.entities_memory.get("files"):
            context["recent_files"] = self.entities_memory["files"][-5:]
        
        return context


class QueryRewriter:
    """Layer 6: Query Rewriting"""
    
    def __init__(self):
        pass
    
    def rewrite(self, original: str, normalized: str, intent: str, entities: Dict, context: Dict) -> str:
        """Rewrite query into clear, explicit format"""
        
        # Start with normalized text
        rewritten = normalized
        
        # Add context if available
        context_additions = []
        
        # If referring to "it", "that", "there" - resolve from context
        pronouns = ["it", "that", "there", "this", "here"]
        for pronoun in pronouns:
            if pronoun in normalized.lower() and context.get("last_path"):
                rewritten = rewritten.replace(pronoun, f"'{context['last_path']}'")
                context_additions.append(f"Referring to: {context['last_path']}")
                break
        
        # Add intent clarification
        intent_descriptions = {
            "list_files": "User wants to list/show files in a directory",
            "read_file": "User wants to read/view file contents",
            "search_files": "User wants to search for files",
            "create_file": "User wants to create a new file/folder",
            "delete_file": "User wants to delete a file/folder",
            "system_status": "User wants system information",
            "execute_command": "User wants to run a command",
            "email_operation": "User wants to perform email operation",
            "web_search": "User wants to search the web",
            "question": "User is asking a question",
            "navigate": "User wants to navigate to a location",
        }
        
        if intent in intent_descriptions:
            context_additions.append(intent_descriptions[intent])
        
        # Build final rewritten query
        if context_additions:
            rewritten = f"{rewritten} [{'; '.join(context_additions)}]"
        
        return rewritten


class SmartNLP:
    """Main Smart NLP Pipeline"""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.context_manager = ContextManager()
        self.query_rewriter = QueryRewriter()
        self.embedding_engine = EmbeddingEngine()
        self.intent_router_llm = self._init_intent_router_llm()
        
        print("[SmartNLP] Initialized with all layers")
    
    def process(self, text: str) -> NLPResult:
        """Process text through the full NLP pipeline"""
        
        # Layer 1: Normalize
        normalized = self.normalizer.normalize(text)
        
        # Layer 3: Extract entities
        entities = self.entity_extractor.extract(text)
        
        # Extract filename separately (preserving original case/spaces)
        extracted_filename = self.entity_extractor.extract_filename(text)
        if extracted_filename:
            entities["extracted_filename"] = extracted_filename
        
        # Get path intent
        path_intent = self.entity_extractor.extract_path_intent(text)
        
        # Layer 4: Classify intent (rule-based first)
        intent, confidence = self.intent_classifier.classify(normalized)
        
        # Layer 5: Get context
        context = self.context_manager.get_relevant_context(intent)
        context_summary = self.context_manager.get_context_summary()

        # Layer 4b: Refine intent with embeddings when confidence is low
        # This makes the system more robust to unusual phrasings.
        if confidence < 0.6 and self.embedding_engine is not None:
            try:
                # Build list of (intent_key, prototype_phrase)
                candidate_pairs: List[Tuple[str, str]] = []
                for intent_key, phrases in self.intent_classifier.intent_patterns.items():
                    # Use human-readable prototypes when available
                    proto_phrases = getattr(self.intent_classifier, "intent_prototypes", {}).get(intent_key, [])
                    if not proto_phrases:
                        # Fall back to a simple derived phrase
                        proto_phrases = [f"user wants to {intent_key.replace('_', ' ')}"]
                    for p in proto_phrases:
                        candidate_pairs.append((intent_key, p))
                if candidate_pairs:
                    phrases = [p for _, p in candidate_pairs]
                    sims = self.embedding_engine.similarity_texts(normalized, phrases)
                    if sims:
                        best_phrase, best_score = sims[0]
                        # Only trust embedding refinement if similarity is reasonably high
                        if best_score >= 0.6:
                            for ik, p in candidate_pairs:
                                if p == best_phrase:
                                    intent = ik
                                    confidence = float(best_score)
                                    break
            except Exception as e:
                # Embedding refinement is best-effort; never break pipeline
                print(f"[SmartNLP WARNING] Embedding-based intent refinement failed: {e}")

        # Layer 4c: Optional LLM-based intent routing for very ambiguous messages
        # Only trigger when confidence is still low after embeddings.
        if confidence < 0.45 and self.intent_router_llm is not None:
            try:
                refined_intent, refined_conf = self._route_intent_via_llm(
                    original=text,
                    normalized=normalized,
                    intent=intent,
                    confidence=confidence,
                    entities=entities,
                    context=context,
                )
                if refined_intent:
                    intent = refined_intent
                    confidence = refined_conf
            except Exception as e:
                print(f"[SmartNLP WARNING] LLM-based intent routing failed: {e}")

        # Determine action after final intent is chosen
        action = self._determine_action(intent, entities)
        
        # SMART FILE RESOLUTION: If we have a filename and context has last_path, build full path
        if extracted_filename and context.get("last_path"):
            last_path = context["last_path"]
            if os.path.isdir(last_path):
                potential_file_path = os.path.join(last_path, extracted_filename)
                if os.path.exists(potential_file_path):
                    path_intent = potential_file_path
                else:
                    # Try case-insensitive search in the directory
                    try:
                        for f in os.listdir(last_path):
                            if f.lower() == extracted_filename.lower():
                                path_intent = os.path.join(last_path, f)
                                break
                    except PermissionError:
                        pass

        # EMBEDDING-BASED FILE RESOLUTION: handle fuzzy phrases like "that pdf" or "sorting file"
        # Only attempt when intent is clearly about reading/viewing files and we don't yet
        # have a specific filename or path.
        if (intent == "read_file" or action == "read") and not extracted_filename:
            try:
                recent_files: List[str] = []
                # Prefer specifically tracked last_files_listed, then recent_files
                if isinstance(context.get("last_files_listed"), list):
                    recent_files.extend(context["last_files_listed"])
                if isinstance(context.get("recent_files"), list):
                    # recent_files from entities_memory may already include some entries
                    for rf in context["recent_files"]:
                        if isinstance(rf, str) and rf not in recent_files:
                            recent_files.append(rf)
                # De-duplicate while preserving order
                seen_files = set()
                deduped_files: List[str] = []
                for f in recent_files:
                    if isinstance(f, str) and f not in seen_files:
                        seen_files.add(f)
                        deduped_files.append(f)
                if deduped_files:
                    sims = self.embedding_engine.similarity_texts(normalized, deduped_files)
                    if sims:
                        best_file, best_score = sims[0]
                        # Require a reasonably strong semantic match
                        if best_score >= 0.55:
                            entities["extracted_filename"] = best_file
                            extracted_filename = best_file
                            # If we have a last_path directory, build a plausible full path
                            if context.get("last_path") and os.path.isdir(context["last_path"]):
                                candidate = os.path.join(context["last_path"], best_file)
                                if os.path.exists(candidate):
                                    path_intent = candidate
            except Exception as e:
                print(f"[SmartNLP WARNING] Embedding-based file resolution failed: {e}")
        
        # Layer 6: Rewrite query
        rewritten = self.query_rewriter.rewrite(text, normalized, intent, entities, context)
        
        # Build result
        result = NLPResult(
            original_text=text,
            normalized_text=normalized,
            rewritten_query=rewritten,
            intent=intent,
            confidence=confidence,
            entities=entities,
            path_intent=path_intent,
            action=action,
            context_summary=context_summary,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "extracted_filename": extracted_filename
            }
        )
        
        return result
    
    def _determine_action(self, intent: str, entities: Dict) -> str:
        """Determine the action to take based on intent"""
        action_map = {
            "list_files": "list",
            "read_file": "read",
            "search_files": "search",
            "create_file": "create",
            "delete_file": "delete",
            "system_status": "system",
            "execute_command": "execute",
            "email_operation": "email",
            "web_search": "search_web",
            "question": "answer",
            "navigate": "navigate",
            "help": "help",
            "general_query": "ai_response",
        }
        return action_map.get(intent, "ai_response")
    
    def update_context(self, user_input: str, result: NLPResult, response: str):
        """Update context after processing"""
        self.context_manager.add_turn(user_input, result, response)

    def _init_intent_router_llm(self):
        """Initialize a lightweight LLM client for intent routing.

        This is intentionally minimal and independent from the main JARVIS LLM setup,
        so SmartNLP can still run even if no keys are configured.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                print("[SmartNLP] Intent router using Gemini")
                return ("gemini", client)
            except Exception as e:
                print(f"[SmartNLP WARNING] Failed to init Gemini for intent router: {e}")
        if groq_key:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
                print("[SmartNLP] Intent router using Groq")
                return ("groq", llm)
            except Exception as e:
                print(f"[SmartNLP WARNING] Failed to init Groq for intent router: {e}")
        return None

    def _route_intent_via_llm(
        self,
        original: str,
        normalized: str,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        """Ask a small LLM to refine the intent when things are ambiguous.

        Returns (refined_intent, refined_confidence). On any failure, returns
        (None, confidence) so the existing pipeline stays intact.
        """
        if not self.intent_router_llm:
            return None, confidence

        provider, client = self.intent_router_llm

        # Construct a compact JSON-style routing task
        router_prompt = {
            "instruction": "You are an intent router for a desktop assistant. Given the user message, normalized text, rule-based intent and entities, choose ONE best intent.",
            "allowed_intents": [
                "list_files",
                "read_file",
                "search_files",
                "create_file",
                "delete_file",
                "system_status",
                "execute_command",
                "email_operation",
                "web_search",
                "question",
                "help",
                "navigate",
                "general_query",
            ],
            "current_intent": intent,
            "current_confidence": confidence,
            "data": {
                "original": original,
                "normalized": normalized,
                "entities": entities,
                "context": context,
            },
            "output_format": {
                "intent": "one of allowed_intents",
                "confidence": "float 0.0-1.0, higher means more certain",
            },
        }

        prompt_text = (
            "You are a JSON-only intent router. Read the following JSON and return a single JSON object "
            "with keys 'intent' and 'confidence'. Do not add explanations.\n" + json.dumps(router_prompt)
        )

        try:
            if provider == "gemini":
                # google-genai simple text completion
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt_text,
                )
                text = getattr(resp, "text", None) or ""
            else:  # groq via LangChain ChatGroq
                # Use a short timeout wrapper
                def _invoke():
                    return client.invoke(prompt_text).content
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_invoke)
                    text = fut.result(timeout=15)

            text = (text or "").strip()
            # Extract JSON payload
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None, confidence
            payload = json.loads(text[start : end + 1])
            new_intent = payload.get("intent")
            new_conf = float(payload.get("confidence", confidence))

            # Validate
            allowed = set([
                "list_files",
                "read_file",
                "search_files",
                "create_file",
                "delete_file",
                "system_status",
                "execute_command",
                "email_operation",
                "web_search",
                "question",
                "help",
                "navigate",
                "general_query",
            ])
            if new_intent in allowed and 0.0 <= new_conf <= 1.0:
                return new_intent, new_conf
            return None, confidence
        except Exception as e:
            print(f"[SmartNLP WARNING] Intent router LLM call error: {e}")
            return None, confidence
    
    def get_known_folders(self) -> Dict[str, str]:
        """Get all known folders"""
        return self.entity_extractor.known_folders


# Test
if __name__ == "__main__":
    nlp = SmartNLP()
    
    test_inputs = [
        "show files in documents folder whcih is in downloads",
        "check my downloads",
        "write a ping command in terminal for google.com",
        "what's in D drive",
        "btw can u show me the pics folder",
    ]
    
    for text in test_inputs:
        print(f"\n{'='*60}")
        print(f"Input: {text}")
        result = nlp.process(text)
        print(f"Normalized: {result.normalized_text}")
        print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
        print(f"Action: {result.action}")
        print(f"Path: {result.path_intent}")
        print(f"Rewritten: {result.rewritten_query}")
