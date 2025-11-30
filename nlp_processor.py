import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class NLPProcessor:
    def __init__(self):
        self.use_spacy = False
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                print("Downloading language model for spaCy (en_core_web_sm)...")
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except Exception:
            self.use_spacy = False
        
        # Common folder name aliases (maps variations to canonical names)
        self.folder_aliases = {
            "download": "Downloads",
            "docs": "Documents",
            "document": "Documents",
            "photos": "Pictures",
            "images": "Pictures",
            "songs": "Music",
            "movies": "Videos",
        }
        
        # Build dynamic known folders from actual user directories
        self.known_folders = self._build_known_folders()
    
    def _build_known_folders(self) -> Dict[str, str]:
        """Dynamically build known folders from actual directories on the system."""
        known = {}
        
        # Add standard user folders
        user_home = Path.home()
        standard_folders = ["Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos", "AppData"]
        for folder in standard_folders:
            folder_path = user_home / folder
            if folder_path.exists():
                known[folder.lower()] = folder
        
        # Scan common locations for custom folders
        scan_locations = [
            user_home,  # User home directory
            Path("C:/"),
            Path("D:/"),
            Path("E:/"),
        ]
        
        for location in scan_locations:
            if location.exists():
                try:
                    for item in location.iterdir():
                        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('$'):
                            # Add folder name (lowercase) -> actual name mapping
                            folder_lower = item.name.lower()
                            if folder_lower not in known:
                                known[folder_lower] = item.name
                except PermissionError:
                    continue
        
        # Add aliases
        for alias, canonical in self.folder_aliases.items():
            if canonical.lower() in known:
                known[alias] = known[canonical.lower()]
        
        return known

    def extract_path_intent(self, text: str) -> Optional[str]:
        """
        Extract the intended file path from natural language.
        Handles nested paths like "documents folder in downloads" -> Downloads/Documents
        """
        text_lower = text.lower()
        
        # Pattern: "[folder1] in [folder2]" or "[folder1] folder in [folder2]"
        # e.g., "documents in downloads", "documents folder in downloads"
        nested_patterns = [
            r"(?:get|show|list|check|open|find)?\s*(?:my\s+)?(\w+)\s+(?:folder\s+)?in\s+(?:my\s+)?(\w+)",
            r"(\w+)\s+(?:folder\s+)?(?:inside|within|under)\s+(?:my\s+)?(\w+)",
        ]
        
        for pattern in nested_patterns:
            match = re.search(pattern, text_lower)
            if match:
                inner_folder = match.group(1).strip()
                outer_folder = match.group(2).strip()
                
                # Resolve outer folder to actual path
                outer_path = self._resolve_folder(outer_folder)
                if outer_path:
                    # Build nested path
                    inner_name = self.known_folders.get(inner_folder, inner_folder.title())
                    full_path = os.path.join(outer_path, inner_name)
                    return full_path
        
        # Pattern: Drive paths like "D drive", "C:", "D:/Projects" - check this BEFORE direct folders
        drive_pattern = r"\b([cdefCDEF])[\s:]*drive\b|([cdefCDEF]):[\\/]?(\S*)"
        match = re.search(drive_pattern, text)
        if match:
            if match.group(1):  # "D drive" format
                drive = match.group(1).upper()
                return f"{drive}:/"
            elif match.group(2):  # "D:/path" format
                drive = match.group(2).upper()
                subpath = match.group(3).strip() if match.group(3) else ""
                if subpath:
                    return f"{drive}:/{subpath}"
                else:
                    return f"{drive}:/"
        
        # Pattern: Direct folder reference (only match known folders)
        # e.g., "my downloads", "the documents folder", "check desktop"
        for folder_key in self.known_folders.keys():
            if folder_key in text_lower:
                resolved = self._resolve_folder(folder_key)
                if resolved:
                    return resolved
        
        # Try to extract any word that might be a folder name and resolve it
        # This handles completely custom folder names not in known_folders
        words = re.findall(r'\b([a-zA-Z][\w-]*)\b', text)
        for word in words:
            if word.lower() not in ['the', 'my', 'in', 'show', 'list', 'check', 'get', 'find', 'folder', 'files', 'directory', 'dir', 'a', 'an', 'and', 'or', 'of', 'to', 'from']:
                resolved = self._resolve_folder(word)
                if resolved:
                    return resolved
        
        return None
    
    def _resolve_folder(self, folder_name: str) -> Optional[str]:
        """Resolve a folder name to an actual path."""
        folder_lower = folder_name.lower()
        
        # First check known folders (includes dynamically discovered ones)
        if folder_lower in self.known_folders:
            canonical_name = self.known_folders[folder_lower]
            # Check if it's a user folder
            user_path = Path.home() / canonical_name
            if user_path.exists():
                return str(user_path)
            # Check common drive locations
            for drive in ["C:", "D:", "E:"]:
                drive_path = Path(f"{drive}/{canonical_name}")
                if drive_path.exists():
                    return str(drive_path)
        
        # If not in known folders, try to find it dynamically
        # This handles completely custom folder names mentioned by user
        search_locations = [
            Path.home(),
            Path.home() / "Downloads",
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path("C:/"),
            Path("D:/"),
            Path("E:/"),
            Path("D:/Projects"),  # Common project locations
            Path("C:/Projects"),
        ]
        
        for location in search_locations:
            if location.exists():
                try:
                    # Try exact match (case-insensitive)
                    for item in location.iterdir():
                        if item.is_dir() and item.name.lower() == folder_lower:
                            return str(item)
                except PermissionError:
                    continue
        
        # Try title case as last resort
        for location in search_locations:
            if location.exists():
                potential_path = location / folder_name.title()
                if potential_path.exists() and potential_path.is_dir():
                    return str(potential_path)
        
        return None
    
    def extract_action_intent(self, text: str) -> Dict[str, Any]:
        """Extract the action intent from the text."""
        text_lower = text.lower()
        
        intent = {
            "action": None,
            "target_path": None,
            "search_term": None,
            "file_type": None,
        }
        
        # Determine action type
        if any(word in text_lower for word in ["list", "show", "get", "check", "what", "display"]):
            intent["action"] = "list"
        elif any(word in text_lower for word in ["read", "open", "view", "content"]):
            intent["action"] = "read"
        elif any(word in text_lower for word in ["search", "find", "look for", "locate"]):
            intent["action"] = "search"
        elif any(word in text_lower for word in ["create", "make", "new"]):
            intent["action"] = "create"
        elif any(word in text_lower for word in ["delete", "remove"]):
            intent["action"] = "delete"
        elif any(word in text_lower for word in ["copy", "move"]):
            intent["action"] = "copy"
        
        # Extract target path
        intent["target_path"] = self.extract_path_intent(text)
        
        # Extract file type if searching
        file_type_match = re.search(r"\.(\w+)\s+files?|(\w+)\s+files?", text_lower)
        if file_type_match:
            intent["file_type"] = file_type_match.group(1) or file_type_match.group(2)
        
        return intent

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and extract entities and intents."""
        result = {
            "entities": [],
            "tokens": [],
            "path_intent": None,
            "action_intent": None,
        }
        
        # Extract path and action intents
        result["path_intent"] = self.extract_path_intent(text)
        result["action_intent"] = self.extract_action_intent(text)
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            result["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
            result["tokens"] = [token.text for token in doc]
        else:
            # Regex-based entity extraction
            emails = re.findall(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", text)
            urls = re.findall(r"https?://\S+|www\.\S+", text)
            money = re.findall(r"[$€£¥]\s?\d+(?:[\.,]\d+)*", text)
            dates = re.findall(r"\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b", text, flags=re.IGNORECASE)
            times = re.findall(r"\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b", text, flags=re.IGNORECASE)
            
            for e in emails:
                result["entities"].append((e, "EMAIL"))
            for u in urls:
                result["entities"].append((u, "URL"))
            for d in dates:
                result["entities"].append((d, "DATE"))
            for t in times:
                result["entities"].append((t, "TIME"))
            for m in money:
                result["entities"].append((m, "MONEY"))
            
            result["tokens"] = re.findall(r"\w+|[^\w\s]", text)
        
        return result
