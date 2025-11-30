#!/usr/bin/env python3
"""
System Monitor and File Manager for JARVIS
==========================================

Handles system monitoring, file operations, and system control
"""

import os
import psutil
import shutil
import subprocess
import platform
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor system resources and status"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "current_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percentage": memory.percent
        }
    
    @staticmethod
    def get_running_processes(limit: int = 10) -> List[Dict[str, Any]]:
        """Get top running processes by CPU usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        return processes[:limit]
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """Get network interface information"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}

class FileManager:
    """Advanced file system operations"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.home()
    
    def list_files(self, directory: str = None, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List all files in a directory"""
        search_dir = Path(directory) if directory else self.base_path
        results = []
        
        try:
            if not search_dir.exists():
                return [{"error": f"Directory not found: {search_dir}"}]
            
            for item in search_dir.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                    
                try:
                    stat = item.stat()
                    results.append({
                        "name": item.name,
                        "path": str(item),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_file": item.is_file(),
                        "is_directory": item.is_dir(),
                        "extension": item.suffix if item.is_file() else None
                    })
                except (OSError, PermissionError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return [{"error": str(e)}]
        
        return sorted(results, key=lambda x: (x["is_directory"], x["name"].lower()))
    
    def read_entire_folder(self, directory: str, max_depth: int = 3, include_content: bool = False) -> Dict[str, Any]:
        """Read entire folder structure with optional content"""
        try:
            folder_path = Path(directory)
            if not folder_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            if not folder_path.is_dir():
                return {"error": f"Path is not a directory: {directory}"}
            
            structure = {
                "path": str(folder_path),
                "name": folder_path.name,
                "type": "directory",
                "contents": [],
                "total_files": 0,
                "total_directories": 0,
                "total_size": 0
            }
            
            def scan_directory(path: Path, current_depth: int = 0):
                if current_depth >= max_depth:
                    return
                
                try:
                    for item in path.iterdir():
                        try:
                            stat = item.stat()
                            item_info = {
                                "name": item.name,
                                "path": str(item),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "is_file": item.is_file(),
                                "is_directory": item.is_dir(),
                                "extension": item.suffix if item.is_file() else None,
                                "depth": current_depth
                            }
                            
                            # Add content for small text files if requested
                            if include_content and item.is_file() and item.suffix.lower() in ['.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.md']:
                                try:
                                    if stat.st_size < 1024 * 1024:  # Less than 1MB
                                        with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                            item_info["content"] = f.read()[:5000]  # First 5000 chars
                                except:
                                    pass
                            
                            structure["contents"].append(item_info)
                            
                            if item.is_file():
                                structure["total_files"] += 1
                                structure["total_size"] += stat.st_size
                            elif item.is_dir():
                                structure["total_directories"] += 1
                                # Recursively scan subdirectories
                                scan_directory(item, current_depth + 1)
                                
                        except (OSError, PermissionError):
                            continue
                            
                except (OSError, PermissionError):
                    pass
            
            scan_directory(folder_path)
            return structure
            
        except Exception as e:
            logger.error(f"Error reading folder {directory}: {e}")
            return {"error": str(e)}
    
    def get_user_profiles(self) -> Dict[str, Any]:
        """Get user profile information and accessible directories"""
        try:
            user_home = Path.home()
            profiles = {
                "user_home": str(user_home),
                "username": user_home.name,
                "accessible_directories": {
                    "Downloads": str(user_home / "Downloads"),
                    "Documents": str(user_home / "Documents"),
                    "Desktop": str(user_home / "Desktop"),
                    "Pictures": str(user_home / "Pictures"),
                    "Music": str(user_home / "Music"),
                    "Videos": str(user_home / "Videos"),
                    "AppData": str(user_home / "AppData"),
                    "AppData_Roaming": str(user_home / "AppData" / "Roaming"),
                    "AppData_Local": str(user_home / "AppData" / "Local")
                }
            }
            
            # Check which directories actually exist
            for name, path in profiles["accessible_directories"].items():
                if not Path(path).exists():
                    profiles["accessible_directories"][name] = None
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error getting user profiles: {e}")
            return {"error": str(e)}
    
    def search_files(self, pattern: str, directory: str = None, file_type: str = None) -> List[Dict[str, Any]]:
        """Search for files matching pattern with fuzzy matching"""
        search_dir = Path(directory) if directory else self.base_path
        results = []
        
        try:
            # First try exact matches
            exact_matches = self._search_exact_matches(pattern, search_dir, file_type)
            if exact_matches:
                return exact_matches
            
            # If no exact matches, try fuzzy matching
            fuzzy_matches = self._search_fuzzy_matches(pattern, search_dir, file_type)
            if fuzzy_matches:
                return fuzzy_matches
            
            # If still no matches, try directory search
            dir_matches = self._search_directories(pattern, search_dir)
            if dir_matches:
                return dir_matches
                
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return [{"error": str(e)}]
        
        return [{"error": f"No files or directories found matching '{pattern}'"}]
    
    def _search_exact_matches(self, pattern: str, search_dir: Path, file_type: str = None) -> List[Dict[str, Any]]:
        """Search for exact matches"""
        results = []
        
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if pattern.lower() in file.lower():
                        file_path = Path(root) / file
                        if file_type and not file.endswith(file_type):
                            continue
                        
                        try:
                            stat = file_path.stat()
                            results.append({
                                "name": file,
                                "path": str(file_path),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "is_file": True,
                                "match_type": "exact"
                            })
                        except (OSError, PermissionError):
                            continue
                
                # Limit results to prevent overwhelming output
                if len(results) >= 50:
                    break
                    
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
        
        return results
    
    def _search_fuzzy_matches(self, pattern: str, search_dir: Path, file_type: str = None) -> List[Dict[str, Any]]:
        """Search for fuzzy matches using similarity scoring"""
        results = []
        pattern_lower = pattern.lower()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    file_lower = file.lower()
                    
                    # Calculate similarity score
                    similarity = self._calculate_similarity(pattern_lower, file_lower)
                    
                    # Only include files with reasonable similarity
                    if similarity > 0.3:  # 30% similarity threshold
                        file_path = Path(root) / file
                        if file_type and not file.endswith(file_type):
                            continue
                        
                        try:
                            stat = file_path.stat()
                            results.append({
                                "name": file,
                                "path": str(file_path),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "is_file": True,
                                "match_type": "fuzzy",
                                "similarity": round(similarity, 2)
                            })
                        except (OSError, PermissionError):
                            continue
                
                # Limit results to prevent overwhelming output
                if len(results) >= 30:
                    break
                    
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results[:20]  # Return top 20 matches
    
    def _search_directories(self, pattern: str, search_dir: Path) -> List[Dict[str, Any]]:
        """Search for directories matching pattern"""
        results = []
        pattern_lower = pattern.lower()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                for dir_name in dirs:
                    dir_lower = dir_name.lower()
                    
                    # Check if directory name contains pattern or is similar
                    if pattern_lower in dir_lower or self._calculate_similarity(pattern_lower, dir_lower) > 0.4:
                        dir_path = Path(root) / dir_name
                        try:
                            # Count files in directory
                            file_count = sum(1 for _ in dir_path.iterdir() if _.is_file())
                            dir_count = sum(1 for _ in dir_path.iterdir() if _.is_dir())
                            
                            results.append({
                                "name": dir_name,
                                "path": str(dir_path),
                                "is_file": False,
                                "is_directory": True,
                                "file_count": file_count,
                                "dir_count": dir_count,
                                "match_type": "directory"
                            })
                        except (OSError, PermissionError):
                            continue
                
                # Limit results
                if len(results) >= 20:
                    break
                    
        except Exception as e:
            logger.error(f"Error in directory search: {e}")
        
        return results
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple similarity calculation
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Calculate character overlap
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def read_file(self, file_path: str, max_size: int = 1024*1024, use_llm_summary: bool = False) -> Dict[str, Any]:
        """Read file content with intelligent extraction for PDFs and images
        
        Args:
            file_path: Path to the file
            max_size: Maximum file size in bytes
            use_llm_summary: If True, use LLM to summarize/understand PDF and image content
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if path.stat().st_size > max_size:
                return {"error": f"File too large (>{max_size} bytes). Use a smaller file or increase max_size."}
            
            suffix = path.suffix.lower()
            
            # Handle PDFs with text extraction
            if suffix == '.pdf':
                return self._read_pdf(path, use_llm_summary)
            
            # Handle images with optional OCR
            elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                return self._read_image(path, use_llm_summary)
            
            # Default: read as text file
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                return {
                    "content": content,
                    "size": len(content),
                    "path": str(path),
                    "lines": len(content.splitlines())
                }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}
    
    def write_file(self, file_path: str, content: str, backup: bool = True) -> Dict[str, Any]:
        """Write content to file with optional backup"""
        try:
            path = Path(file_path)
            
            # Create backup if file exists
            if backup and path.exists():
                backup_path = path.with_suffix(f"{path.suffix}.backup.{int(datetime.now().timestamp())}")
                shutil.copy2(path, backup_path)
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path),
                "size": len(content),
                "backup_created": backup and path.exists()
            }
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return {"error": str(e)}
    
    def organize_files(self, directory: str, by_type: bool = True) -> Dict[str, Any]:
        """Organize files in directory by type or date"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            organized = {}
            moved_count = 0
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    if by_type:
                        # Organize by file extension
                        ext = file_path.suffix.lower() or 'no_extension'
                        target_dir = dir_path / ext[1:]  # Remove the dot
                    else:
                        # Organize by modification date
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        target_dir = dir_path / mod_time.strftime('%Y-%m-%d')
                    
                    target_dir.mkdir(exist_ok=True)
                    target_file = target_dir / file_path.name
                    
                    if not target_file.exists():
                        shutil.move(str(file_path), str(target_file))
                        moved_count += 1
                    
                    folder_name = str(target_dir.relative_to(dir_path))
                    if folder_name not in organized:
                        organized[folder_name] = []
                    organized[folder_name].append(file_path.name)
            
            return {
                "success": True,
                "moved_files": moved_count,
                "organized_folders": organized
            }
            
        except Exception as e:
            logger.error(f"Error organizing files in {directory}: {e}")
            return {"error": str(e)}
    
    def _read_pdf(self, path: Path, use_llm_summary: bool = True) -> Dict[str, Any]:
        """Extract text from PDF and optionally summarize with LLM"""
        try:
            logger.info(f"Reading PDF: {path.name}")
            # Try pypdf first
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text_parts = []
                logger.info(f"Extracting text from {len(reader.pages)} pages...")
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                extracted_text = "\n".join(text_parts)
                logger.info(f"Extracted {len(extracted_text)} characters from PDF")
            except ImportError:
                # Fallback: try PyPDF2
                try:
                    import PyPDF2
                    with open(path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text_parts = []
                        for page in reader.pages:
                            text_parts.append(page.extract_text())
                    extracted_text = "\n".join(text_parts)
                except ImportError:
                    return {
                        "error": "PDF reading requires 'pypdf' or 'PyPDF2'. Install with: pip install pypdf",
                        "path": str(path)
                    }
            
            if not extracted_text.strip():
                return {
                    "content": "[PDF appears to be image-based or empty. No text could be extracted.]",
                    "path": str(path),
                    "size": 0,
                    "lines": 0
                }
            
            # If LLM summary requested, use it to understand the content
            if use_llm_summary and len(extracted_text) > 500:
                logger.info(f"Requesting LLM summary for {path.name}...")
                summary = self._summarize_with_llm(extracted_text, str(path))
                if summary:
                    logger.info(f"LLM summary completed for {path.name}")
                    return {
                        "content": summary,
                        "raw_text": extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text,
                        "path": str(path),
                        "size": len(extracted_text),
                        "lines": len(extracted_text.splitlines()),
                        "summarized": True
                    }
                else:
                    logger.warning(f"LLM summary failed for {path.name}, returning raw text")
            
            # Return extracted text (truncated if too long)
            if len(extracted_text) > 3000:
                content = extracted_text[:3000] + f"\n\n[... truncated, total {len(extracted_text)} characters]"
            else:
                content = extracted_text
            
            return {
                "content": content,
                "path": str(path),
                "size": len(extracted_text),
                "lines": len(extracted_text.splitlines())
            }
            
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            return {"error": f"Failed to read PDF: {str(e)}"}
    
    def _read_image(self, path: Path, use_llm_summary: bool = True) -> Dict[str, Any]:
        """Read image with optional OCR or vision LLM"""
        try:
            # Option 1: Use Gemini Vision if available (preferred for understanding)
            if use_llm_summary:
                vision_result = self._analyze_image_with_vision_llm(path)
                if vision_result:
                    return vision_result
            
            # Option 2: Fallback to OCR if pytesseract available
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(path)
                text = pytesseract.image_to_string(img)
                
                if text.strip():
                    return {
                        "content": f"[OCR extracted text from image]\n\n{text}",
                        "path": str(path),
                        "size": len(text),
                        "lines": len(text.splitlines()),
                        "method": "ocr"
                    }
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"OCR failed for {path}: {e}")
            
            # Option 3: Just return image metadata
            return {
                "content": f"[Image file: {path.name}]\nTo read text from images, install pytesseract or configure Gemini Vision.",
                "path": str(path),
                "size": path.stat().st_size,
                "lines": 0,
                "method": "metadata_only"
            }
            
        except Exception as e:
            logger.error(f"Error reading image {path}: {e}")
            return {"error": f"Failed to read image: {str(e)}"}
    
    def _summarize_with_llm(self, text: str, file_path: str, timeout: int = 15) -> Optional[str]:
        """Use LLM to summarize or understand document content
        
        Args:
            text: Document text to summarize
            file_path: Path to the document
            timeout: Maximum seconds to wait for LLM response (default 15)
        """
        import concurrent.futures
        
        def _call_gemini():
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                return None
            
            from google import genai
            client = genai.Client(api_key=gemini_key)
            
            prompt = f"""You are analyzing a document. Provide a clear, structured summary.

Document: {Path(file_path).name}

Content:
{text[:4000]}

Provide:
1. Brief overview (2-3 sentences)
2. Key points or sections
3. Main topics covered

Be concise and informative."""
            
            logger.info("Calling Gemini API for summarization...")
            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
            )
            summary = getattr(resp, "text", None)
            if summary:
                return f"[AI Summary of {Path(file_path).name}]\n\n{summary}"
            return None
        
        try:
            # Try Gemini with timeout
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(_call_gemini)
                        try:
                            result = future.result(timeout=timeout)
                            if result:
                                return result
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Gemini summarization timed out after {timeout}s")
                            return None
                except Exception as e:
                    logger.warning(f"Gemini summarization failed: {e}")
            
            # Fallback to Groq if available
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    from langchain_groq import ChatGroq
                    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
                    
                    prompt = f"Summarize this document in 3-5 key points:\n\n{text[:3000]}"
                    summary = llm.invoke(prompt).content
                    if summary:
                        return f"[AI Summary of {Path(file_path).name}]\n\n{summary}"
                except Exception as e:
                    logger.warning(f"Groq summarization failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"LLM summarization error: {e}")
            return None
    
    def _analyze_image_with_vision_llm(self, path: Path) -> Optional[Dict[str, Any]]:
        """Use Gemini Vision to understand image content"""
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                return None
            
            from google import genai
            from PIL import Image
            import base64
            import io
            
            client = genai.Client(api_key=gemini_key)
            
            # Load and encode image
            img = Image.open(path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()
            
            # Call Gemini Vision
            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    "Describe this image in detail. If there is text, extract it. Explain what you see.",
                    {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}
                ],
            )
            
            description = getattr(resp, "text", None)
            if description:
                return {
                    "content": f"[AI Vision Analysis of {path.name}]\n\n{description}",
                    "path": str(path),
                    "size": len(description),
                    "lines": len(description.splitlines()),
                    "method": "gemini_vision"
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Gemini Vision analysis failed for {path}: {e}")
            return None
    
    def get_directory_info(self, directory: str) -> Dict[str, Any]:
        """Get comprehensive directory information"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            total_size = 0
            file_count = 0
            dir_count = 0
            file_types = {}
            
            for item in dir_path.rglob('*'):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
                    ext = item.suffix.lower() or 'no_extension'
                    file_types[ext] = file_types.get(ext, 0) + 1
                elif item.is_dir():
                    dir_count += 1
            
            return {
                "path": str(dir_path),
                "total_size": total_size,
                "file_count": file_count,
                "directory_count": dir_count,
                "file_types": file_types,
                "size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting directory info for {directory}: {e}")
            return {"error": str(e)}

class SystemController:
    """Control system operations"""
    
    @staticmethod
    def execute_command(command: str, shell: bool = True) -> Dict[str, Any]:
        """Execute system command safely"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {"error": str(e)}
    
    @staticmethod
    def execute_powershell_command(command: str) -> Dict[str, Any]:
        """Execute PowerShell command safely"""
        try:
            # Use PowerShell with proper execution policy
            ps_command = f"powershell.exe -ExecutionPolicy Bypass -Command \"{command}\""
            result = subprocess.run(
                ps_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": command,
                "powershell_command": ps_command
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "PowerShell command timed out after 60 seconds"}
        except Exception as e:
            logger.error(f"Error executing PowerShell command '{command}': {e}")
            return {"error": str(e)}
    
    @staticmethod
    def shutdown_system(delay: int = 0) -> Dict[str, Any]:
        """Shutdown system with optional delay"""
        try:
            if platform.system() == "Windows":
                command = f"shutdown /s /t {delay}"
            else:
                command = f"shutdown -h +{delay//60}" if delay > 0 else "shutdown -h now"
            
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return {
                "success": True,
                "message": f"System will shutdown in {delay} seconds",
                "command": command
            }
            
        except Exception as e:
            logger.error(f"Error shutting down system: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def restart_system(delay: int = 0) -> Dict[str, Any]:
        """Restart system with optional delay"""
        try:
            if platform.system() == "Windows":
                command = f"shutdown /r /t {delay}"
            else:
                command = f"shutdown -r +{delay//60}" if delay > 0 else "shutdown -r now"
            
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return {
                "success": True,
                "message": f"System will restart in {delay} seconds",
                "command": command
            }
            
        except Exception as e:
            logger.error(f"Error restarting system: {e}")
            return {"error": str(e)}

# Integration with JARVIS
class JARVISSystemIntegration:
    """System integration for JARVIS"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.file_manager = FileManager()
        self.controller = SystemController()
    
    def handle_system_command(self, command: str) -> str:
        """Handle system-related commands"""
        command_lower = command.lower()
        
        # Check for NLP-extracted path in command
        nlp_path = None
        if "[NLP_PATH:" in command:
            import re
            match = re.search(r'\[NLP_PATH:([^\]]+)\]', command)
            if match:
                nlp_path = match.group(1)
                # Remove the NLP tag from command for further processing
                command = re.sub(r'\s*\[NLP_PATH:[^\]]+\]', '', command)
                command_lower = command.lower()
        
        # If NLP extracted a path, use it directly for file operations
        if nlp_path:
            # Check if it's a file FIRST (priority for read operations)
            if os.path.isfile(nlp_path):
                logger.info(f"Reading file from NLP path: {nlp_path}")
                print(f"[DEBUG] About to read file: {nlp_path}")
                result = self.file_manager.read_file(nlp_path)
                print(f"[DEBUG] File read completed, got result with keys: {result.keys()}")
                if "error" in result:
                    return f"Error reading file: {result['error']}"
                # Show summary info with content
                summary_note = " [AI Summarized]" if result.get('summarized') else ""
                method_note = f" [{result['method']}]" if 'method' in result else ""
                print(f"[DEBUG] Returning result to user")
                return f"File: {os.path.basename(nlp_path)} ({result.get('lines', 0)} lines){summary_note}{method_note}\n\n{result['content']}"
            
            # If not a file, check if it's a directory
            elif os.path.isdir(nlp_path):
                logger.info(f"Listing directory from NLP path: {nlp_path}")
                results = self.file_manager.list_files(nlp_path)
                if results and "error" not in results[0]:
                    result = f"Files in {nlp_path}:\n"
                    for file_info in results:
                        file_type = "[DIR]" if file_info['is_directory'] else "[FILE]"
                        size_str = f"({file_info['size']} bytes)" if file_info['is_file'] else ""
                        result += f"{file_type} {file_info['name']} {size_str}\n"
                    return result
                else:
                    return f"Error listing files in {nlp_path}: {results[0].get('error', 'Unknown error') if results else 'No results'}"
            
            # Path doesn't exist - try case-insensitive search
            else:
                logger.warning(f"Path not found: {nlp_path}, attempting case-insensitive search...")
                # Try to find the file with case-insensitive matching
                parent_dir = os.path.dirname(nlp_path)
                filename = os.path.basename(nlp_path)
                
                if os.path.isdir(parent_dir):
                    try:
                        for item in os.listdir(parent_dir):
                            if item.lower() == filename.lower():
                                found_path = os.path.join(parent_dir, item)
                                logger.info(f"Found file with different case: {found_path}")
                                result = self.file_manager.read_file(found_path)
                                if "error" not in result:
                                    summary_note = " [AI Summarized]" if result.get('summarized') else ""
                                    return f"File: {item} ({result.get('lines', 0)} lines){summary_note}\n\n{result['content']}"
                    except PermissionError:
                        pass
                
                return f"Path not found: {nlp_path}. Please check if the file/folder exists."
        
        if "system status" in command_lower or "system info" in command_lower:
            info = self.monitor.get_system_info()
            return f"System Status:\n{json.dumps(info, indent=2)}"
        
        elif "cpu usage" in command_lower:
            cpu = self.monitor.get_cpu_usage()
            return f"Current CPU Usage: {cpu}%"
        
        elif "memory usage" in command_lower:
            memory = self.monitor.get_memory_usage()
            return f"Memory Usage: {memory['percentage']:.1f}% ({memory['used']//1024//1024}MB used of {memory['total']//1024//1024}MB total)"
        
        elif "running processes" in command_lower:
            processes = self.monitor.get_running_processes()
            result = "Top Running Processes:\n"
            for proc in processes:
                result += f"- {proc['name']} (PID: {proc['pid']}) - CPU: {proc['cpu_percent']:.1f}%\n"
            return result
        
        elif command_lower.startswith("search files "):
            pattern = command[13:].strip()
            results = self.file_manager.search_files(pattern)
            if results and "error" not in results[0]:
                result = f"Found {len(results)} files matching '{pattern}':\n\n"
                
                # Group results by match type
                exact_matches = [r for r in results if r.get('match_type') == 'exact']
                fuzzy_matches = [r for r in results if r.get('match_type') == 'fuzzy']
                dir_matches = [r for r in results if r.get('match_type') == 'directory']
                
                if exact_matches:
                    result += "[EXACT MATCHES]:\n"
                    for file_info in exact_matches[:5]:
                        file_type = "[DIR]" if file_info.get('is_directory') else "[FILE]"
                        size_str = f"({file_info['size']} bytes)" if file_info.get('is_file') else ""
                        result += f"  {file_type} {file_info['name']} {size_str}\n"
                        result += f"        Path: {file_info['path']}\n"
                    result += "\n"
                
                if fuzzy_matches:
                    result += "[SIMILAR MATCHES]:\n"
                    for file_info in fuzzy_matches[:5]:
                        similarity = file_info.get('similarity', 0)
                        file_type = "[DIR]" if file_info.get('is_directory') else "[FILE]"
                        size_str = f"({file_info['size']} bytes)" if file_info.get('is_file') else ""
                        result += f"  {file_type} {file_info['name']} {size_str} (similarity: {similarity})\n"
                        result += f"        Path: {file_info['path']}\n"
                    result += "\n"
                
                if dir_matches:
                    result += "[DIRECTORY MATCHES]:\n"
                    for dir_info in dir_matches[:3]:
                        file_count = dir_info.get('file_count', 0)
                        dir_count = dir_info.get('dir_count', 0)
                        result += f"  [DIR] {dir_info['name']} ({file_count} files, {dir_count} dirs)\n"
                        result += f"        Path: {dir_info['path']}\n"
                    result += "\n"
                
                if len(results) > 15:
                    result += f"... and {len(results) - 15} more matches\n"
                
                return result
            else:
                return f"No files found matching '{pattern}'"
        
        elif command_lower.startswith("read file "):
            file_path = command[9:].strip()
            
            # Check if it's a PDF or image to inform user about processing
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.pdf':
                logger.info(f"Processing PDF file: {file_path}")
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                logger.info(f"Processing image file: {file_path}")
            
            result = self.file_manager.read_file(file_path)
            if "error" in result:
                return f"Error: {result['error']}"
            else:
                # Show if it was summarized
                summary_note = " [AI Summarized]" if result.get('summarized') else ""
                method_note = f" [{result['method']}]" if 'method' in result else ""
                return f"File content ({result['lines']} lines){summary_note}{method_note}:\n{result['content'][:500]}{'...' if len(result['content']) > 500 else ''}"
        
        elif command_lower.startswith("write file "):
            parts = command[10:].strip().split(" ", 1)
            if len(parts) == 2:
                file_path, content = parts
                result = self.file_manager.write_file(file_path, content)
                if "error" in result:
                    return f"Error: {result['error']}"
                else:
                    return f"Successfully wrote {result['size']} characters to {result['path']}"
            else:
                return "Invalid write command format. Use: write file [path] [content]"
        
        elif command_lower.startswith("organize files "):
            directory = command[15:].strip()
            result = self.file_manager.organize_files(directory)
            if "error" in result:
                return f"Error: {result['error']}"
            else:
                return f"Organized {result['moved_files']} files into folders"
        
        elif (command_lower.startswith("list files") or command_lower.startswith("ls ") or 
              "files in" in command_lower or "list files in" in command_lower or 
              "show files in" in command_lower or "what files in" in command_lower):
            # Handle various file listing commands
            directory = None
            
            # Extract directory from different patterns
            if command_lower.startswith("list files in "):
                directory = command[14:].strip()
            elif command_lower.startswith("list files "):
                directory = command[11:].strip()
            elif command_lower.startswith("show files in "):
                directory = command[14:].strip()
            elif command_lower.startswith("ls "):
                directory = command[3:].strip()
            elif "files in " in command_lower:
                directory = command[command_lower.find("files in ") + 9:].strip()
            elif "what files in" in command_lower:
                directory = command[command_lower.find("what files in") + 13:].strip()
            
            # Handle context-aware directory resolution
            if directory and " in " in directory.lower():
                # Handle "projects in d drive" format
                parts = directory.lower().split(" in ")
                if len(parts) == 2:
                    dir_name = parts[0].strip()
                    drive_name = parts[1].strip()
                    
                    # Map drive names to actual paths
                    if "d drive" in drive_name or "d:" in drive_name:
                        if dir_name == "projects":
                            directory = "D:/Projects"
                        else:
                            directory = f"D:/{dir_name.title()}"
                    elif "c drive" in drive_name or "c:" in drive_name:
                        if dir_name == "projects":
                            directory = "C:/Projects"
                        else:
                            directory = f"C:/{dir_name.title()}"
                    elif "e drive" in drive_name or "e:" in drive_name:
                        if dir_name == "projects":
                            directory = "E:/Projects"
                        else:
                            directory = f"E:/{dir_name.title()}"
            
            # Handle special directories
            elif directory and directory.lower() in ["downloads", "download", "in downloads", "downloads folder"]:
                downloads_path = Path.home() / "Downloads"
                directory = str(downloads_path)
            elif directory and directory.lower() in ["documents", "docs", "in documents", "documents folder"]:
                docs_path = Path.home() / "Documents"
                directory = str(docs_path)
            elif directory and directory.lower() in ["desktop", "in desktop", "desktop folder"]:
                desktop_path = Path.home() / "Desktop"
                directory = str(desktop_path)
            elif directory and directory.lower() in ["pictures", "images", "photos"]:
                pictures_path = Path.home() / "Pictures"
                directory = str(pictures_path)
            elif directory and directory.lower() in ["music", "songs"]:
                music_path = Path.home() / "Music"
                directory = str(music_path)
            elif directory and directory.lower() in ["videos", "movies"]:
                videos_path = Path.home() / "Videos"
                directory = str(videos_path)
            
            # Try to list files in the specified directory
            results = self.file_manager.list_files(directory)
            if results and "error" not in results[0]:
                result = f"Files in {directory or 'current directory'}:\n"
                for file_info in results:
                    file_type = "[DIR]" if file_info['is_directory'] else "[FILE]"
                    size_str = f"({file_info['size']} bytes)" if file_info['is_file'] else ""
                    result += f"{file_type} {file_info['name']} {size_str}\n"
                return result
            else:
                # If directory not found, try intelligent search
                search_results = self._intelligent_directory_search(directory)
                if search_results:
                    return search_results
                else:
                    return f"Error listing files: {results[0].get('error', 'Unknown error')}"
        
        elif command_lower.startswith("read and make a list of all files from "):
            # Handle "read and make a list of all files from [directory]" commands
            directory_part = command[39:].strip()
            
            # Handle special directories
            if directory_part.lower() in ["downloads", "download"]:
                directory = str(Path.home() / "Downloads")
            elif directory_part.lower() in ["documents", "docs"]:
                directory = str(Path.home() / "Documents")
            elif directory_part.lower() in ["desktop"]:
                directory = str(Path.home() / "Desktop")
            else:
                directory = directory_part
            
            results = self.file_manager.list_files(directory)
            if results and "error" not in results[0]:
                result = f"Complete file list for {directory}:\n\n"
                files = [f for f in results if f['is_file']]
                directories = [f for f in results if f['is_directory']]
                
                if directories:
                    result += "[DIRECTORIES]:\n"
                    for dir_info in directories:
                        result += f"  [DIR] {dir_info['name']}\n"
                    result += "\n"
                
                if files:
                    result += f"[FILES] ({len(files)} total):\n"
                    for file_info in files:
                        size_mb = file_info['size'] / (1024 * 1024)
                        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size']} bytes"
                        result += f"  [FILE] {file_info['name']} ({size_str})\n"
                
                return result
            else:
                return f"Error accessing directory: {results[0].get('error', 'Unknown error')}"
        
        elif command_lower.startswith("read entire folder") or command_lower.startswith("read folder"):
            # Handle "read entire folder [directory]" or "read folder [directory]" commands
            if command_lower.startswith("read entire folder "):
                directory_part = command[19:].strip()
            else:
                directory_part = command[12:].strip()
            
            # Handle special directories
            if directory_part.lower() in ["downloads", "download"]:
                directory = str(Path.home() / "Downloads")
            elif directory_part.lower() in ["documents", "docs"]:
                directory = str(Path.home() / "Documents")
            elif directory_part.lower() in ["desktop"]:
                directory = str(Path.home() / "Desktop")
            elif directory_part.lower() in ["user profile", "user profiles", "profile"]:
                return self.handle_user_profile_command()
            else:
                directory = directory_part
            
            # Read entire folder structure
            folder_data = self.file_manager.read_entire_folder(directory, max_depth=2, include_content=False)
            if "error" in folder_data:
                return f"Error reading folder: {folder_data['error']}"
            
            result = f"Complete folder structure for {directory}:\n\n"
            result += f"Summary: {folder_data['total_files']} files, {folder_data['total_directories']} directories\n"
            result += f"Total size: {folder_data['total_size'] / (1024 * 1024):.2f} MB\n\n"
            
            # Organize by depth
            by_depth = {}
            for item in folder_data['contents']:
                depth = item.get('depth', 0)
                if depth not in by_depth:
                    by_depth[depth] = []
                by_depth[depth].append(item)
            
            # Display structure
            for depth in sorted(by_depth.keys()):
                indent = "  " * depth
                for item in by_depth[depth]:
                    if item['is_directory']:
                        result += f"{indent}[DIR] {item['name']}/\n"
                    else:
                        size_mb = item['size'] / (1024 * 1024)
                        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{item['size']} bytes"
                        result += f"{indent}[FILE] {item['name']} ({size_str})\n"
            
            return result
        
        elif command_lower.startswith("user profile") or command_lower.startswith("user profiles"):
            return self.handle_user_profile_command()
        
        elif command_lower.startswith("powershell ") or command_lower.startswith("ps "):
            # Handle PowerShell commands
            if command_lower.startswith("powershell "):
                ps_command = command[11:].strip()
            else:
                ps_command = command[3:].strip()
            
            result = self.controller.execute_powershell_command(ps_command)
            if result.get("success"):
                return f"[SUCCESS] PowerShell Command Executed:\n{result['stdout']}"
            else:
                return f"[ERROR] PowerShell Command Failed:\n{result.get('error', 'Unknown error')}\n{result.get('stderr', '')}"
        
        elif command_lower.startswith("cmd ") or command_lower.startswith("command "):
            # Handle CMD commands
            if command_lower.startswith("cmd "):
                cmd_command = command[4:].strip()
            else:
                cmd_command = command[8:].strip()
            
            result = self.controller.execute_command(cmd_command)
            if result.get("success"):
                return f"[SUCCESS] Command Executed:\n{result['stdout']}"
            else:
                return f"[ERROR] Command Failed:\n{result.get('error', 'Unknown error')}\n{result.get('stderr', '')}"
        
        else:
            # Return None so the command gets passed to AI for understanding
            return None
    
    def _intelligent_directory_search(self, directory_name: str) -> str:
        """Intelligently search for directories when exact path not found"""
        try:
            # Search in common locations
            search_locations = [
                Path.home(),  # User home directory
                Path("C:/"),  # C drive
                Path("D:/"),  # D drive
                Path("E:/"),  # E drive
            ]
            
            found_directories = []
            
            for search_root in search_locations:
                if not search_root.exists():
                    continue
                    
                # Search for directories matching the pattern
                search_results = self.file_manager._search_directories(directory_name.lower(), search_root)
                found_directories.extend(search_results)
            
            if found_directories:
                result = f"Directory '{directory_name}' not found, but found similar directories:\n\n"
                
                # Group by drive
                by_drive = {}
                for dir_info in found_directories[:10]:  # Limit to top 10
                    drive = Path(dir_info['path']).drive
                    if drive not in by_drive:
                        by_drive[drive] = []
                    by_drive[drive].append(dir_info)
                
                for drive, dirs in by_drive.items():
                    result += f"[{drive}] Drive:\n"
                    for dir_info in dirs:
                        file_count = dir_info.get('file_count', 0)
                        dir_count = dir_info.get('dir_count', 0)
                        result += f"  [DIR] {dir_info['name']} ({file_count} files, {dir_count} dirs)\n"
                        result += f"        Path: {dir_info['path']}\n"
                    result += "\n"
                
                result += "Try using one of these paths, for example:\n"
                result += f"  'list files in {found_directories[0]['path']}'\n"
                
                return result
            else:
                # Try file search as last resort
                file_search_results = self.file_manager.search_files(directory_name)
                if file_search_results and "error" not in file_search_results[0]:
                    result = f"Directory '{directory_name}' not found, but found files with similar names:\n\n"
                    for file_info in file_search_results[:10]:
                        if file_info.get('is_file'):
                            result += f"  [FILE] {file_info['name']}\n"
                            result += f"        Path: {file_info['path']}\n"
                    return result
                
                return None
                
        except Exception as e:
            logger.error(f"Error in intelligent directory search: {e}")
            return None
    
    def handle_user_profile_command(self) -> str:
        """Handle user profile commands"""
        try:
            profiles = self.file_manager.get_user_profiles()
            if "error" in profiles:
                return f"Error getting user profiles: {profiles['error']}"
            
            result = f"[USER] User Profile Information:\n\n"
            result += f"[HOME] Home Directory: {profiles['user_home']}\n"
            result += f"[USER] Username: {profiles['username']}\n\n"
            result += f"[DIR] Accessible Directories:\n"
            
            for name, path in profiles['accessible_directories'].items():
                if path:
                    try:
                        path_obj = Path(path)
                        if path_obj.exists():
                            # Count files and directories
                            file_count = sum(1 for _ in path_obj.iterdir() if _.is_file())
                            dir_count = sum(1 for _ in path_obj.iterdir() if _.is_dir())
                            result += f"  [OK] {name}: {path} ({file_count} files, {dir_count} directories)\n"
                        else:
                            result += f"  [MISSING] {name}: {path} (does not exist)\n"
                    except:
                        result += f"  [DENIED] {name}: {path} (access denied)\n"
                else:
                    result += f"  [UNAVAILABLE] {name}: Not available on this system\n"
            
            return result
            
        except Exception as e:
            return f"Error handling user profile command: {e}"
