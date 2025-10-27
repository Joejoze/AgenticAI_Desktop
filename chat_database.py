#!/usr/bin/env python3
"""
Chat Session Database Manager
============================

Manages chat sessions and messages using SQLite database.
Provides functionality to create, retrieve, update, and delete chat sessions.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

class ChatDatabase:
    """Manages chat sessions and messages in SQLite database"""
    
    def __init__(self, db_path: str = "chat_sessions.db"):
        """Initialize the database connection"""
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Create index for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages (session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages (timestamp)
            """)
            
            conn.commit()
    
    def create_session(self, name: str = None) -> str:
        """Create a new chat session"""
        if not name:
            name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Deactivate all other sessions
            cursor.execute("UPDATE sessions SET is_active = 0")
            
            # Create new session
            cursor.execute("""
                INSERT INTO sessions (id, name, is_active)
                VALUES (?, ?, 1)
            """, (session_id, name))
            
            conn.commit()
        
        return session_id
    
    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """Get the currently active session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, created_at, updated_at
                FROM sessions 
                WHERE is_active = 1
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'name': result[1],
                    'created_at': result[2],
                    'updated_at': result[3]
                }
        return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions ordered by most recent"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, created_at, updated_at, is_active
                FROM sessions 
                ORDER BY updated_at DESC
            """)
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[2],
                    'updated_at': row[3],
                    'is_active': bool(row[4])
                })
            
            return sessions
    
    def switch_to_session(self, session_id: str) -> bool:
        """Switch to a specific session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Deactivate all sessions
            cursor.execute("UPDATE sessions SET is_active = 0")
            
            # Activate the selected session
            cursor.execute("""
                UPDATE sessions 
                SET is_active = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> int:
        """Add a message to a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Add message
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, json.dumps(metadata) if metadata else None))
            
            # Update session timestamp
            cursor.execute("""
                UPDATE sessions 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_messages(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get messages for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT id, role, content, timestamp, metadata
                FROM messages 
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row[0],
                    'role': row[1],
                    'content': row[2],
                    'timestamp': row[3],
                    'metadata': json.loads(row[4]) if row[4] else None
                })
            
            return messages
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete messages first (foreign key constraint)
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_name, session_id))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get message count
            cursor.execute("""
                SELECT COUNT(*) FROM messages WHERE session_id = ?
            """, (session_id,))
            message_count = cursor.fetchone()[0]
            
            # Get first and last message timestamps
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM messages WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            first_message = result[0] if result[0] else None
            last_message = result[1] if result[1] else None
            
            return {
                'message_count': message_count,
                'first_message': first_message,
                'last_message': last_message
            }
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up sessions older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old sessions and their messages
            cursor.execute("""
                DELETE FROM sessions 
                WHERE updated_at < datetime('now', '-{} days')
            """.format(days))
            
            conn.commit()
            return cursor.rowcount

# Global database instance
chat_db = ChatDatabase()
