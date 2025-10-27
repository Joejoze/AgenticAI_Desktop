# 💬 JARVIS Chat Sessions

A comprehensive chat session management system for JARVIS with persistent storage and sidebar navigation.

## ✨ Features

### 🗂️ **Session Management**
- **Multiple Chat Sessions** - Create unlimited chat sessions
- **Persistent Storage** - All chats saved in SQLite database
- **Session Switching** - Easy switching between different conversations
- **Session Renaming** - Customize session names
- **Session Deletion** - Remove unwanted sessions

### 🎯 **User Interface**
- **Sidebar Navigation** - Clean sidebar with all your chat sessions
- **Active Session Indicator** - See which session is currently active
- **Message Count** - View message count for each session
- **Last Activity** - See when each session was last used
- **One-Click Switching** - Click any session to switch to it

### 💾 **Data Storage**
- **SQLite Database** - Lightweight, local database storage
- **Message History** - Complete conversation history preserved
- **Metadata Support** - Store additional message metadata
- **Automatic Cleanup** - Optional cleanup of old sessions

## 🚀 Quick Start

### Option 1: Desktop Application (Recommended)
```bash
python quick_launch_desktop.py
```

### Option 2: Browser Interface
```bash
python run_jarvis_sessions.py
```

### Option 3: Manual Launch
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Launch with sessions
python -m streamlit run jarvis_streamlit_sessions.py
```

## 📋 How to Use

### Creating a New Chat
1. Click the **"➕ New Chat"** button in the sidebar
2. A new session will be created automatically
3. Start chatting immediately

### Switching Between Chats
1. Click on any session name in the sidebar
2. The chat will switch instantly
3. Your conversation history is preserved

### Renaming a Session
1. Select the session you want to rename
2. Enter a new name in the "Rename current session" field
3. Click **"✏️ Rename"**

### Deleting a Session
1. Click the **🗑️** button next to any session
2. The session and all its messages will be deleted
3. If you delete the current session, a new one will be created

## 🗄️ Database Schema

### Sessions Table
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,           -- Unique session ID
    name TEXT NOT NULL,            -- Session display name
    created_at TIMESTAMP,          -- Creation timestamp
    updated_at TIMESTAMP,          -- Last update timestamp
    is_active BOOLEAN DEFAULT 1    -- Currently active session
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,        -- Auto-increment message ID
    session_id TEXT NOT NULL,      -- Foreign key to sessions
    role TEXT NOT NULL,            -- 'user' or 'assistant'
    content TEXT NOT NULL,         -- Message content
    timestamp TIMESTAMP,           -- Message timestamp
    metadata TEXT,                 -- JSON metadata
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
```

## 🔧 API Reference

### ChatDatabase Class

#### `create_session(name=None) -> str`
Creates a new chat session and returns the session ID.

#### `get_active_session() -> Dict`
Gets the currently active session.

#### `get_all_sessions() -> List[Dict]`
Gets all chat sessions ordered by most recent.

#### `switch_to_session(session_id) -> bool`
Switches to a specific session.

#### `add_message(session_id, role, content, metadata=None) -> int`
Adds a message to a session.

#### `get_messages(session_id, limit=None) -> List[Dict]`
Gets messages for a session.

#### `delete_session(session_id) -> bool`
Deletes a session and all its messages.

#### `rename_session(session_id, new_name) -> bool`
Renames a session.

#### `get_session_stats(session_id) -> Dict`
Gets statistics for a session.

## 🎨 UI Components

### Sidebar Features
- **Session List** - Shows all your chat sessions
- **Active Indicator** - Highlights the current session
- **Message Count** - Shows number of messages per session
- **Last Activity** - Shows when session was last updated
- **New Chat Button** - Creates a new session
- **Delete Buttons** - Remove unwanted sessions
- **Rename Interface** - Change session names

### Main Chat Area
- **Session Title** - Shows current session name
- **Message History** - Displays all messages in the session
- **Chat Input** - Type your messages here
- **Status Bar** - Shows connection status and message count

## 🔒 Data Security

- **Local Storage** - All data stored locally on your machine
- **No Cloud Sync** - Your conversations stay private
- **SQLite Encryption** - Optional database encryption support
- **Automatic Backups** - Database file can be backed up easily

## 🚀 Performance

- **Fast Switching** - Instant session switching
- **Efficient Queries** - Optimized database queries
- **Memory Efficient** - Only loads current session messages
- **Scalable** - Handles thousands of sessions and messages

## 🛠️ Troubleshooting

### "Database not found"
- The database is created automatically on first run
- Check file permissions in the project directory

### "Session not switching"
- Refresh the page and try again
- Check if the session exists in the database

### "Messages not saving"
- Check database file permissions
- Ensure the session ID is valid

### "Performance issues"
- Consider cleaning up old sessions
- Check database file size

## 🎉 Benefits

### For Users
- **Organized Conversations** - Keep different topics separate
- **Easy Navigation** - Quick access to any conversation
- **Persistent History** - Never lose your chat history
- **Custom Organization** - Name sessions as you like

### For Developers
- **Clean Architecture** - Well-structured database design
- **Easy Integration** - Simple API for session management
- **Extensible** - Easy to add new features
- **Maintainable** - Clear separation of concerns

**Your JARVIS now has professional-grade chat session management! 🎉**
