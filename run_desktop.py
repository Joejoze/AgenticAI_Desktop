#!/usr/bin/env python3
"""
Email Agent Desktop Application Launcher

This script launches the Email Agent as a desktop application.
Make sure you have all dependencies installed and your GROQ_API_KEY set.
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import tkinter
        from agent import fetch_recent_emails, process_email
        from emailapi import get_service, send_email, get_user_email
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_environment():
    """Check if environment variables are set"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Warning: GROQ_API_KEY environment variable not set.")
        print("Classification and reply drafting will not work.")
        print("Set it with: export GROQ_API_KEY=your_key_here")
        return False
    return True

def main():
    """Main launcher function"""
    print("📬 Email Agent Desktop Application")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Launch the desktop app
    try:
        from desktop_app import main as launch_app
        print("Starting Email Agent Desktop...")
        launch_app()
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
