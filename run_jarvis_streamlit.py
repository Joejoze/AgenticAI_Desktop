#!/usr/bin/env python3
"""
JARVIS Streamlit Quick Launcher
===============================

Quick launcher for JARVIS Streamlit interface
Works with virtual environment
"""

import subprocess
import sys
import os
from pathlib import Path

def check_venv():
    """Check if virtual environment is activated"""
    if not os.environ.get('VIRTUAL_ENV'):
        print("WARNING: Virtual environment not detected")
        print("Please activate your virtual environment first:")
        print("Windows: venv\\Scripts\\activate")
        print("Linux/Mac: source venv/bin/activate")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import psutil
        return True
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please run: python setup_venv.py")
        return False

def main():
    """Launch JARVIS Streamlit app"""
    print("JARVIS AI Assistant - Streamlit Interface")
    print("=" * 50)
    
    # Check virtual environment
    if not check_venv():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    print("SUCCESS: Virtual environment active")
    print("SUCCESS: Dependencies available")
    print("Launching Streamlit app...")
    print("The app will open in your browser at: http://localhost:8501")
    print("WARNING: Make sure you have GROQ_API_KEY in your .env file")
    print("WARNING: Ensure credentials.json is configured for Gmail")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "jarvis_streamlit.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        except KeyboardInterrupt:
            print("\nJARVIS Streamlit app stopped")
        except Exception as e:
            print(f"ERROR: Error launching Streamlit: {e}")

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nERROR: Launch failed. Please fix the issues above.")
        sys.exit(1)
