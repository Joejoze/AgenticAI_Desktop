#!/usr/bin/env python3
"""
Quick Launch JARVIS Desktop Application
======================================

This script automatically activates the virtual environment and launches JARVIS Desktop.
No need to manually activate venv first.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Quick launch JARVIS Desktop"""
    print("Quick Launch JARVIS Desktop")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("jarvis_streamlit_sessions.py").exists():
        print("ERROR: jarvis_streamlit_sessions.py not found!")
        print("Please run this script from the AgenticAI directory")
        input("Press Enter to exit...")
        return False
    
    # Check if virtual environment exists
    venv_python = Path("venv/Scripts/python.exe")
    if not venv_python.exists():
        print("ERROR: Virtual environment not found!")
        print("Please run: python setup_venv.py")
        input("Press Enter to exit...")
        return False
    
    print("SUCCESS: Virtual environment found")
    
    # Check if PyWebView is installed
    print("Checking PyWebView...")
    try:
        result = subprocess.run([
            str(venv_python), "-c", "import webview"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Installing PyWebView...")
            install_result = subprocess.run([
                str(venv_python), "-m", "pip", "install", "pywebview"
            ], capture_output=True, text=True)
            
            if install_result.returncode != 0:
                print("ERROR: Failed to install PyWebView")
                print(install_result.stderr)
                input("Press Enter to exit...")
                return False
            
            print("SUCCESS: PyWebView installed successfully")
        else:
            print("SUCCESS: PyWebView already installed")
    
    except Exception as e:
        print(f"ERROR: Error checking PyWebView: {e}")
        input("Press Enter to exit...")
        return False
    
    # Launch JARVIS Desktop
    print("Launching JARVIS Desktop Application...")
    try:
        subprocess.run([str(venv_python), "jarvis_desktop.py"])
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"ERROR: Error launching JARVIS Desktop: {e}")
        input("Press Enter to exit...")
        return False
    
    return True

if __name__ == "__main__":
    main()
