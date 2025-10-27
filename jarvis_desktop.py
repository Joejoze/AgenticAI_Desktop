#!/usr/bin/env python3
"""
JARVIS Desktop Application
==========================

A native desktop application for JARVIS using PyWebView.
This creates a true desktop window (no browser UI) for the JARVIS interface.
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    print("ERROR: PyWebView not installed. Run: pip install pywebview")

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if we're in the right directory
    if not Path("jarvis_streamlit.py").exists():
        print("ERROR: jarvis_streamlit.py not found. Please run from the correct directory.")
        return False
    
    # Check if virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("WARNING: Virtual environment not detected")
        print("   For best results, activate your virtual environment first:")
        print("   Windows: venv\\Scripts\\activate")
        print("   Linux/Mac: source venv/bin/activate")
    
    # Check required dependencies
    try:
        import streamlit
        import psutil
        print("SUCCESS: Streamlit and psutil available")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please run: pip install streamlit psutil")
        return False
    
    # Check environment variables
    required_env = ["GROQ_API_KEY"]
    missing_env = []
    
    for env_var in required_env:
        if not os.getenv(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        print(f"WARNING: Missing environment variables: {', '.join(missing_env)}")
        print("   Please set these in your .env file")
    
    print("SUCCESS: Requirements check completed")
    return True

def start_streamlit_server():
    """Start Streamlit server in the background"""
    print("Starting Streamlit server...")
    
    try:
        # Start Streamlit server with session-based interface
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "jarvis_streamlit_sessions.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("SUCCESS: Streamlit server started successfully")
            return process
        else:
            print("ERROR: Failed to start Streamlit server")
            return None
            
    except Exception as e:
        print(f"ERROR: Error starting Streamlit server: {e}")
        return None

def wait_for_server():
    """Wait for Streamlit server to be ready"""
    import requests
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8501", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
        print(f"Waiting for server... ({attempt + 1}/{max_attempts})")
    
    return False

def create_desktop_window():
    """Create the desktop window using PyWebView"""
    print("Creating desktop window...")
    
    try:
        # Create the webview window
        webview.create_window(
            title="JARVIS AI Assistant",
            url="http://localhost:8501",
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            fullscreen=False,
            minimized=False,
            on_top=False,
            shadow=True,
            focus=True
        )
        
        print("SUCCESS: Desktop window created")
        return True
        
    except Exception as e:
        print(f"ERROR: Error creating desktop window: {e}")
        return False

def main():
    """Main entry point"""
    print("JARVIS Desktop Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\nERROR: Requirements check failed")
        input("Press Enter to exit...")
        return False
    
    # Check if PyWebView is available
    if not WEBVIEW_AVAILABLE:
        print("\nERROR: PyWebView not available")
        print("Please install with: pip install pywebview")
        input("Press Enter to exit...")
        return False
    
    print("\nStarting JARVIS Desktop Application...")
    
    # Start Streamlit server in background
    streamlit_process = start_streamlit_server()
    if not streamlit_process:
        print("\nERROR: Failed to start Streamlit server")
        input("Press Enter to exit...")
        return False
    
    # Wait for server to be ready
    if not wait_for_server():
        print("\nERROR: Streamlit server failed to start")
        streamlit_process.terminate()
        input("Press Enter to exit...")
        return False
    
    print("SUCCESS: Server is ready!")
    print("Opening desktop window...")
    
    try:
        # Create the webview window first
        webview.create_window(
            title="JARVIS AI Assistant",
            url="http://localhost:8501",
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            fullscreen=False,
            minimized=False,
            on_top=False,
            shadow=True,
            focus=True
        )
        
        # Start the webview
        webview.start(
            debug=False,
            http_server=False,
            private_mode=False
        )
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"\nERROR: Error running desktop application: {e}")
    finally:
        # Clean up: terminate Streamlit process
        if streamlit_process and streamlit_process.poll() is None:
            print("Stopping Streamlit server...")
            streamlit_process.terminate()
            streamlit_process.wait()
            print("SUCCESS: Streamlit server stopped")
    
    print("JARVIS Desktop Application closed")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
