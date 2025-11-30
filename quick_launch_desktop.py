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
    
    # Upgrade pip
    print("Upgrading pip...")
    try:
        install_result = subprocess.run([
            str(venv_python), "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True)
        
        if install_result.returncode != 0:
            print("WARNING: Failed to upgrade pip. Continuing with existing version.")
            print(install_result.stderr)
        else:
            print("SUCCESS: pip upgraded successfully")
    
    except Exception as e:
        print(f"WARNING: Error upgrading pip: {e}")
    
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

    # Ensure Gemini SDK (google-genai) for optional direct usage
    print("Checking Gemini SDK (google-genai)...")
    try:
        result = subprocess.run([
            str(venv_python), "-c", "import importlib; importlib.import_module('google.genai')"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("Installing google-genai...")
            install_result = subprocess.run([
                str(venv_python), "-m", "pip", "install", "google-genai"
            ], capture_output=True, text=True)
            if install_result.returncode != 0:
                print("WARNING: Failed to install google-genai (optional)")
                print(install_result.stderr)
            else:
                print("SUCCESS: google-genai installed")
        else:
            print("SUCCESS: google-genai already installed")
    except Exception as e:
        print(f"WARNING: Error ensuring google-genai: {e}")
    
    # Check if spaCy is installed (optional)
    print("Checking spaCy (optional)...")
    try:
        result = subprocess.run([
            str(venv_python), "-c", "import spacy"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("WARNING: spaCy not installed. NLP will use a lightweight fallback.")
            print("         To install later: venv\\Scripts\\pip install spacy && venv\\Scripts\\python -m spacy download en_core_web_sm")
        else:
            print("SUCCESS: spaCy already installed")
    except Exception as e:
        print(f"WARNING: Error checking spaCy: {e}")
        print("         Continuing without spaCy.")
    
    # Ensure core Python dependencies
    print("Checking core Python packages (streamlit, psutil, requests)...")
    for mod_name, pip_name in [("streamlit", "streamlit"), ("psutil", "psutil"), ("requests", "requests")]:
        try:
            result = subprocess.run([
                str(venv_python), "-c", f"import {mod_name}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Installing {pip_name}...")
                install_result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", pip_name
                ], capture_output=True, text=True)
                if install_result.returncode != 0:
                    print(f"ERROR: Failed to install {pip_name}")
                    print(install_result.stderr)
                    input("Press Enter to exit...")
                    return False
                print(f"SUCCESS: {pip_name} installed")
            else:
                print(f"SUCCESS: {pip_name} already installed")
        except Exception as e:
            print(f"ERROR: Error ensuring {pip_name}: {e}")
            input("Press Enter to exit...")
            return False

    # Ensure AI dependencies used by JARVIS
    print("Checking AI packages (langchain-core, langchain-groq, langchain-google-genai, python-dotenv)...")
    for mod_name, pip_name in [
        ("langchain_core", "langchain-core"),
        ("langchain_groq", "langchain-groq"),
        ("langchain_google_genai", "langchain-google-genai"),
        ("dotenv", "python-dotenv"),
    ]:
        try:
            result = subprocess.run([
                str(venv_python), "-c", f"import {mod_name}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Installing {pip_name}...")
                install_result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", pip_name
                ], capture_output=True, text=True)
                if install_result.returncode != 0:
                    print(f"ERROR: Failed to install {pip_name}")
                    print(install_result.stderr)
                    input("Press Enter to exit...")
                    return False
                print(f"SUCCESS: {pip_name} installed")
            else:
                print(f"SUCCESS: {pip_name} already installed")
        except Exception as e:
            print(f"ERROR: Error ensuring {pip_name}: {e}")
            input("Press Enter to exit...")
            return False

    # Ensure PDF and image reading packages
    print("Checking PDF/image packages (pypdf, pillow)...")
    for mod_name, pip_name in [
        ("pypdf", "pypdf"),
        ("PIL", "pillow"),
    ]:
        try:
            result = subprocess.run([
                str(venv_python), "-c", f"import {mod_name}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Installing {pip_name}...")
                install_result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", pip_name
                ], capture_output=True, text=True)
                if install_result.returncode != 0:
                    print(f"WARNING: Failed to install {pip_name} (optional for PDF/image reading)")
                    print(install_result.stderr)
                else:
                    print(f"SUCCESS: {pip_name} installed")
            else:
                print(f"SUCCESS: {pip_name} already installed")
        except Exception as e:
            print(f"WARNING: Error ensuring {pip_name}: {e}")

    # Ensure Google API dependencies for Gmail integration
    print("Checking Google API packages (google-api-python-client, google-auth-oauthlib)...")
    for mod_name, pip_name in [
        ("googleapiclient.discovery", "google-api-python-client"),
        ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
    ]:
        try:
            result = subprocess.run([
                str(venv_python), "-c", f"import {mod_name}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Installing {pip_name}...")
                install_result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", pip_name
                ], capture_output=True, text=True)
                if install_result.returncode != 0:
                    print(f"ERROR: Failed to install {pip_name}")
                    print(install_result.stderr)
                    input("Press Enter to exit...")
                    return False
                print(f"SUCCESS: {pip_name} installed")
            else:
                print(f"SUCCESS: {pip_name} already installed")
        except Exception as e:
            print(f"ERROR: Error ensuring {pip_name}: {e}")
            input("Press Enter to exit...")
            return False

    # Launch JARVIS Desktop
    print("Launching JARVIS Desktop Application...")
    try:
        env = os.environ.copy()
        venv_dir = str(Path("venv").resolve())
        venv_scripts = str(Path("venv/Scripts").resolve())
        env["VIRTUAL_ENV"] = venv_dir
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
        subprocess.run([str(venv_python), "jarvis_desktop.py"], env=env)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"ERROR: Error launching JARVIS Desktop: {e}")
        input("Press Enter to exit...")
        return False
    
    return True

if __name__ == "__main__":
    main()
