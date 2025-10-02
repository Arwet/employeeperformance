#!/usr/bin/env python3
"""
Launcher script for the INX Future Inc Employee Performance Prediction App
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements_streamlit.txt")
    if requirements_file.exists():
        print("📦 Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Packages installed successfully!")
    else:
        print("⚠️  Requirements file not found. Please ensure requirements_streamlit.txt exists.")

def run_streamlit_app():
    """Run the Streamlit application."""
    app_file = Path("streamlit_app.py")
    if app_file.exists():
        print("🚀 Starting Streamlit application...")
        print("📱 The app will open in your default web browser.")
        print("🌐 If it doesn't open automatically, navigate to: http://localhost:8501")
        print("⏹️  Press Ctrl+C in the terminal to stop the application.")
        print("")
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", str(app_file)
            ])
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user.")
        except Exception as e:
            print(f"❌ Error running application: {e}")
    else:
        print("❌ streamlit_app.py not found. Please ensure you're in the correct directory.")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("🏢 INX Future Inc - Employee Performance Prediction App")
    print("=" * 60)
    print("")
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found in current directory.")
        print("📁 Please navigate to the src directory and run this script again.")
        return
    
    # Install/update requirements if Streamlit is not available
    if not check_streamlit():
        print("📦 Streamlit not found. Installing requirements...")
        install_requirements()
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
