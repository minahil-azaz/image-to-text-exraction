#!/usr/bin/env python3
"""
Installation script for Image-to-Text OCR System
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def check_tesseract():
    """Check if Tesseract is installed"""
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Tesseract OCR is not installed")
    return False

def install_tesseract_instructions():
    """Provide instructions for installing Tesseract"""
    system = platform.system().lower()
    
    print("\n📋 Tesseract OCR Installation Instructions:")
    print("=" * 50)
    
    if system == "darwin":  # macOS
        print("🍎 macOS:")
        print("   brew install tesseract")
        print("   brew install tesseract-lang  # For additional languages")
        
    elif system == "linux":
        print("🐧 Linux (Ubuntu/Debian):")
        print("   sudo apt-get update")
        print("   sudo apt-get install tesseract-ocr")
        print("   sudo apt-get install tesseract-ocr-[lang]  # For specific languages")
        print("\n   Linux (CentOS/RHEL):")
        print("   sudo yum install tesseract")
        
    elif system == "windows":
        print("🪟 Windows:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Run the installer")
        print("   3. Add Tesseract to your PATH environment variable")
        
    print("\n💡 After installation, restart your terminal/command prompt")

def test_installation():
    """Test if the installation is working"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import pytesseract
        import streamlit
        from PIL import Image
        import numpy as np
        
        print("✅ All Python packages imported successfully")
        
        # Test Tesseract
        if check_tesseract():
            print("✅ Tesseract is working")
            return True
        else:
            print("❌ Tesseract is not working")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation function"""
    print("🔍 Image-to-Text OCR System - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install Python dependencies
    if not install_python_dependencies():
        return
    
    # Check Tesseract
    if not check_tesseract():
        install_tesseract_instructions()
        return
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
    else:
        print("\n❌ Installation failed. Please check the errors above.")

if __name__ == "__main__":
    main() 