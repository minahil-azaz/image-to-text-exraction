# 🔍 Image-to-Text (OCR) Extraction System

An intelligent AI-powered system that extracts readable and editable text from images using Optical Character Recognition (OCR) with advanced image preprocessing techniques.

## ✨ Features

- **Image Upload**: Support for JPG, PNG, and other common image formats
- **Advanced Preprocessing**: Grayscale conversion, thresholding, denoising using OpenCV
- **OCR Text Extraction**: Powered by Tesseract OCR engine
- **Multi-language Support**: Recognize text in multiple languages
- **Export Options**: Save extracted text as .txt or .docx files
- **Text Translation**: Translate extracted text using Google Translate API
- **Text-to-Speech**: Convert extracted text to audio (optional)
- **Web Interface**: Beautiful and intuitive Streamlit-based GUI

## 🛠️ Installation

### Prerequisites

1. **Install Tesseract OCR**:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

 **Run the application**:
   ```bash
   streamlit run app.py
   ```





