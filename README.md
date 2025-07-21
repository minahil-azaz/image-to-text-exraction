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

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Upload an image** containing text

4. **Configure settings** (language, preprocessing options)

5. **Extract text** and download results

## 📁 Project Structure

```
image-to-text/
├── app.py                 # Main Streamlit application
├── ocr_engine.py          # Core OCR processing logic
├── image_preprocessor.py  # Image preprocessing functions
├── text_processor.py      # Text processing and export utilities
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

The system supports various configuration options:
- **OCR Language**: Choose from multiple languages
- **Preprocessing**: Enable/disable image enhancement
- **Export Format**: TXT or DOCX
- **Translation**: Target language for translation

## 📝 License

This project is open source and available under the MIT License. # image-to-text-exraction
