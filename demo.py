#!/usr/bin/env python3
"""
Demo script for Image-to-Text OCR System
This script demonstrates the core functionality without the web interface
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Import our modules
from image_preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from text_processor import TextProcessor

def create_sample_image():
    """Create a sample image with text for testing"""
    print("🎨 Creating sample image...")
    
    # Create a white image
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Add sample text
    sample_text = [
        "Hello World!",
        "This is a sample text for OCR testing.",
        "1234567890",
        "Special characters: @#$%^&*()",
        "Multiple lines of text",
        "for demonstration purposes."
    ]
    
    y_position = 50
    for line in sample_text:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 40
    
    # Save the image
    image.save("sample_image.png")
    print("✅ Sample image created: sample_image.png")
    
    return np.array(image)

def run_demo():
    """Run the OCR demo"""
    print("🔍 Image-to-Text OCR System - Demo")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine()
    text_processor = TextProcessor()
    
    # Create sample image
    original_image = create_sample_image()
    
    print(f"\n📊 Original image info:")
    print(f"   Size: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"   Channels: {original_image.shape[2] if len(original_image.shape) == 3 else 1}")
    
    # Preprocess image
    print("\n🖼️ Preprocessing image...")
    preprocessing_options = {
        'grayscale': True,
        'denoise': True,
        'threshold': True,
        'deskew': False,
        'resize': False
    }
    
    processed_image = preprocessor.preprocess_image(original_image, preprocessing_options)
    print("✅ Image preprocessing completed")
    
    # Extract text
    print("\n📝 Extracting text...")
    ocr_results = ocr_engine.extract_text(
        processed_image,
        language='eng',
        config='default',
        confidence_threshold=60
    )
    
    if ocr_results['success']:
        print("✅ Text extraction completed")
        print(f"   Confidence: {ocr_results['confidence']:.1f}%")
        print(f"   Words detected: {len(ocr_results['lines'])}")
        
        # Display extracted text
        print(f"\n📄 Extracted Text:")
        print("-" * 40)
        print(ocr_results['text'])
        print("-" * 40)
        
        # Text statistics
        text_stats = text_processor.get_word_count(ocr_results['text'])
        print(f"\n📊 Text Statistics:")
        print(f"   Characters: {text_stats['characters']}")
        print(f"   Words: {text_stats['words']}")
        print(f"   Sentences: {text_stats['sentences']}")
        print(f"   Paragraphs: {text_stats['paragraphs']}")
        
        # Clean text
        cleaned_text = text_processor.clean_text(ocr_results['text'])
        print(f"\n🧹 Cleaned Text:")
        print("-" * 40)
        print(cleaned_text)
        print("-" * 40)
        
        # Extract structured data
        structured_data = text_processor.extract_structured_data(ocr_results['text'])
        print(f"\n🔍 Structured Data:")
        if structured_data['numbers']:
            print(f"   Numbers: {', '.join(structured_data['numbers'][:5])}")
        if structured_data['emails']:
            print(f"   Emails: {', '.join(structured_data['emails'])}")
        if structured_data['urls']:
            print(f"   URLs: {', '.join(structured_data['urls'])}")
        
        # Test translation (if available)
        if hasattr(text_processor, 'translator') and text_processor.translator:
            print(f"\n🌍 Testing translation...")
            translation_result = text_processor.translate_text(ocr_results['text'], 'es')
            if translation_result['success']:
                print(f"   Translated to Spanish:")
                print(f"   {translation_result['translated_text'][:100]}...")
        
        # Export options
        print(f"\n📤 Export options available:")
        print(f"   - TXT export")
        print(f"   - DOCX export")
        print(f"   - Text-to-speech")
        
    else:
        print("❌ Text extraction failed")
        print(f"   Error: {ocr_results.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 To run the full web interface:")
    print(f"   streamlit run app.py")

if __name__ == "__main__":
    run_demo() 