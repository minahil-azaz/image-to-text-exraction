#!/usr/bin/env python3
"""
Test script for long paragraph extraction
This script creates sample images with long paragraphs to test OCR capabilities
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Import our modules
from image_preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from text_processor import TextProcessor

def create_long_paragraph_image():
    """Create a sample image with long paragraphs for testing"""
    print("🎨 Creating long paragraph test image...")
    
    # Create a large image
    width, height = 1200, 1600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except:
            font = ImageFont.load_default()
    
    # Long paragraph text
    long_paragraphs = [
        """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.""",
        
        """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. Pangrams are often used to display font samples and test keyboards. In typography and graphic design, pangrams are used to show the visual weight and style of different fonts. They help designers and typographers evaluate how letters look together in various typefaces.""",
        
        """Artificial intelligence has become an integral part of modern technology. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Deep learning models, particularly neural networks, have revolutionized fields such as computer vision, natural language processing, and speech recognition. These advances have led to the development of sophisticated applications that can understand and interact with humans in increasingly natural ways.""",
        
        """Climate change represents one of the most significant challenges facing humanity in the 21st century. Rising global temperatures, melting polar ice caps, and increasing sea levels threaten ecosystems and human communities worldwide. Scientists have documented the correlation between human activities, particularly the burning of fossil fuels, and the acceleration of global warming. International efforts to reduce greenhouse gas emissions and transition to renewable energy sources are crucial for mitigating these effects and ensuring a sustainable future for generations to come."""
    ]
    
    # Position and draw paragraphs
    y_position = 50
    margin = 50
    line_height = 25
    
    for i, paragraph in enumerate(long_paragraphs):
        # Add paragraph number
        draw.text((margin, y_position), f"Paragraph {i+1}:", fill='black', font=font)
        y_position += line_height + 10
        
        # Split paragraph into lines that fit the image width
        words = paragraph.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] > width - 2 * margin:
                # Line is too long, remove last word and start new line
                current_line.pop()
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw the lines
        for line in lines:
            draw.text((margin, y_position), line, fill='black', font=font)
            y_position += line_height
        
        y_position += 30  # Space between paragraphs
    
    # Save the image
    image.save("long_paragraphs_test.png")
    print("✅ Long paragraph test image created: long_paragraphs_test.png")
    
    return np.array(image)

def test_long_paragraph_extraction():
    """Test OCR extraction on long paragraphs"""
    print("🔍 Testing long paragraph extraction...")
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine()
    text_processor = TextProcessor()
    
    # Create test image
    original_image = create_long_paragraph_image()
    
    print(f"\n📊 Image info:")
    print(f"   Size: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"   Channels: {original_image.shape[2] if len(original_image.shape) == 3 else 1}")
    
    # Test different preprocessing options
    preprocessing_options = {
        'grayscale': True,
        'denoise': True,
        'threshold': True,
        'deskew': False,
        'resize': False
    }
    
    processed_image = preprocessor.preprocess_image(original_image, preprocessing_options)
    print("✅ Image preprocessing completed")
    
    # Test different OCR configurations
    test_configs = ['default', 'long_paragraphs', 'document', 'academic']
    
    for config in test_configs:
        print(f"\n🧪 Testing OCR config: {config}")
        
        if config in ['long_paragraphs', 'document', 'academic']:
            result = ocr_engine.extract_text_optimized_for_paragraphs(
                processed_image,
                language='eng',
                confidence_threshold=60
            )
        else:
            result = ocr_engine.extract_text(
                processed_image,
                language='eng',
                config=config,
                confidence_threshold=60
            )
        
        if result['success']:
            print(f"   ✅ Success - Confidence: {result['confidence']:.1f}%")
            print(f"   📝 Text length: {len(result['text'])} characters")
            
            # Count paragraphs
            paragraphs = [p for p in result['text'].split('\n\n') if p.strip()]
            print(f"   📄 Paragraphs detected: {len(paragraphs)}")
            
            if 'paragraph_count' in result:
                print(f"   📊 Paragraph count (metadata): {result['paragraph_count']}")
                print(f"   📏 Avg paragraph length: {result['avg_paragraph_length']:.1f} words")
            
            # Show first 200 characters
            preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"   📖 Preview: {preview}")
            
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Long paragraph testing completed!")

def test_multilingual_paragraphs():
    """Test paragraph extraction in different languages"""
    print("\n🌍 Testing multilingual paragraph extraction...")
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine()
    
    # Test languages
    test_languages = ['eng', 'fra', 'deu', 'spa', 'ita']
    
    for lang in test_languages:
        print(f"\n🧪 Testing language: {lang} ({ocr_engine.get_language_name(lang)})")
        
        # Create a simple test paragraph in the target language
        test_texts = {
            'eng': "This is a test paragraph in English. It contains multiple sentences to test the OCR system's ability to handle long text and maintain proper paragraph formatting.",
            'fra': "Ceci est un paragraphe de test en français. Il contient plusieurs phrases pour tester la capacité du système OCR à gérer du texte long et maintenir un formatage de paragraphe approprié.",
            'deu': "Dies ist ein Testabsatz auf Deutsch. Er enthält mehrere Sätze, um die Fähigkeit des OCR-Systems zu testen, lange Texte zu verarbeiten und eine ordnungsgemäße Absatzformatierung beizubehalten.",
            'spa': "Este es un párrafo de prueba en español. Contiene múltiples oraciones para probar la capacidad del sistema OCR para manejar texto largo y mantener el formato de párrafo apropiado.",
            'ita': "Questo è un paragrafo di test in italiano. Contiene più frasi per testare la capacità del sistema OCR di gestire testi lunghi e mantenere la formattazione appropriata del paragrafo."
        }
        
        # Create test image with the text
        test_image = create_simple_test_image(test_texts.get(lang, test_texts['eng']))
        
        # Process and extract
        processed = preprocessor.preprocess_image(test_image)
        result = ocr_engine.extract_text_optimized_for_paragraphs(
            processed,
            language=lang,
            confidence_threshold=50
        )
        
        if result['success']:
            print(f"   ✅ Success - Confidence: {result['confidence']:.1f}%")
            print(f"   📝 Extracted: {result['text'][:100]}...")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

def create_simple_test_image(text):
    """Create a simple test image with given text"""
    image = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Split text into lines
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] > 750:
            current_line.pop()
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw lines
    y = 20
    for line in lines:
        draw.text((20, y), line, fill='black', font=font)
        y += 30
    
    return np.array(image)

if __name__ == "__main__":
    print("🧪 Long Paragraph OCR Testing")
    print("=" * 50)
    
    # Test long paragraph extraction
    test_long_paragraph_extraction()
    
    # Test multilingual paragraphs
    test_multilingual_paragraphs()
    
    print(f"\n🎉 All tests completed!")
    print(f"💡 Check the generated images to see the test results.") 