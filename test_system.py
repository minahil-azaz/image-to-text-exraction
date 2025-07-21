#!/usr/bin/env python3
"""
System test script for Image-to-Text OCR System
This script tests all components to ensure they're working correctly
"""

import sys
import os
import traceback
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✅ PyTesseract imported successfully")
    except ImportError as e:
        print(f"❌ PyTesseract import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"❌ PIL/Pillow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test if our custom modules can be imported"""
    print("\n🔍 Testing custom modules...")
    
    try:
        from image_preprocessor import ImagePreprocessor
        print("✅ ImagePreprocessor imported successfully")
    except ImportError as e:
        print(f"❌ ImagePreprocessor import failed: {e}")
        return False
    
    try:
        from ocr_engine import OCREngine
        print("✅ OCREngine imported successfully")
    except ImportError as e:
        print(f"❌ OCREngine import failed: {e}")
        return False
    
    try:
        from text_processor import TextProcessor
        print("✅ TextProcessor imported successfully")
    except ImportError as e:
        print(f"❌ TextProcessor import failed: {e}")
        return False
    
    try:
        import utils
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"❌ Utils module import failed: {e}")
        return False
    
    return True

def test_tesseract():
    """Test if Tesseract is working"""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Test available languages
        languages = pytesseract.get_languages()
        print(f"✅ Available languages: {len(languages)} languages")
        print(f"   Sample languages: {languages[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def test_image_processing():
    """Test image processing functionality"""
    print("\n🔍 Testing image processing...")
    
    try:
        import numpy as np
        from PIL import Image
        from image_preprocessor import ImagePreprocessor
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test preprocessor
        preprocessor = ImagePreprocessor()
        
        # Test preprocessing
        processed = preprocessor.preprocess_image(test_image)
        print("✅ Image preprocessing test passed")
        
        # Test utility functions
        import utils
        info = utils.get_image_info(test_image)
        print("✅ Image info extraction test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        traceback.print_exc()
        return False

def test_ocr_functionality():
    """Test OCR functionality"""
    print("\n🔍 Testing OCR functionality...")
    
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from ocr_engine import OCREngine
        
        # Create a simple test image with text
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 40), "TEST", fill='black', font=font)
        test_image = np.array(img)
        
        # Test OCR engine
        ocr_engine = OCREngine()
        
        # Test text extraction
        result = ocr_engine.extract_text(test_image, language='eng')
        print("✅ OCR text extraction test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_text_processing():
    """Test text processing functionality"""
    print("\n🔍 Testing text processing...")
    
    try:
        from text_processor import TextProcessor
        
        text_processor = TextProcessor()
        
        # Test text cleaning
        test_text = "  Hello   World!  "
        cleaned = text_processor.clean_text(test_text)
        print("✅ Text cleaning test passed")
        
        # Test word count
        stats = text_processor.get_word_count(test_text)
        print("✅ Word count test passed")
        
        # Test structured data extraction
        test_text_with_data = "Contact us at test@example.com or call 123-456-7890"
        structured = text_processor.extract_structured_data(test_text_with_data)
        print("✅ Structured data extraction test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        traceback.print_exc()
        return False

def test_export_functionality():
    """Test export functionality"""
    print("\n🔍 Testing export functionality...")
    
    try:
        from text_processor import TextProcessor
        
        text_processor = TextProcessor()
        test_text = "This is a test text for export functionality."
        
        # Test TXT export
        txt_data = text_processor.export_to_txt(test_text)
        print("✅ TXT export test passed")
        
        # Test DOCX export (if available)
        try:
            docx_data = text_processor.export_to_docx(test_text)
            print("✅ DOCX export test passed")
        except ImportError:
            print("⚠️ DOCX export not available (python-docx not installed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Export functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🧪 Image-to-Text OCR System - Comprehensive Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Tesseract OCR", test_tesseract),
        ("Image Processing", test_image_processing),
        ("OCR Functionality", test_ocr_functionality),
        ("Text Processing", test_text_processing),
        ("Export Functionality", test_export_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\n🚀 You can now run:")
        print("   - python demo.py (for command-line demo)")
        print("   - streamlit run app.py (for web interface)")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Install Tesseract OCR")
        print("   - Check system requirements")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 