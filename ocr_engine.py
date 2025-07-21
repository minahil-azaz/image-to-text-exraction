import pytesseract
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from typing import Dict, List, Tuple, Optional

class OCREngine:
    """
    Core OCR engine using Tesseract for text extraction
    """
    
    def __init__(self):
        # Complete language mapping for all 125+ Tesseract languages
        self.supported_languages = {
            'afr': 'Afrikaans',
            'amh': 'Amharic',
            'ara': 'Arabic',
            'asm': 'Assamese',
            'aze': 'Azerbaijani',
            'aze_cyrl': 'Azerbaijani (Cyrillic)',
            'bel': 'Belarusian',
            'ben': 'Bengali',
            'bod': 'Tibetan',
            'bos': 'Bosnian',
            'bre': 'Breton',
            'bul': 'Bulgarian',
            'cat': 'Catalan',
            'ceb': 'Cebuano',
            'ces': 'Czech',
            'chi_sim': 'Chinese (Simplified)',
            'chi_sim_vert': 'Chinese (Simplified, Vertical)',
            'chi_tra': 'Chinese (Traditional)',
            'chi_tra_vert': 'Chinese (Traditional, Vertical)',
            'chr': 'Cherokee',
            'cos': 'Corsican',
            'cym': 'Welsh',
            'dan': 'Danish',
            'deu': 'German',
            'div': 'Dhivehi',
            'dzo': 'Dzongkha',
            'ell': 'Greek',
            'enm': 'English (Middle)',
            'eng': 'English',
            'epo': 'Esperanto',
            'equ': 'Math/Equation',
            'est': 'Estonian',
            'eus': 'Basque',
            'fao': 'Faroese',
            'fas': 'Persian',
            'fil': 'Filipino',
            'fin': 'Finnish',
            'fra': 'French',
            'frk': 'German (Frankish)',
            'frm': 'French (Middle)',
            'fry': 'Frisian',
            'gla': 'Scottish Gaelic',
            'gle': 'Irish',
            'glg': 'Galician',
            'grc': 'Greek (Ancient)',
            'guj': 'Gujarati',
            'hat': 'Haitian Creole',
            'heb': 'Hebrew',
            'hin': 'Hindi',
            'hrv': 'Croatian',
            'hun': 'Hungarian',
            'hye': 'Armenian',
            'iku': 'Inuktitut',
            'ind': 'Indonesian',
            'isl': 'Icelandic',
            'ita': 'Italian',
            'ita_old': 'Italian (Old)',
            'jav': 'Javanese',
            'jpn': 'Japanese',
            'jpn_vert': 'Japanese (Vertical)',
            'kan': 'Kannada',
            'kat': 'Georgian',
            'kat_old': 'Georgian (Old)',
            'kaz': 'Kazakh',
            'khm': 'Khmer',
            'kir': 'Kyrgyz',
            'kmr': 'Kurdish (Kurmanji)',
            'kor': 'Korean',
            'kor_vert': 'Korean (Vertical)',
            'lao': 'Lao',
            'lat': 'Latin',
            'lav': 'Latvian',
            'lit': 'Lithuanian',
            'ltz': 'Luxembourgish',
            'mal': 'Malayalam',
            'mar': 'Marathi',
            'mkd': 'Macedonian',
            'mlt': 'Maltese',
            'mon': 'Mongolian',
            'mri': 'Maori',
            'msa': 'Malay',
            'mya': 'Burmese',
            'nep': 'Nepali',
            'nld': 'Dutch',
            'nor': 'Norwegian',
            'oci': 'Occitan',
            'osd': 'Orientation and Script Detection',
            'pan': 'Punjabi',
            'pol': 'Polish',
            'por': 'Portuguese',
            'pus': 'Pashto',
            'que': 'Quechua',
            'ron': 'Romanian',
            'rus': 'Russian',
            'san': 'Sanskrit',
            'sin': 'Sinhala',
            'slk': 'Slovak',
            'slv': 'Slovenian',
            'snd': 'Sindhi',
            'spa': 'Spanish',
            'spa_old': 'Spanish (Old)',
            'sqi': 'Albanian',
            'srp': 'Serbian',
            'srp_latn': 'Serbian (Latin)',
            'sun': 'Sundanese',
            'swa': 'Swahili',
            'swe': 'Swedish',
            'syr': 'Syriac',
            'tam': 'Tamil',
            'tat': 'Tatar',
            'tel': 'Telugu',
            'tgk': 'Tajik',
            'tha': 'Thai',
            'tir': 'Tigrinya',
            'ton': 'Tongan',
            'tur': 'Turkish',
            'uig': 'Uyghur',
            'ukr': 'Ukrainian',
            'urd': 'Urdu',
            'uzb': 'Uzbek',
            'uzb_cyrl': 'Uzbek (Cyrillic)',
            'vie': 'Vietnamese',
            'yid': 'Yiddish',
            'yor': 'Yoruba'
        }
        
        # OCR configuration options - optimized for long paragraphs and documents
        self.ocr_configs = {
            'default': '--oem 3 --psm 6',
            'paragraphs': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'long_paragraphs': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'document': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'single_line': '--oem 3 --psm 7',
            'single_word': '--oem 3 --psm 8',
            'single_char': '--oem 3 --psm 10',
            'sparse_text': '--oem 3 --psm 11',
            'sparse_text_osd': '--oem 3 --psm 12',
            'raw_line': '--oem 3 --psm 13',
            'uniform_block': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            'numbers_only': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            'letters_only': '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            'long_text': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'academic': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'newspaper': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'handwritten': '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        }
    
    def extract_text(self, image, language='eng', config='default', confidence_threshold=60):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image: Preprocessed image (numpy array or PIL Image)
            language: Language code for OCR
            config: OCR configuration preset
            confidence_threshold: Minimum confidence score for text
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Get OCR configuration
            ocr_config = self.ocr_configs.get(config, self.ocr_configs['default'])
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                lang=language, 
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results with improved paragraph handling
            extracted_text = []
            confidence_scores = []
            bounding_boxes = []
            current_line = []
            current_line_y = -1
            line_height = 0
            
            # Group text by lines and preserve paragraph structure
            for i, conf in enumerate(data['conf']):
                if conf > confidence_threshold:
                    text = data['text'][i].strip()
                    if text:
                        y_pos = data['top'][i]
                        height = data['height'][i]
                        
                        # Check if this is a new line
                        if current_line_y == -1:
                            current_line_y = y_pos
                            line_height = height
                        elif abs(y_pos - current_line_y) > line_height * 0.5:
                            # New line detected
                            if current_line:
                                extracted_text.append(' '.join(current_line))
                            current_line = [text]
                            current_line_y = y_pos
                            line_height = height
                        else:
                            # Same line
                            current_line.append(text)
                        
                        confidence_scores.append(conf)
                        bounding_boxes.append({
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        })
            
            # Add the last line
            if current_line:
                extracted_text.append(' '.join(current_line))
            
            # Join lines with proper paragraph breaks
            full_text = self._join_paragraphs(extracted_text)
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            return {
                'text': full_text,
                'lines': extracted_text,
                'confidence': avg_confidence,
                'confidence_scores': confidence_scores,
                'bounding_boxes': bounding_boxes,
                'language': language,
                'config': config,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'lines': [],
                'confidence': 0,
                'confidence_scores': [],
                'bounding_boxes': [],
                'language': language,
                'config': config,
                'success': False,
                'error': str(e)
            }
    
    def _join_paragraphs(self, lines):
        """
        Join lines into paragraphs with proper spacing
        
        Args:
            lines: List of text lines
            
        Returns:
            Formatted text with paragraph breaks
        """
        if not lines:
            return ""
        
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double line breaks
        return '\n\n'.join(paragraphs)
    
    def extract_long_text(self, image, language='eng', confidence_threshold=60):
        """
        Extract text optimized for long paragraphs and documents
        
        Args:
            image: Preprocessed image
            language: Language code for OCR
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Use document-optimized configuration
            result = self.extract_text(
                image, 
                language=language, 
                config='document', 
                confidence_threshold=confidence_threshold
            )
            
            if result['success']:
                # Post-process for better paragraph detection
                result['text'] = self._improve_paragraph_formatting(result['text'])
            
            return result
            
        except Exception as e:
            return {
                'text': '',
                'lines': [],
                'confidence': 0,
                'confidence_scores': [],
                'bounding_boxes': [],
                'language': language,
                'config': 'document',
                'success': False,
                'error': str(e)
            }
    
    def _improve_paragraph_formatting(self, text):
        """
        Improve paragraph formatting for long text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Formatted text with improved paragraph structure
        """
        if not text:
            return text
        
        # Split into lines
        lines = text.split('\n')
        
        # Clean and normalize lines
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive whitespace
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        # Group lines into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned_lines:
            # Check if this line might be a paragraph break
            # (short line, ends with period, or has different characteristics)
            is_paragraph_break = (
                len(line) < 50 or  # Short line
                line.endswith('.') or  # Ends with period
                line.endswith('!') or  # Ends with exclamation
                line.endswith('?') or  # Ends with question mark
                line.isupper() or  # All caps (might be heading)
                (len(line.split()) <= 3 and line.endswith('.'))  # Very short sentence
            )
            
            if is_paragraph_break and current_paragraph:
                # End current paragraph
                current_paragraph.append(line)
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Add remaining lines as last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with proper spacing
        result = '\n\n'.join(paragraphs)
        
        # Final cleanup
        result = result.replace('\n\n\n', '\n\n')  # Remove excessive line breaks
        result = result.strip()
        
        return result
    
    def extract_text_optimized_for_paragraphs(self, image, language='eng', confidence_threshold=60):
        """
        Extract text with special optimization for long paragraphs and documents
        
        Args:
            image: Preprocessed image
            language: Language code for OCR
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Use document configuration for better paragraph handling
            result = self.extract_text(
                image, 
                language=language, 
                config='document', 
                confidence_threshold=confidence_threshold
            )
            
            if result['success']:
                # Apply advanced paragraph formatting
                result['text'] = self._improve_paragraph_formatting(result['text'])
                
                # Add paragraph count to metadata
                paragraphs = [p for p in result['text'].split('\n\n') if p.strip()]
                result['paragraph_count'] = len(paragraphs)
                result['avg_paragraph_length'] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
            
            return result
            
        except Exception as e:
            return {
                'text': '',
                'lines': [],
                'confidence': 0,
                'confidence_scores': [],
                'bounding_boxes': [],
                'language': language,
                'config': 'document',
                'success': False,
                'error': str(e)
            }
    
    def extract_text_with_boxes(self, image, language='eng', config='default'):
        """
        Extract text with bounding box information for visualization
        
        Args:
            image: Input image
            language: Language code
            config: OCR configuration
            
        Returns:
            List of dictionaries with text and bounding box info
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            ocr_config = self.ocr_configs.get(config, self.ocr_configs['default'])
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                lang=language, 
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i, conf in enumerate(data['conf']):
                if conf > 0:  # Include all detected text
                    text = data['text'][i].strip()
                    if text:
                        results.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        })
            
            return results
            
        except Exception as e:
            return []
    
    def get_available_languages(self):
        """Get list of available languages"""
        try:
            return pytesseract.get_languages()
        except:
            return ['eng']  # Default to English if Tesseract not available
    
    def detect_language(self, image):
        """
        Attempt to detect the language of text in the image
        
        Args:
            image: Input image
            
        Returns:
            Detected language code or None
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Try common languages
            test_languages = ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'kor']
            
            best_language = None
            best_confidence = 0
            
            for lang in test_languages:
                try:
                    result = self.extract_text(image, language=lang, confidence_threshold=0)
                    if result['confidence'] > best_confidence:
                        best_confidence = result['confidence']
                        best_language = lang
                except:
                    continue
            
            return best_language if best_confidence > 30 else 'eng'
            
        except Exception as e:
            return 'eng'  # Default to English
    
    def validate_language(self, language_code):
        """Check if a language code is supported"""
        return language_code in self.supported_languages
    
    def get_language_name(self, language_code):
        """Get human-readable language name"""
        return self.supported_languages.get(language_code, language_code)
    
    def get_ocr_configs(self):
        """Get available OCR configurations"""
        return self.ocr_configs 