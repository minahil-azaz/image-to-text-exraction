import cv2
import numpy as np
from PIL import Image
import streamlit as st

class ImagePreprocessor:
    """
    Handles image preprocessing operations to improve OCR accuracy
    """
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def preprocess_image(self, image, preprocessing_options=None):
        """
        Apply preprocessing techniques to improve OCR accuracy
        
        Args:
            image: PIL Image or numpy array
            preprocessing_options: dict with preprocessing settings
            
        Returns:
            Preprocessed image as numpy array
        """
        if preprocessing_options is None:
            preprocessing_options = {
                'grayscale': True,
                'denoise': True,
                'threshold': True,
                'deskew': False,
                'resize': False
            }
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        if preprocessing_options.get('grayscale', True):
            image = self._convert_to_grayscale(image)
        
        if preprocessing_options.get('denoise', True):
            image = self._denoise_image(image)
        
        if preprocessing_options.get('threshold', True):
            image = self._apply_threshold(image)
        
        if preprocessing_options.get('deskew', False):
            image = self._deskew_image(image)
        
        if preprocessing_options.get('resize', False):
            image = self._resize_image(image)
        
        return image
    
    def _convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def _denoise_image(self, image):
        """Apply denoising to reduce noise"""
        # Apply bilateral filter to preserve edges while reducing noise
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _apply_threshold(self, image):
        """Apply adaptive thresholding for better text extraction"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive threshold
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    
    def _deskew_image(self, image):
        """Deskew the image if it's rotated"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find the angle of rotation
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate the image
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _resize_image(self, image, scale_factor=2.0):
        """Resize image for better OCR accuracy"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def remove_background_noise(self, image):
        """Remove background noise using morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Create kernel for morphological operations
        kernel = np.ones((1, 1), np.uint8)
        
        # Apply opening operation to remove noise
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply closing operation to fill gaps
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing
    
    def get_preprocessing_options(self):
        """Return available preprocessing options for UI"""
        return {
            'grayscale': 'Convert to grayscale',
            'denoise': 'Remove noise',
            'threshold': 'Apply adaptive threshold',
            'deskew': 'Auto-rotate skewed text',
            'resize': 'Upscale image (2x)',
            'enhance_contrast': 'Enhance contrast',
            'remove_noise': 'Remove background noise'
        } 