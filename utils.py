import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

def validate_image_file(uploaded_file) -> bool:
    """
    Validate uploaded image file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        True if valid image file, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension not in allowed_extensions:
        return False
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False
    
    return True

def load_image(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        # Read image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * max_size / height)
    else:
        new_width = max_size
        new_height = int(height * max_size / width)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized

def display_image_with_boxes(image: np.ndarray, ocr_results: List[dict], title: str = "OCR Results") -> None:
    """
    Display image with bounding boxes around detected text
    
    Args:
        image: Input image
        ocr_results: List of OCR results with bounding box information
        title: Plot title
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        
        # Draw bounding boxes
        for result in ocr_results:
            bbox = result.get('bbox', {})
            if bbox:
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Create rectangle
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add text label
                ax.text(x, y-5, result.get('text', '')[:20], fontsize=8, color='red', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ax.set_title(title)
        ax.axis('off')
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error displaying image with boxes: {str(e)}")

def create_comparison_view(original_image: np.ndarray, processed_image: np.ndarray) -> None:
    """
    Create side-by-side comparison of original and processed images
    
    Args:
        original_image: Original input image
        processed_image: Processed image
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display original image
        if len(original_image.shape) == 3:
            ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display processed image
        if len(processed_image.shape) == 3:
            ax2.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(processed_image, cmap='gray')
        ax2.set_title('Processed Image')
        ax2.axis('off')
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error creating comparison view: {str(e)}")

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about the image
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'size_pixels': width * height,
        'aspect_ratio': round(width / height, 2),
        'dtype': str(image.dtype)
    }

def create_download_button(data: bytes, filename: str, button_text: str) -> None:
    """
    Create a download button in Streamlit
    
    Args:
        data: File data as bytes
        filename: Name of the file to download
        button_text: Text to display on the button
    """
    st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime="application/octet-stream"
    )

def show_processing_progress(progress_bar, current_step: int, total_steps: int, step_name: str) -> None:
    """
    Update processing progress bar
    
    Args:
        progress_bar: Streamlit progress bar
        current_step: Current step number
        total_steps: Total number of steps
        step_name: Name of current step
    """
    progress = current_step / total_steps
    progress_bar.progress(progress)
    st.write(f"Step {current_step}/{total_steps}: {step_name}")

def create_metadata_display(metadata: dict) -> None:
    """
    Display metadata in a formatted way
    
    Args:
        metadata: Dictionary containing metadata
    """
    if not metadata:
        return
    
    st.subheader("📊 Metadata")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        for key, value in list(metadata.items())[:len(metadata)//2]:
            st.metric(key.replace('_', ' ').title(), value)
    
    with col2:
        for key, value in list(metadata.items())[len(metadata)//2:]:
            st.metric(key.replace('_', ' ').title(), value)

def create_error_message(error: str, error_type: str = "Error") -> None:
    """
    Create a formatted error message
    
    Args:
        error: Error message
        error_type: Type of error
    """
    st.error(f"🚨 **{error_type}**: {error}")

def create_success_message(message: str) -> None:
    """
    Create a formatted success message
    
    Args:
        message: Success message
    """
    st.success(f"✅ {message}")

def create_info_message(message: str) -> None:
    """
    Create a formatted info message
    
    Args:
        message: Info message
    """
    st.info(f"ℹ️ {message}")

def create_warning_message(message: str) -> None:
    """
    Create a formatted warning message
    
    Args:
        message: Warning message
    """
    st.warning(f"⚠️ {message}")

def format_confidence_score(confidence: float) -> str:
    """
    Format confidence score with color coding
    
    Args:
        confidence: Confidence score (0-100)
        
    Returns:
        Formatted confidence string
    """
    if confidence >= 90:
        return f"🟢 {confidence:.1f}% (Excellent)"
    elif confidence >= 75:
        return f"🟡 {confidence:.1f}% (Good)"
    elif confidence >= 60:
        return f"🟠 {confidence:.1f}% (Fair)"
    else:
        return f"🔴 {confidence:.1f}% (Poor)"

def create_timestamp() -> str:
    """
    Create a formatted timestamp
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "extracted_text"
    
    return filename 