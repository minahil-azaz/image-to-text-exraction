import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Import our custom modules
from image_preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from text_processor import TextProcessor
import utils
from ui_helpers import display_export_options, display_translation_options, display_tts_options, display_structured_data, display_text_analysis

# Page configuration
st.set_page_config(
    page_title="Image-to-Text OCR System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache OCR components"""
    return {
        'preprocessor': ImagePreprocessor(),
        'ocr_engine': OCREngine(),
        'text_processor': TextProcessor()
    }

components = load_components()

def main():
    """Main application function"""
    
    # Remove the header and sub-header (navbar)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # OCR Language Selection
        st.subheader("🌍 Language")
        available_languages = components['ocr_engine'].get_available_languages()
        
        # Create language options with search functionality
        language_options = {}
        for lang in available_languages:
            lang_name = components['ocr_engine'].get_language_name(lang)
            language_options[lang_name] = lang
        
        # Add search box for languages
        search_term = st.text_input("🔍 Search languages:", placeholder="Type to search...")
        
        # Filter languages based on search
        if search_term:
            filtered_options = {name: code for name, code in language_options.items() 
                              if search_term.lower() in name.lower() or search_term.lower() in code.lower()}
        else:
            # Show most common languages first, then all others
            common_languages = ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'kor', 'ara', 'hin']
            common_options = {components['ocr_engine'].get_language_name(lang): lang 
                            for lang in common_languages if lang in available_languages}
            other_options = {name: code for name, code in language_options.items() 
                           if code not in common_languages}
            filtered_options = {**common_options, **other_options}
        
        selected_language = st.selectbox(
            "Select OCR Language",
            options=list(filtered_options.keys()),
            index=0,
            help=f"Available: {len(available_languages)} languages. Use search to find specific languages."
        )
        ocr_language = filtered_options[selected_language]
        
        # Confidence Threshold
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=60,
            help="Minimum confidence score for text detection"
        )
        
        # Preprocessing Options
        st.subheader("🖼️ Image Preprocessing")
        preprocessing_options = components['preprocessor'].get_preprocessing_options()
        
        enable_grayscale = st.checkbox("Convert to Grayscale", value=True)
        enable_denoise = st.checkbox("Remove Noise", value=True)
        enable_threshold = st.checkbox("Apply Threshold", value=True)
        enable_deskew = st.checkbox("Auto-rotate Text", value=False)
        enable_resize = st.checkbox("Upscale Image", value=False)
        
        # Advanced Options
        with st.expander("Advanced Options"):
            enable_contrast = st.checkbox("Enhance Contrast", value=False)
            enable_noise_removal = st.checkbox("Remove Background Noise", value=False)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
            help="Upload an image containing text to extract"
        )
        
        if uploaded_file is not None:
            # Validate file
            if not utils.validate_image_file(uploaded_file):
                utils.create_error_message("Invalid file format or size. Please upload a valid image file (max 10MB).")
                return
            
            # Load image
            original_image = utils.load_image(uploaded_file)
            if original_image is None:
                return
            
            st.session_state.original_image = original_image
            
            # Display original image
            st.subheader("📷 Original Image")
            st.image(original_image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Image information
            image_info = utils.get_image_info(original_image)
            utils.create_metadata_display(image_info)
            
            # Process button
            if st.button("🚀 Extract Text", type="primary"):
                process_image(original_image, ocr_language, confidence_threshold,
                            enable_grayscale, enable_denoise, enable_threshold, 
                            enable_deskew, enable_resize, enable_contrast, enable_noise_removal)
    
    # Remove the col2 Quick Info section entirely

def process_image(original_image, ocr_language, confidence_threshold,
                 enable_grayscale, enable_denoise, enable_threshold, 
                 enable_deskew, enable_resize, enable_contrast, enable_noise_removal):
    """Process image and extract text"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Preprocess image
        utils.show_processing_progress(progress_bar, 1, 5, "Preprocessing image...")
        
        preprocessing_options = {
            'grayscale': enable_grayscale,
            'denoise': enable_denoise,
            'threshold': enable_threshold,
            'deskew': enable_deskew,
            'resize': enable_resize
        }
        
        processed_image = components['preprocessor'].preprocess_image(
            original_image, preprocessing_options
        )
        
        # Apply additional preprocessing if enabled
        if enable_contrast:
            processed_image = components['preprocessor'].enhance_contrast(processed_image)
        
        if enable_noise_removal:
            processed_image = components['preprocessor'].remove_background_noise(processed_image)
        
        st.session_state.processed_image = processed_image
        
        # Step 2: Display processed image
        utils.show_processing_progress(progress_bar, 2, 5, "Displaying processed image...")
        
        st.subheader(" Processed Image")
        utils.create_comparison_view(original_image, processed_image)
        
        # Step 3: Extract text
        utils.show_processing_progress(progress_bar, 3, 5, "Extracting text...")
        
        # Use appropriate extraction method based on config
        ocr_results = components['ocr_engine'].extract_text(
            processed_image, 
            language=ocr_language,
            confidence_threshold=confidence_threshold
        )
        
        st.session_state.ocr_results = ocr_results
        
        # Step 4: Display results
        utils.show_processing_progress(progress_bar, 4, 5, "Displaying results...")
        
        display_ocr_results(ocr_results, processed_image)
        
        # Step 5: Complete
        utils.show_processing_progress(progress_bar, 5, 5, "Processing complete!")
        
        utils.create_success_message("Text extraction completed successfully!")
        
    except Exception as e:
        utils.create_error_message(f"Error during processing: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_ocr_results(ocr_results, processed_image):
    """Display OCR results and additional features"""
    
    if not ocr_results['success']:
        utils.create_error_message(ocr_results.get('error', 'OCR processing failed'))
        return
    
    # Results section
    st.header("📝 Extracted Text")
    
    # Display confidence score
    confidence = ocr_results['confidence']
    st.markdown(f"**Confidence Score:** {utils.format_confidence_score(confidence)}")
    
    # Display extracted text
    extracted_text = ocr_results['text']
    if extracted_text:
        st.text_area("Extracted Text", extracted_text, height=200)
        
        # Text statistics
        text_stats = components['text_processor'].get_word_count(extracted_text)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", text_stats['characters'])
        with col2:
            st.metric("Words", text_stats['words'])
        with col3:
            st.metric("Sentences", text_stats['sentences'])
        with col4:
            st.metric("Paragraphs", text_stats['paragraphs'])
        
        # Additional features
        st.header("🔧 Additional Features")
        
        # Create tabs for different features
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Export", "🌍 Translate", "🔊 Text-to-Speech", "📊 Structured Data", "📋 Text Analysis"])
        
        with tab1:
            display_export_options(extracted_text, ocr_results, components)
        
        with tab2:
            display_translation_options(extracted_text, components)
        
        with tab3:
            display_tts_options(extracted_text, components)
        
        with tab4:
            display_structured_data(extracted_text, components)
        
        with tab5:
            display_text_analysis(extracted_text, ocr_results, components)
        
        # Display bounding boxes if available
        if ocr_results.get('bounding_boxes'):
            st.header("📍 Text Detection Visualization")
            ocr_boxes = components['ocr_engine'].extract_text_with_boxes(
                processed_image, 
                language=ocr_results['language']
            )
            utils.display_image_with_boxes(processed_image, ocr_boxes)
    
    else:
        utils.create_warning_message("No text was detected in the image. Try adjusting the preprocessing options or confidence threshold.")

if __name__ == "__main__":
    main() 