import streamlit as st
import numpy as np
import utils

# These functions are moved from app.py for UI modularity

def display_export_options(extracted_text, ocr_results, components):
    """Display export options"""
    st.subheader("📤 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Export as Text File**")
        if st.button("📄 Download TXT"):
            try:
                txt_data = components['text_processor'].export_to_txt(extracted_text)
                timestamp = utils.create_timestamp().replace(":", "-")
                filename = f"extracted_text_{timestamp}.txt"
                utils.create_download_button(txt_data, filename, "📄 Download TXT File")
            except Exception as e:
                utils.create_error_message(f"Error creating TXT file: {str(e)}")
    with col2:
        st.write("**Export as Word Document**")
        if st.button("📝 Download DOCX"):
            try:
                metadata = {
                    'Language': ocr_results['language'],
                    'Confidence': f"{ocr_results['confidence']:.1f}%",
                    'Config': ocr_results['config'],
                    'Timestamp': utils.create_timestamp()
                }
                docx_data = components['text_processor'].export_to_docx(extracted_text, metadata=metadata)
                timestamp = utils.create_timestamp().replace(":", "-")
                filename = f"extracted_text_{timestamp}.docx"
                utils.create_download_button(docx_data, filename, "📝 Download DOCX File")
            except Exception as e:
                utils.create_error_message(f"Error creating DOCX file: {str(e)}")

def display_translation_options(extracted_text, components):
    """Display translation options"""
    st.subheader("🌍 Translation")
    if not extracted_text.strip():
        utils.create_warning_message("No text to translate")
        return
    translation_languages = components['text_processor'].get_translation_languages()
    target_language = st.selectbox(
        "Translate to:",
        options=list(translation_languages.keys()),
        format_func=lambda x: translation_languages[x],
        index=0
    )
    if st.button("🌍 Translate"):
        with st.spinner("Translating..."):
            translation_result = components['text_processor'].translate_text(
                extracted_text, target_language
            )
            if translation_result['success']:
                st.success("Translation completed!")
                st.text_area(
                    "Translated Text",
                    translation_result['translated_text'],
                    height=150
                )
                if st.button("📄 Download Translated Text"):
                    try:
                        txt_data = components['text_processor'].export_to_txt(
                            translation_result['translated_text']
                        )
                        timestamp = utils.create_timestamp().replace(":", "-")
                        filename = f"translated_text_{target_language}_{timestamp}.txt"
                        utils.create_download_button(txt_data, filename, "📄 Download Translation")
                    except Exception as e:
                        utils.create_error_message(f"Error downloading translation: {str(e)}")
            else:
                utils.create_error_message(translation_result['error'])

def display_tts_options(extracted_text, components):
    """Display text-to-speech options"""
    st.subheader("🔊 Text-to-Speech")
    if not extracted_text.strip():
        utils.create_warning_message("No text to convert to speech")
        return
    tts_language = st.selectbox(
        "Language for TTS:",
        options=['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh-cn'],
        format_func=lambda x: components['text_processor'].get_language_name(x),
        index=0
    )
    if st.button("🔊 Generate Speech"):
        with st.spinner("Generating speech..."):
            audio_data = components['text_processor'].text_to_speech(
                extracted_text, tts_language
            )
            if audio_data:
                st.success("Speech generated successfully!")
                timestamp = utils.create_timestamp().replace(":", "-")
                filename = f"speech_{tts_language}_{timestamp}.mp3"
                utils.create_download_button(audio_data, filename, "🔊 Download Audio")
            else:
                utils.create_error_message("Failed to generate speech")

def display_structured_data(extracted_text, components):
    """Display structured data extraction"""
    st.subheader("📊 Structured Data")
    if not extracted_text.strip():
        utils.create_warning_message("No text to analyze")
        return
    structured_data = components['text_processor'].extract_structured_data(extracted_text)
    col1, col2 = st.columns(2)
    with col1:
        if structured_data['emails']:
            st.write("**📧 Email Addresses:**")
            for email in structured_data['emails']:
                st.write(f"• {email}")
        if structured_data['phone_numbers']:
            st.write("**📞 Phone Numbers:**")
            for phone in structured_data['phone_numbers']:
                st.write(f"• {phone}")
    with col2:
        if structured_data['urls']:
            st.write("**🌐 URLs:**")
            for url in structured_data['urls']:
                st.write(f"• {url}")
        if structured_data['dates']:
            st.write("**📅 Dates:**")
            for date in structured_data['dates']:
                st.write(f"• {date}")
    if structured_data['numbers']:
        st.write("**🔢 Numbers:**")
        numbers_text = ", ".join(structured_data['numbers'][:10])
        if len(structured_data['numbers']) > 10:
            numbers_text += f" ... and {len(structured_data['numbers']) - 10} more"
        st.write(numbers_text)

def display_text_analysis(extracted_text, ocr_results, components):
    """Display text analysis"""
    st.subheader("📋 Text Analysis")
    if not extracted_text.strip():
        utils.create_warning_message("No text to analyze")
        return
    cleaned_text = components['text_processor'].clean_text(extracted_text)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Text:**")
        st.text_area("Original", extracted_text, height=100, disabled=True)
    with col2:
        st.write("**Cleaned Text:**")
        st.text_area("Cleaned", cleaned_text, height=100, disabled=True)
    if ocr_results.get('confidence_scores'):
        st.write("**Confidence Analysis:**")
        confidence_scores = ocr_results['confidence_scores']
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        with col2:
            st.metric("Minimum Confidence", f"{min_confidence:.1f}%")
        with col3:
            st.metric("Maximum Confidence", f"{max_confidence:.1f}%") 