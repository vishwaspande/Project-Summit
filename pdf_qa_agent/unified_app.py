"""
Unified PDF Q&A Streamlit App
Automatically handles both regular PDFs and scanned PDFs

Simply upload any PDF - the system automatically detects if OCR is needed!
"""

import streamlit as st
import tempfile
import os
from pdf_qa_unified import PDFQA

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize PDF Q&A in session state
if 'pdf_qa' not in st.session_state:
    st.session_state.pdf_qa = PDFQA()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # Check OCR availability
    stats = st.session_state.pdf_qa.get_stats()
    if stats['ocr_available']:
        st.success("✓ OCR Available")
        st.caption("System will automatically handle scanned PDFs")
    else:
        st.warning("⚠ OCR Not Installed")
        st.caption("Only text-based PDFs will work")
        with st.expander("Install OCR"):
            st.code("pip install pytesseract pdf2image")
            st.markdown("[Setup Guide](https://github.com/tesseract-ocr/tesseract)")
    
    st.divider()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload any PDF - regular or scanned. System auto-detects type."
    )
    
    # Load button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load PDFs", type="primary", use_container_width=True):
            if uploaded_files:
                temp_paths = []
                
                # Create progress container
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save files
                    status_text.text("Saving files...")
                    for i, uploaded_file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            temp_paths.append(tmp_file.name)
                        progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                    
                    # Process files
                    status_text.text("Processing PDFs (OCR may take 30-60s per scanned page)...")
                    load_stats = st.session_state.pdf_qa.load_pdfs(temp_paths)
                    progress_bar.progress(1.0)
                    
                    # Clean up
                    for path in temp_paths:
                        try:
                            os.unlink(path)
                        except:
                            pass
                    
                    # Show results
                    progress_bar.empty()
                    status_text.empty()
                
                if load_stats['new_chunks'] > 0:
                    st.success(f"✓ Loaded {load_stats['new_chunks']} chunks from {len(uploaded_files)} file(s)")
                    
                    # Show breakdown
                    if load_stats['regular_files']:
                        st.info(f"📄 Regular PDFs: {len(load_stats['regular_files'])}")
                    if load_stats['ocr_files']:
                        st.info(f"🔍 Scanned PDFs (OCR): {len(load_stats['ocr_files'])}")
                else:
                    st.error("⚠ No text extracted. Check if PDFs are valid or OCR is installed.")
            else:
                st.warning("Please upload PDF files first")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.pdf_qa.clear()
            st.session_state.messages = []
            st.success("Cleared!")
            st.rerun()
    
    # Show loaded documents
    st.divider()
    stats = st.session_state.pdf_qa.get_stats()
    
    if stats['loaded_files']:
        st.subheader(f"📚 Loaded Documents ({len(stats['loaded_files'])})")
        
        for filename in stats['loaded_files']:
            file_type = stats['file_types'].get(filename, 'unknown')
            icon = "🔍" if "OCR" in file_type else "📄"
            st.text(f"{icon} {filename}")
            st.caption(f"   Type: {file_type}")
        
        st.metric("Total Chunks", stats['total_chunks'])
    else:
        st.info("No documents loaded yet")

# Main content area
st.title("🤖 PDF Q&A Agent")
st.caption("Ask questions about your documents - supports both regular and scanned PDFs")

# Instructions if no documents
if st.session_state.pdf_qa.get_stats()['total_chunks'] == 0:
    st.info("""
    ### 👋 Welcome! Get Started in 3 Steps:
    
    1. **Upload PDFs** in the sidebar (regular or scanned - we handle both!)
    2. **Click "Load PDFs"** to process them
    3. **Ask questions** below
    
    The system automatically detects if a PDF is scanned and uses OCR when needed.
    """)
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{source['filename']}**")
                            st.caption(source['preview'])
                        with col2:
                            st.metric("Relevance", f"{source['relevance_score']:.0%}")
                        if i < len(message["sources"]):
                            st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents loaded
    if st.session_state.pdf_qa.get_stats()['total_chunks'] == 0:
        st.error("⚠ Please upload and load documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                result = st.session_state.pdf_qa.ask(prompt, top_k=3)
                
                st.markdown(result['answer'])
                
                # Show sources
                if result.get('sources'):
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{source['filename']}**")
                                st.caption(source['preview'])
                            with col2:
                                st.metric("Relevance", f"{source['relevance_score']:.0%}")
                            if i < len(result['sources']):
                                st.divider()
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result.get('sources', [])
            })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💡 Tip: Ask specific questions for better results")
with col2:
    st.caption("📄 Supports regular PDFs")
with col3:
    st.caption("🔍 Auto-detects scanned PDFs")
