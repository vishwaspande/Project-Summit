"""
Unified PDF Q&A Module - Handles both regular and scanned PDFs automatically
Usage: 
    from pdf_qa_unified import PDFQA
    
    qa = PDFQA(api_key="your_key")
    qa.load_pdfs(["doc1.pdf", "scanned_doc2.pdf"])  # Automatically detects type
    answer = qa.ask("your question")
"""

import os
from typing import List, Dict
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic


# Add after: import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR dependencies (optional - will auto-detect)
try:
    from pdf2image import convert_from_path
    import pytesseract
    
    # Try to automatically find Tesseract on Windows
    if os.name == 'nt':  # Windows
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✓ Found Tesseract at: {path}")
                break
    
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class PDFQA:
    """Unified PDF Q&A system that automatically handles both regular and scanned PDFs."""
    
    def __init__(self, api_key: str = None, chunk_size: int = 500, auto_ocr: bool = True):
        """
        Initialize PDF Q&A system.
        
        Args:
            api_key: Anthropic API key (optional if ANTHROPIC_API_KEY env var is set)
            chunk_size: Characters per chunk
            auto_ocr: Automatically use OCR when text extraction fails (default: True)
        """
        self.chunk_size = chunk_size
        self.auto_ocr = auto_ocr
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.loaded_files = []
        self.file_types = {}  # Track which files used OCR
        
        # Initialize Anthropic client
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set as env variable")
        self.client = Anthropic(api_key=api_key)
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple:
        """
        Extract text from PDF, automatically using OCR if needed.
        
        Returns:
            tuple: (extracted_text, used_ocr)
        """
        filename = os.path.basename(pdf_path)
        
        # First, try regular text extraction
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 0:
                    text += f"\n[Page {page_num}]\n{page_text}"
            
            # Check if we got meaningful text
            if len(text.strip()) > 100:
                print(f"  ✓ Text extracted successfully (regular PDF)")
                return text, False
            
            # If very little text, try OCR
            print(f"  ⚠ Little/no text found - attempting OCR...")
            
            if self.auto_ocr and OCR_AVAILABLE:
                return self._extract_with_ocr(pdf_path), True
            elif self.auto_ocr and not OCR_AVAILABLE:
                print(f"  ⚠ OCR not available. Install: pip install pytesseract pdf2image")
                print(f"     Also install Tesseract: https://github.com/tesseract-ocr/tesseract")
                return text, False
            else:
                return text, False
                
        except Exception as e:
            print(f"  Error with regular extraction: {e}")
            if self.auto_ocr and OCR_AVAILABLE:
                print(f"  Attempting OCR as fallback...")
                return self._extract_with_ocr(pdf_path), True
            else:
                raise e
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (for scanned PDFs)."""
        if not OCR_AVAILABLE:
            raise ValueError("OCR libraries not installed. Install: pip install pytesseract pdf2image")
        
        try:
            # Convert PDF to images
            print(f"    Converting PDF to images...")
            
            # Add poppler path for Windows
            poppler_path = None
            if os.name == 'nt':  # Windows
                if os.path.exists(r'C:\poppler\Library\bin'):
                    poppler_path = r'C:\poppler\Library\bin'
            
            if poppler_path:
                images = convert_from_path(pdf_path, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path)
            
            text = ""
            total_pages = len(images)
            
            for page_num, image in enumerate(images, 1):
                print(f"    OCR processing page {page_num}/{total_pages}...", end='\r')
                page_text = pytesseract.image_to_string(image)
                text += f"\n[Page {page_num}]\n{page_text}"
            
            print(f"  ✓ OCR completed ({total_pages} pages)           ")
            return text
            
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")
        
    def chunk_text(self, text: str, filename: str) -> List[Dict]:
        """Split text into chunks with metadata."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - 50):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'filename': filename
                })
        return chunks
    
    def load_pdfs(self, pdf_paths: List[str]) -> Dict:
        """
        Load and process PDF files (automatically detects regular vs scanned).
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Dictionary with loading statistics
        """
        new_chunks = []
        ocr_files = []
        regular_files = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠ Warning: {pdf_path} not found, skipping...")
                continue
            
            try:
                filename = os.path.basename(pdf_path)
                print(f"\nProcessing: {filename}")
                
                # Extract text (auto-detects if OCR needed)
                text, used_ocr = self.extract_text_from_pdf(pdf_path)
                
                # Track file type
                self.file_types[filename] = 'scanned (OCR)' if used_ocr else 'regular'
                if used_ocr:
                    ocr_files.append(filename)
                else:
                    regular_files.append(filename)
                
                # Validate extraction
                if len(text.strip()) < 50:
                    print(f"  ⚠ Very little text extracted from {filename}")
                    if not OCR_AVAILABLE:
                        print(f"     Install OCR for scanned PDFs: pip install pytesseract pdf2image")
                    continue
                
                # Create chunks
                chunks = self.chunk_text(text, filename)
                new_chunks.extend(chunks)
                
                self.loaded_files.append(filename)
                print(f"  ✓ Loaded: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"✗ Error loading {pdf_path}: {str(e)}")
        
        if new_chunks:
            # Create embeddings
            print(f"\n📊 Creating embeddings for {len(new_chunks)} chunks...")
            texts = [chunk['text'] for chunk in new_chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Add to index
            self.chunks.extend(new_chunks)
            self.index.add(embeddings.astype('float32'))
            
            print(f"✅ Successfully loaded {len(self.loaded_files)} file(s)")
            if regular_files:
                print(f"   • Regular PDFs: {', '.join(regular_files)}")
            if ocr_files:
                print(f"   • Scanned PDFs (OCR): {', '.join(ocr_files)}")
            print(f"   • Total chunks: {len(self.chunks)}")
        
        return {
            'total_chunks': len(self.chunks),
            'loaded_files': self.loaded_files,
            'new_chunks': len(new_chunks),
            'ocr_files': ocr_files,
            'regular_files': regular_files
        }
    
    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks for a question."""
        if len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([question])[0]
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'text': self.chunks[idx]['text'],
                'filename': self.chunks[idx]['filename'],
                'score': float(1 / (1 + distance))
            })
        
        return results
    
    def ask(self, question: str, top_k: int = 3, return_sources: bool = True) -> Dict:
        """
        Ask a question and get an answer based on loaded PDFs.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            return_sources: Whether to return source information
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        if len(self.chunks) == 0:
            return {
                'answer': "No documents loaded. Please load PDFs first using load_pdfs().",
                'sources': []
            }
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information in the loaded documents.",
                'sources': []
            }
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[Document {i}: {chunk['filename']}]\n{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Answer the question based ONLY on the provided document excerpts below.

Documents:
{context}

Question: {question}

Instructions:
- Answer based solely on the information in the documents
- If the documents don't contain enough information, say so
- Mention which document(s) you're referencing
- Be concise and accurate

Answer:"""

        # Get answer from Claude
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = message.content[0].text
            
            result = {'answer': answer}
            
            if return_sources:
                result['sources'] = [
                    {
                        'filename': chunk['filename'],
                        'preview': chunk['text'][:150] + "...",
                        'relevance_score': chunk['score']
                    }
                    for chunk in relevant_chunks
                ]
            
            return result
            
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'sources': []
            }
    
    def clear(self):
        """Clear all loaded documents."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.loaded_files = []
        self.file_types = {}
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded documents."""
        return {
            'total_chunks': len(self.chunks),
            'loaded_files': self.loaded_files,
            'total_files': len(self.loaded_files),
            'file_types': self.file_types,
            'ocr_available': OCR_AVAILABLE
        }


# Example usage
if __name__ == "__main__":
    # Initialize (automatically handles both types)
    qa = PDFQA()
    
    # Load mix of regular and scanned PDFs
    qa.load_pdfs(["regular_doc.pdf", "scanned_resume.pdf", "another_doc.pdf"])
    
    # Ask questions
    result = qa.ask("What are the main topics?")
    print(result['answer'])
