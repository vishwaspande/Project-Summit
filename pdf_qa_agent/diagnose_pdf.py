"""
PDF Extraction Diagnostic Tool
Run this to test if your PDFs have extractable text
Usage: python diagnose_pdf.py your_document.pdf
"""

import sys
from PyPDF2 import PdfReader

def diagnose_pdf(pdf_path):
    """Diagnose PDF text extraction issues."""
    print(f"\n{'='*60}")
    print(f"Diagnosing: {pdf_path}")
    print('='*60)
    
    try:
        reader = PdfReader(pdf_path)
        print(f"✓ PDF opened successfully")
        print(f"  Total pages: {len(reader.pages)}")
        
        # Check each page
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            print(f"\n--- Page {page_num} ---")
            print(f"  Characters extracted: {len(text)}")
            print(f"  First 200 characters:")
            print(f"  {text[:200]}")
            
            if len(text.strip()) < 10:
                print(f"  ⚠️ WARNING: Very little text found on this page")
                print(f"     This might be a scanned image PDF (needs OCR)")
        
        # Overall assessment
        total_text = ""
        for page in reader.pages:
            total_text += page.extract_text()
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total characters: {len(total_text)}")
        print(f"  Total words (approx): {len(total_text.split())}")
        
        if len(total_text.strip()) < 100:
            print(f"\n❌ ISSUE: This PDF has very little extractable text")
            print(f"   Possible causes:")
            print(f"   1. Scanned/image-based PDF (needs OCR)")
            print(f"   2. Protected/encrypted PDF")
            print(f"   3. Corrupted PDF file")
            print(f"\n   Solutions:")
            print(f"   - Use OCR tool (pytesseract + pdf2image)")
            print(f"   - Try a different PDF")
        else:
            print(f"\n✅ PDF text extraction looks good!")
        
        print('='*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   The PDF might be corrupted or password-protected")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pdf.py your_document.pdf")
        sys.exit(1)
    
    diagnose_pdf(sys.argv[1])
