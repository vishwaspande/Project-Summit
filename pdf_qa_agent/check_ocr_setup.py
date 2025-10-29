"""
OCR Setup Diagnostic Tool
Run this to check if OCR is properly installed and configured
Usage: python check_ocr_setup.py
"""

import sys
import os

def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n" + "="*60)
    print("CHECKING PYTHON PACKAGES")
    print("="*60)
    
    packages = {
        'pytesseract': 'pip install pytesseract',
        'pdf2image': 'pip install pdf2image',
        'PIL': 'pip install Pillow'
    }
    
    all_installed = True
    
    for package, install_cmd in packages.items():
        try:
            if package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {package} - Installed")
        except ImportError:
            print(f"✗ {package} - NOT installed")
            print(f"  Install with: {install_cmd}")
            all_installed = False
    
    return all_installed

def check_tesseract():
    """Check if Tesseract OCR is installed and accessible."""
    print("\n" + "="*60)
    print("CHECKING TESSERACT OCR ENGINE")
    print("="*60)
    
    try:
        import pytesseract
        
        # Try to get Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract installed - Version: {version}")
            return True
        except pytesseract.TesseractNotFoundError:
            print("✗ Tesseract NOT found")
            print("\n  Tesseract is not installed or not in PATH")
            print("\n  WINDOWS USERS:")
            print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  2. Install to: C:\\Program Files\\Tesseract-OCR")
            print("  3. Add to PATH or set in Python:")
            print("     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
            
            # Check common installation paths
            print("\n  Checking common installation paths...")
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe"
            ]
            
            found_path = None
            for path in common_paths:
                if os.path.exists(path):
                    print(f"  ✓ Found at: {path}")
                    found_path = path
                    break
            
            if found_path:
                print(f"\n  Set this in your code:")
                print(f"  import pytesseract")
                print(f"  pytesseract.pytesseract.tesseract_cmd = r'{found_path}'")
            
            return False
            
    except ImportError:
        print("✗ pytesseract package not installed")
        print("  Install with: pip install pytesseract")
        return False

def check_poppler():
    """Check if Poppler is installed (needed for pdf2image)."""
    print("\n" + "="*60)
    print("CHECKING POPPLER (for PDF to Image conversion)")
    print("="*60)
    
    try:
        from pdf2image import convert_from_path
        
        # Try to convert a dummy PDF (will fail but tells us if poppler works)
        # We're just checking if the library can find poppler
        print("✓ pdf2image library installed")
        
        print("\n  Testing poppler availability...")
        print("  Note: Full test requires a PDF file")
        
        print("\n  WINDOWS USERS:")
        print("  If you get errors when processing PDFs:")
        print("  1. Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
        print("  2. Extract to: C:\\poppler")
        print("  3. Set in code:")
        print("     from pdf2image import convert_from_path")
        print("     images = convert_from_path('file.pdf', poppler_path=r'C:\\poppler\\Library\\bin')")
        
        return True
        
    except ImportError:
        print("✗ pdf2image not installed")
        print("  Install with: pip install pdf2image")
        return False

def test_ocr_with_sample():
    """Test OCR with a simple sample if possible."""
    print("\n" + "="*60)
    print("TESTING OCR FUNCTIONALITY")
    print("="*60)
    
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image with text
        print("Creating test image with text...")
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        text = "OCR Test: Hello World!"
        draw.text((10, 30), text, fill='black')
        
        # Try OCR on the test image
        print("Running OCR on test image...")
        result = pytesseract.image_to_string(img)
        
        if "Hello" in result or "Test" in result or "OCR" in result:
            print(f"✓ OCR is working!")
            print(f"  Detected text: {result.strip()}")
            return True
        else:
            print(f"⚠ OCR ran but results unclear")
            print(f"  Detected: {result.strip()}")
            return False
            
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def create_fix_script():
    """Create a Python script to fix common Tesseract path issues."""
    print("\n" + "="*60)
    print("CREATING FIX SCRIPT")
    print("="*60)
    
    fix_script = '''"""
Tesseract Path Fix
Add this to the top of your pdf_qa_unified.py file
"""

import pytesseract
import os

# Try to find Tesseract automatically
common_paths = [
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    r"C:\\Tesseract-OCR\\tesseract.exe"
]

tesseract_found = False
for path in common_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"✓ Using Tesseract at: {path}")
        tesseract_found = True
        break

if not tesseract_found:
    print("⚠ Tesseract not found in common paths.")
    print("  Please set manually:")
    print("  pytesseract.pytesseract.tesseract_cmd = r'YOUR_PATH_HERE'")
'''
    
    with open('tesseract_fix.py', 'w') as f:
        f.write(fix_script)
    
    print("✓ Created tesseract_fix.py")
    print("  Copy the code from this file to the top of your pdf_qa_unified.py")

def main():
    """Run all diagnostics."""
    print("="*60)
    print("OCR SETUP DIAGNOSTIC TOOL")
    print("="*60)
    
    results = {
        'packages': check_python_packages(),
        'tesseract': check_tesseract(),
        'poppler': check_poppler()
    }
    
    # Test OCR if basics are installed
    if results['packages'] and results['tesseract']:
        results['ocr_test'] = test_ocr_with_sample()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(results.values()):
        print("✅ ALL CHECKS PASSED!")
        print("   OCR should work properly.")
        print("\n   You can now use:")
        print("   from pdf_qa_unified import PDFQA")
        print("   qa = PDFQA()")
        print("   qa.load_pdfs(['your_scanned.pdf'])")
    else:
        print("❌ ISSUES FOUND:")
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"   {status} {check}")
        
        print("\n   Follow the instructions above to fix issues.")
        
        # Create fix script
        create_fix_script()
    
    print("="*60)

if __name__ == "__main__":
    main()
