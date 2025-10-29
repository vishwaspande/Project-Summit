# Unified PDF Q&A Agent

## ✨ ONE System for ALL PDFs

This is the **simplest solution** - automatically handles both regular and scanned PDFs!

### 🎯 What You Get

- ✅ **Auto-detection**: System automatically detects if PDF is scanned
- ✅ **Smart processing**: Uses regular extraction for text PDFs, OCR for scanned PDFs
- ✅ **One interface**: Same API for all PDF types
- ✅ **Progress tracking**: Shows which files used OCR
- ✅ **Mix & match**: Load regular and scanned PDFs together

## 🚀 Quick Start

### 1. Install Dependencies

**Basic (for regular PDFs only):**
```powershell
pip install streamlit anthropic pypdf2 sentence-transformers faiss-cpu python-dotenv
```

**With OCR (for scanned PDFs):**
```powershell
pip install streamlit anthropic pypdf2 sentence-transformers faiss-cpu python-dotenv pytesseract pdf2image
```

Then install Tesseract: https://github.com/tesseract-ocr/tesseract

### 2. Set API Key

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY="your_api_key_here"

# Or create .env file
echo ANTHROPIC_API_KEY=your_api_key_here > .env
```

### 3. Run the App

```powershell
streamlit run app.py
```

That's it! 🎉

## 📖 Usage

### In Streamlit:
1. Upload any PDFs (regular or scanned)
2. Click "Load PDFs"
3. System automatically detects type and processes accordingly
4. Ask questions!

### In Python:
```python
from pdf_qa_unified import PDFQA

# Initialize
qa = PDFQA()

# Load any mix of PDFs
qa.load_pdfs([
    "regular_document.pdf",      # ← Regular PDF (fast)
    "scanned_resume.pdf",         # ← Scanned (uses OCR automatically)
    "another_document.pdf"
])

# Ask questions
result = qa.ask("What are the key skills mentioned?")
print(result['answer'])

# Check which files used OCR
stats = qa.get_stats()
print(stats['file_types'])
# Output: {'regular_document.pdf': 'regular', 'scanned_resume.pdf': 'scanned (OCR)'}
```

## 🔍 How It Works

```
Upload PDF
    ↓
Try regular text extraction
    ↓
Got text? → YES → Use it (fast!)
    ↓
    NO
    ↓
Is OCR available? → YES → Use OCR (slower but works!)
    ↓              → NO → Show warning
    ↓
Create chunks & embeddings
    ↓
Ready for Q&A!
```

## 📁 Files You Need

**Essential:**
1. **pdf_qa_unified.py** - The unified module (ONE module for all PDFs)
2. **app.py** - Streamlit interface

**Optional:**
3. **.env** - For API key storage

## ⚡ Performance

| PDF Type | Processing Speed | Notes |
|----------|------------------|-------|
| Regular PDF (10 pages) | 1-3 seconds | Instant text extraction |
| Scanned PDF (10 pages) | 30-90 seconds | OCR processing time |
| Mixed (5 regular + 5 scanned) | 15-50 seconds | Combines both |

## 🎯 Benefits vs Previous Versions

| Feature | Old (2 modules) | New (Unified) |
|---------|----------------|---------------|
| **Simplicity** | Choose module manually | Automatic detection |
| **Files needed** | 2 modules + integration | 1 module + app |
| **User experience** | Toggle OCR manually | Fully automatic |
| **Error handling** | Manual fallback | Auto-fallback |
| **Mixed PDFs** | Manual sorting needed | Works seamlessly |

## 🛠️ Troubleshooting

**"OCR Not Available" warning:**
→ You can still use regular PDFs! To add OCR support:
```powershell
pip install pytesseract pdf2image
```

**"No text extracted":**
→ Check if:
1. PDF is valid/not corrupted
2. PDF is scanned → Install OCR (see above)
3. PDF is password-protected → Remove password first

**Slow processing:**
→ Normal for scanned PDFs! OCR takes 3-9 seconds per page.

## 💡 Pro Tips

1. **Test first**: Try with 1-2 PDFs before loading many
2. **Mix is fine**: Load regular and scanned PDFs together
3. **Monitor console**: Shows which PDFs use OCR in real-time
4. **OCR optional**: System works fine without OCR for regular PDFs

## 🎓 Example Use Cases

✅ **Resume screening**: Mix of Word-converted and scanned resumes  
✅ **Research papers**: Some PDFs are text, some are scanned  
✅ **Legal documents**: Original PDFs + scanned signatures  
✅ **Mixed archive**: Old scanned docs + new digital docs  

## 🔧 Customization

```python
# Change chunk size
qa = PDFQA(chunk_size=800)

# Disable auto-OCR (manual control)
qa = PDFQA(auto_ocr=False)

# Retrieve more context
result = qa.ask("question", top_k=5)
```

## 📊 What's Next?

Once working:
1. **Scale up**: Test with 10-20 documents
2. **Fine-tune**: Adjust chunk_size based on your docs
3. **Deploy**: Push to Streamlit Cloud (free)
4. **Enhance**: Add conversation memory, filters, etc.

---

**This is your best option** - one system, handles everything! 🚀
