# Bug Fixes

## Bug 1: App crashes on large PDFs
- Issue: No file size validation — large PDFs caused memory errors
- Fix: Added 50MB file size check before processing

## Bug 2: App crashes on empty/scanned PDFs
- Issue: PyMuPDF returns empty string for image-only PDFs
- Fix: Added empty chunks check with user-friendly error message

## Bug 3: ChromaDB duplicate ID error on re-upload
- Issue: Re-uploading same PDF caused duplicate ID errors in ChromaDB
- Fix: Create fresh ChromaDB client for each new PDF upload

## Bug 4: Short/vague questions cause poor retrieval
- Issue: 1-2 word questions return irrelevant chunks
- Fix: Added minimum question length check (5 characters)