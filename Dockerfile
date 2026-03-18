FROM python:3.11-slim

WORKDIR /app

# Install system dependencies:
# - build-essential: compilation tools
# - tesseract-ocr + language packs: OCR for scanned documents/images
# - poppler-utils: PDF page-to-image conversion (pdf2image)
# - ffmpeg: audio/video processing (Whisper transcription)
# - libmagic1: file type detection (unstructured)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-eng \
        poppler-utils \
        ffmpeg \
        libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/chroma_db uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
