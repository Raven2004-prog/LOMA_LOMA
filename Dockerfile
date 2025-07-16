FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app
COPY examples/ ./examples

# Run placeholder command (to override at runtime)
ENTRYPOINT ["python", "app/main.py"]
