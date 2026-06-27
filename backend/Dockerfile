# Use a slim Python base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if required by standard libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first to save space and memory
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Create the data directory (for SQLite and FAISS)
RUN mkdir -p /app/data

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Start FastAPI using uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
