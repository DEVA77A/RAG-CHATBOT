# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Setup Python and package backend + frontend
FROM python:3.12-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for Hugging Face compatibility
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy requirement files and change ownership
COPY --chown=user:user backend/requirements.txt ./backend/

# Switch to the non-root user
USER user
ENV PATH=/home/user/.local/bin:$PATH

# Install PyTorch CPU-only first to save space/memory
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Copy backend and frontend assets with correct ownership
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user --from=frontend-builder /frontend/dist ./frontend/dist

# Create database data directory
RUN mkdir -p data

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Run FastAPI from the backend directory to resolve module imports correctly
WORKDIR /home/user/app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
