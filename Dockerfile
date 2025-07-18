FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Copy requirements first (better caching)
COPY requirements_railway.txt requirements.txt

# Install packages with error handling
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/signatures data/test

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

# Run the app
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
