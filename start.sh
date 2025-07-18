#!/bin/bash
# Railway startup script với port handling

# Set default port if not provided
PORT=${PORT:-8080}

echo "🚀 Starting Streamlit app on port $PORT"

# Start streamlit with proper port
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
