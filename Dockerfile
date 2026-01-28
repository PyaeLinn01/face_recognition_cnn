# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, MTCNN, and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default MongoDB connection for Docker (connects to host MongoDB)
# On macOS/Windows, use host.docker.internal to access host machine
# On Linux, use host.docker.internal or --network host when running docker
# local mongodb 
ENV MONGODB_CONNECTION_STRING=mongodb://host.docker.internal:27017/ 
ENV MONGODB_DATABASE_NAME=face_attendance

# MLflow tracking URI (file-based by default, can be overridden)
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# DVC remote configuration (optional)
ENV DVC_REMOTE=local

# Create startup script that uses environment variables
RUN echo '#!/bin/bash\n\
# Use environment variables if set, otherwise use defaults\n\
export MONGODB_CONN=${MONGODB_CONNECTION_STRING:-mongodb://host.docker.internal:27017/}\n\
export MONGODB_DB=${MONGODB_DATABASE_NAME:-face_attendance}\n\
\n\
# Start Streamlit with environment variables\n\
exec streamlit run attend_app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose Streamlit default port
EXPOSE 8501
EXPOSE 27017

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run startup script
CMD ["/app/start.sh"]
