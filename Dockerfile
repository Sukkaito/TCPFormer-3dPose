FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

WORKDIR /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV VLM_API_URL="http://localhost:8000"

# Install runtime system dependencies (kept together for single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    zstd \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache for deps
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy small application files (keep large model weights last to avoid cache invalidation)
COPY start.sh vlm.py local_pose_3d_server.py tcpformer_model.py train.pkl ./

# Prepare runtime directories and permissions
RUN mkdir -p /app/videos && chmod +x /app/start.sh

# Copy large model weight last (minimize rebuilds of earlier layers)
COPY TCPFormer_ap3d_81.pth.tr ./

# Install Ollama and bake model into image (this step is large; keep at end)
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama --version
RUN ollama pull qwen2.5vl:7b

# Expose ports (optional, services communicate via localhost inside container)
EXPOSE 8000
EXPOSE 11434

# Start script will launch Ollama (if present), VLM service and pose server
CMD ["/bin/bash", "./start.sh"]
