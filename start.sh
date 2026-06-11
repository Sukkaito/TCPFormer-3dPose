#!/bin/bash
set -e

# Start Ollama server if available
if command -v ollama >/dev/null 2>&1; then
  echo "Starting ollama serve..."
  nohup ollama serve > /var/log/ollama.log 2>&1 &
  sleep 8
  echo "Model baked into image; skipping pull"
else
  echo "ollama not installed, skipping ollama serve"
fi

# Start VLM service (background)
echo "Starting VLM service on port 8001"
python vlm.py &

# Start main pose server (foreground)
echo "Starting local pose 3d server on port 8000"
exec python local_pose_3d_server.py
