# Docker Build & Run Guide

## Build Docker Image
```bash
docker build -t pose3d-server:latest .
```

## Run Container (Basic)
```bash
docker run -p 8000:8000 pose3d-server:latest
```

## Run Container (With VLM API URL)
```bash
docker run -p 8000:8000 \
  -e VLM_API_URL="https://your-colab-ngrok-url/analyze-pair" \
  pose3d-server:latest
```

## Run Container (With Volume for Videos)
```bash
docker run -p 8000:8000 \
  -e VLM_API_URL="https://your-colab-ngrok-url/analyze-pair" \
  -v /path/to/local/videos:/app/videos \
  pose3d-server:latest
```

## Test the Server
```bash
curl http://localhost:8000/
```

## Push to Docker Hub (Optional)
```bash
docker tag pose3d-server:latest your-username/pose3d-server:latest
docker push your-username/pose3d-server:latest
```

## Docker Compose (Optional)
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  pose3d:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VLM_API_URL=https://your-colab-ngrok-url/analyze-pair
    volumes:
      - ./videos:/app/videos
    restart: unless-stopped
```

Run with:
```bash
docker-compose up --build
```

## Notes
- Image size: ~2GB (due to MediaPipe + OpenCV)
- Python version: 3.10 (slim variant)
- GPU support: Not included in base image
- If you need GPU: Add `--gpus all` flag or use `nvidia/cuda:11.8-runtime-ubuntu22.04` base image
