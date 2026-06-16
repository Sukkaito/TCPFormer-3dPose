#!/bin/bash
set -e

# Start main pose server (foreground)
echo "Starting local pose 3d server on port 8000"
exec python local_pose_3d_server.py
