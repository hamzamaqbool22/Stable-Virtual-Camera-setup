#!/bin/bash
# Paste this into the Vast.ai template "On-start Script" field.
# Starts Jupyter (from launch mode) plus FastAPI on :8000.
# GPU/SEVA install is NOT done here — run ./setup_vast.sh in Jupyter after boot.
env >> /etc/environment
mkdir -p /workspace
cd /workspace

if [ ! -d "/workspace/Stable-Virtual-Camera-setup" ]; then
  git clone https://github.com/hamzamaqbool22/Stable-Virtual-Camera-setup.git
fi

cd /workspace/Stable-Virtual-Camera-setup
pip3 install -r requirements-api.txt

pkill -f "uvicorn main:app" || true
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
