#!/bin/bash
trap 'kill 0; exit 0' SIGINT SIGTERM
cd "$(dirname "$0")"
echo "Sketch Bomb → http://localhost:8000"
uvicorn webapp.backend:app --host 0.0.0.0 --port 8000 &
wait
