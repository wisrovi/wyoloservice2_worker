docker run -d \
  --name wisrovi-agent-gpu \
  --hostname wAgent \
  --restart unless-stopped \
  --init \
  -i -t \
  --shm-size 16g \
  --cpus 6.0 \
  --memory 16g \
  --gpus all \
  --log-opt max-size=50m \
  -e TZ=Europe/Madrid \
  -v "$(pwd)":/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.ssh:/root/.ssh:ro \
  wisrovi/agents:gpu-slim-yolo
