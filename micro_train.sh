#!/bin/bash
# Docker run command for testing/debugging the executor container
# Usage: ./micro_train.sh [-d] [--bash] --config "/wyolo/worker/request/config_train.yaml"
#   -d:      Run in detached mode (daemon). Default: foreground with auto-remove.
#   --bash:  Keep container alive with tail -f /dev/null for debugging.

set -euo pipefail

CONFIG_FILE="/wyolo/worker/request/config_train.yaml"
DETACHED=false
BASH_MODE=false

# Samba credentials (ajustar según entorno)
CONTROL_HOST="192.168.10.252"
CIFS_USER="wisrovi"
CIFS_PASS="wyoloservice"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            DETACHED=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --bash)
            BASH_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-d] [--bash] --config <config_file_path>"
            exit 1
            ;;
    esac
done

if [[ "$BASH_MODE" == false ]]; then
    # Determine host path for validation (volume is mounted at /home/wyolo/request -> /wyolo/worker/request)
    if [[ "$CONFIG_FILE" == /wyolo/worker/request/* ]]; then
        HOST_CONFIG="/home/wyolo/request/${CONFIG_FILE#/wyolo/worker/request/}"
    elif [[ "$CONFIG_FILE" == /home/wyolo/request/* ]]; then
        HOST_CONFIG="$CONFIG_FILE"
    else
        HOST_CONFIG="$CONFIG_FILE"
    fi

    # Validate config file exists on host
    if [[ ! -f "$HOST_CONFIG" ]]; then
        echo "ERROR: Config file not found: $HOST_CONFIG" >&2
        exit 1
    fi
fi

# Convert to container path for the --file argument
if [[ "$CONFIG_FILE" == /home/wyolo/request/* ]]; then
    CONTAINER_CONFIG="/wyolo/worker/request/${CONFIG_FILE#/home/wyolo/request/}"
elif [[ "$CONFIG_FILE" == /wyolo/worker/request/* ]]; then
    CONTAINER_CONFIG="$CONFIG_FILE"
else
    CONTAINER_CONFIG="$CONFIG_FILE"
fi

# Build docker run command based on mode
if [[ "$DETACHED" == true ]]; then
    DOCKER_RUN_ARGS=(-d --name wyolo_executor_test)
else
    if [[ -t 0 ]]; then
        DOCKER_RUN_ARGS=(--rm -it --name wyolo_executor_test)
    else
        DOCKER_RUN_ARGS=(--rm -i --name wyolo_executor_test)
    fi
fi

if [[ "$BASH_MODE" == true ]]; then
    DETACHED=false
    CMD="zsh"
    DOCKER_RUN_ARGS=(--rm -it --name wyolo_executor_test)
else
    CMD="nvidia-smi && echo \"[EXECUTOR] Starting mount...\" && /usr/local/bin/mount-cifs.sh && echo \"[EXECUTOR] Mount OK. Starting training...\" && python main.py --file $CONTAINER_CONFIG"
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
  --hostname default_user \
  --privileged \
  --network host \
  --shm-size=16g \
  --cpus=8 \
  --memory=24g \
  --cap-add=SYS_ADMIN \
  --cap-add=DAC_READ_SEARCH \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_RESOURCE \
  --gpus '"device=0"' \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TZ=Europe/Madrid \
  -e PYTHONUNBUFFERED=1 \
  -e CONTROL_HOST="$CONTROL_HOST" \
  -e CIFS_USER="$CIFS_USER" \
  -e CIFS_PASS="$CIFS_PASS" \
  -v /home/wyolo/events:/wyolo/worker/events:rw \
  -v /home/wyolo/train_service_results:/wyolo/worker/train_service_results:rw \
  -v /home/wyolo/request:/wyolo/worker/request:rw \
  wisrovi/train_service:worker_executor_v1.0.0 \
  bash -c "$CMD"