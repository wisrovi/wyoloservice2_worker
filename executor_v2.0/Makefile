
user.env:
	echo "USER=$$(whoami)" > user.env
	echo "TZ=Europe/Madrid" >> user.env
	echo "WORKER_HOST=$$(hostname -I | awk '{print $$1}')" >> user.env
	echo "WORKER_HOSTNAME=$$(hostname)" >> user.env
	echo "WORKER_OS=$$(uname -s)" >> user.env
	echo "WORKER_KERNEL_VERSION=$$(uname -r)" >> user.env
	echo "WORKER_CPU_CORES=$$(nproc)" >> user.env
	echo "WORKER_GATEWAY=$$(ip route | grep default | awk '{print $$3}')" >> user.env
	echo "WORKER_NETWORK_INTERFACE=$$(ip route | grep default | awk '{print $$5}')" >> user.env
	echo "WORKER_DOCKER_VERSION=$$(docker --version | awk '{print $$3}' | sed 's/,//')" >> user.env
	echo "WORKER_APP_BASE_PATH=$$(pwd)" >> user.env
	echo "WORKER_APP_ENV=production" >> user.env
	echo "WORKER_HOME_DIR=$$HOME" >> user.env
	echo "WORKER_CURRENT_DATE=$$(date '+%Y-%m-%d')" >> user.env
	echo "WORKER_CURRENT_TIME=$$(date '+%H:%M:%S')" >> user.env
	echo "WORKER_GPU_COUNT=$$(nvidia-smi --query-gpu=count --format=csv,noheader)" >> user.env
	echo "WORKER_GPU_MODEL=$$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" >> user.env
	echo "WORKER_GPU_MEMORY=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)" >> user.env
	echo "WORKER_GPU_MEMORY=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)" >> user.env
	MEM_TOTAL=$$(awk '/^MemTotal:/ {print $$2}' /proc/meminfo); \
    echo "WORKER_RAM_MEMORY=$$((MEM_TOTAL / 1048576 * 8 / 10))g" >> user.env
	echo "WORKER_CPU_CORES=$$((`nproc` - 1)).0" >> user.env

config.py:
	pip install customtkinter
	python production/config.py

start: user.env config.py
	mv user.env ./config/
	docker-compose -f docker-compose.yaml --env-file ./config/user.env  --compatibility up -d --build --force-recreate --no-deps  --pull always

build:
	docker-compose -f docker-compose.yaml  --env-file ./config/user.env  build
	docker-compose -f docker-compose.son.yaml  --env-file ./config/user.env  build

stop:
	docker-compose -f docker-compose.yaml --env-file ./config/user.env down  --remove-orphans

into:
	docker-compose -f docker-compose.yaml --env-file ./config/user.env  exec worker zsh

son_prepare:
	docker-compose -f docker-compose.son.yaml  --env-file ./config/user.env  up -d --build --force-recreate
	docker-compose -f docker-compose.son.yaml  --env-file ./config/user.env  down

son_test:
	docker-compose -f docker-compose.son.yaml --env-file ./config/user.env  up --build --force-recreate

trainer_example:
	docker run --rm \
	--name wyolo_son \
	--privileged \
	--cap-add=SYS_ADMIN \
	--cap-add=DAC_READ_SEARCH \
	--cap-add=NET_ADMIN \
	--cap-add=SYS_RESOURCE \
	--network host \
	--hostname "willia.rodriguez" \
	--shm-size="24g" \
	--memory="24g" \
	--cpus=6 \
	--gpus '"device=0"' \
	-e NVIDIA_VISIBLE_DEVICES=0 \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	-e TZ=Europe/Madrid \
	--env-file config/control_host.env \
	--env-file config/user.env \
	-v /home/wyolo/events:/wyolo/worker/events \
	-v /home/wyolo/train_service_results:/wyolo/worker/train_service_results \
	wisrovi/wyolo_trainer:v2.0.0 \
	python /app/wtrain.py --file /app/lib/datasets_config_examples/clasification/config_train.yaml