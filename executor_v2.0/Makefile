
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

stop:
	docker-compose -f docker-compose.yaml --env-file ./config/user.env down  --remove-orphans

into:
	docker-compose -f docker-compose.yaml --env-file ./config/user.env  exec executor zsh






# python main.py --file "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml"
trainer_example:
	docker run --rm -it \
		--name executor \
		--hostname "${USER}" \
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
		--env NVIDIA_VISIBLE_DEVICES=0 \
		--env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--env TZ=Europe/Madrid \
		--env-file config/control_host.env \
		--env-file config/user.env \
		--label autoheal=true \
		--label com.centurylinklabs.watchtower.enable=true \
		--log-driver json-file \
		--log-opt max-size=10m \
		--log-opt max-file=3 \
		-v /home/wyolo/events:/wyolo/worker/events \
		-v /home/wyolo/train_service_results:/wyolo/worker/train_service_results \
		wisrovi/wyolo_trainer:v2.1.0 \
		zsh

trainer_example_2:
	docker run --rm -it \
		--name executor \
		--hostname "${USER}" \
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
		--env NVIDIA_VISIBLE_DEVICES=0 \
		--env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--env TZ=Europe/Madrid \
		--env-file config/control_host.env \
		--env-file config/user.env \
		--label autoheal=true \
		--label com.centurylinklabs.watchtower.enable=true \
		--log-driver json-file \
		--log-opt max-size=10m \
		--log-opt max-file=3 \
		-v /home/wyolo/events:/wyolo/worker/events \
		-v /home/wyolo/train_service_results:/wyolo/worker/train_service_results \
		wisrovi/wyolo_trainer:v2.1.0 \
		bash -c '/usr/local/bin/mount-cifs.sh && python main.py --file "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml"'