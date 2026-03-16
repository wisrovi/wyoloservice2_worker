# Makefile for Worker Invoker
# Author: William Rodríguez - wisrovi

.PHONY: up down restart status logs

# Detectar IP del sistema
IP_SYSTEM := $(shell hostname -I | awk '{print $$1}')
export WORKER_NAME=$(IP_SYSTEM)
export REDIS_URL=redis://192.168.10.252:23437/0

start:
	@echo "Levantando stack para el worker: $(WORKER_NAME)"
	docker-compose up -d

stop:
	@echo "Deteniendo y eliminando el stack"
	docker-compose down

restart:
	@echo "Reiniciando stack..."
	docker-compose restart

status:
	docker-compose ps

logs:
	docker-compose logs -f

into:
	@echo "Entrando al contenedor del worker..."
	docker-compose exec worker_invoker bash