SHELL := /bin/bash

.PHONY: run dev test build docker-build

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t hyperion:local -f deploy/docker/Dockerfile .

compose:
	docker compose -f deploy/docker/docker-compose.yml up --build
