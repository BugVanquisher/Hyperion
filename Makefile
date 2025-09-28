SHELL := /bin/bash

.PHONY: run dev test test-cov lint format check security docker-build compose install

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

install:
	pip install -e ".[dev]"
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black src/ tests/
	isort src/ tests/

check:
	black --check src/ tests/
	isort --check-only src/ tests/

security:
	safety check
	bandit -r src/

ci: check lint test

docker-build:
	docker build -t hyperion:local -f deploy/docker/Dockerfile .

compose:
	docker compose -f deploy/docker/docker-compose.yml up --build
