.PHONY: help build run stop logs clean install run-local test docker-shell

# Variables
IMAGE_NAME = face-recognition-cnn
CONTAINER_NAME = face-recognition
PORT = 8501

# Default target
help:
	@echo "Face Recognition CNN - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build          - Build Docker image"
	@echo "  make run            - Run Docker container"
	@echo "  make stop           - Stop running container"
	@echo "  make logs           - View container logs"
	@echo "  make restart        - Restart container"
	@echo "  make docker-shell   - Open shell in running container"
	@echo "  make clean          - Remove container and image"
	@echo ""
	@echo "Local Development:"
	@echo "  make install        - Install Python dependencies locally"
	@echo "  make run-local      - Run app locally (without Docker)"
	@echo ""
	@echo "Utility:"
	@echo "  make clean-cache    - Clean Python cache files"
	@echo "  make clean-all      - Clean everything (cache, containers, images)"

# Docker Commands
build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running container: $(CONTAINER_NAME) on port $(PORT)"
	@docker run -d -p $(PORT):8501 --name $(CONTAINER_NAME) $(IMAGE_NAME) || \
		docker start $(CONTAINER_NAME)
	@echo "Container is running. Access the app at http://localhost:$(PORT)"

stop:
	@echo "Stopping container: $(CONTAINER_NAME)"
	@docker stop $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"

logs:
	@echo "Viewing logs for: $(CONTAINER_NAME)"
	@docker logs -f $(CONTAINER_NAME)

restart: stop run

docker-shell:
	@echo "Opening shell in container: $(CONTAINER_NAME)"
	@docker exec -it $(CONTAINER_NAME) /bin/bash

# Clean up
clean:
	@echo "Removing container and image..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Cleanup complete"

clean-cache:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cache cleaned"

clean-all: clean clean-cache
	@echo "All cleanup complete"

# Local Development
install:
	@echo "Installing Python dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "Dependencies installed"

run-local:
	@echo "Running app locally..."
	streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0

# Rebuild and run (useful for development)
rebuild: clean build run

# Quick start (build and run in one command)
start: build run
	@echo "Build and run complete. Access at http://localhost:$(PORT)"
