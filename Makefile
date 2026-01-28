.PHONY: help build run stop logs clean install run-local test docker-shell

# Variables
IMAGE_NAME = face-attendance-allinone
CONTAINER_NAME = face-attendance
PORT_FRONTEND = 8080
PORT_API = 5001


# Default target
help:
	@echo "Face Recognition CNN - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "🐳 Docker (All-in-One):"
	@echo "  make build          - Build the all-in-one Docker image"
	@echo "  make run            - Run the container"
	@echo "  make start          - Build and run (one command)"
	@echo "  make stop           - Stop the container"
	@echo "  make restart        - Restart the container"
	@echo "  make logs           - View container logs"
	@echo "  make shell          - Open shell in container"
	@echo "  make clean          - Remove container and image"
	@echo ""
	@echo "💻 Local Development (no Docker):"
	@echo "  make install        - Install all dependencies"
	@echo "  make run-api        - Run API server locally"
	@echo "  make run-frontend   - Run frontend locally"
	@echo "  make run-local      - Run Streamlit app locally"
	@echo ""
	@echo "📋 Default Accounts:"
	@echo "  Admin:   admin@gmail.com / 123456"
	@echo "  Teacher: teacher@gmail.com / 123456"
	@echo ""
	@echo "🌐 Access (after running):"
	@echo "  Frontend: http://localhost:8080"
	@echo "  API:      http://localhost:5001"
	

# ==================== DOCKER ALL-IN-ONE ====================

# Build the all-in-one image
build:
	@echo "🔨 Building all-in-one Docker image..."
	docker build -f Dockerfile.allinone -t $(IMAGE_NAME) .
	@echo "✅ Build complete!"

# Run the container
run:
	@echo "🚀 Starting container..."
	@docker run -d \
		-p $(PORT_FRONTEND):8080 \
		-p $(PORT_API):5001 \
		-e MONGODB_CONNECTION_STRING=mongodb://host.docker.internal:27017/ \
		-v $(PWD)/images:/app/images \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) 2>/dev/null || docker start $(CONTAINER_NAME)
	@echo ""
	@echo "✅ Container started!"
	@echo "   Frontend: http://localhost:$(PORT_FRONTEND)"
	@echo "   API:      http://localhost:$(PORT_API)"
	@echo ""
	@echo "⚠️  MongoDB must be running separately on localhost:27017"

# Build and run in one command
start: build run
	@echo "✅ Ready to use!"

# Stop the container
stop:
	@echo "🛑 Stopping container..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"
	@echo "✅ Stopped"

# Restart
restart: stop run

# View logs
logs:
	@docker logs -f $(CONTAINER_NAME)

# Shell access
shell:
	@docker exec -it $(CONTAINER_NAME) /bin/bash

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "✅ Cleanup complete"

# ==================== LOCAL DEVELOPMENT ====================

# Install dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	cd face-attend-main && npm install
	@echo "✅ Dependencies installed"

# Run API locally
run-api:
	@echo "🐍 Running API server locally..."
	python3 api_server.py

# Run frontend locally
run-frontend:
	@echo "⚛️ Running frontend locally..."
	cd face-attend-main && npm run dev

# Run Streamlit locally
run-local:
	@echo "📊 Running Streamlit app locally..."
	streamlit run attend_app.py --server.port=8501 --server.address=0.0.0.0

# Clean Python cache
clean-cache:
	@echo "🧹 Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cache cleaned"
