#!/bin/bash

# Face Attendance System - Docker Startup Script

echo "🚀 Starting Face Attendance System..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
echo "📦 Building and starting services..."
docker-compose up --build -d mongodb api-server frontend

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access the application:"
echo "   - Frontend:  http://localhost:8080"
echo "   - API:       http://localhost:5001"
echo "   - MongoDB:   mongodb://localhost:27017"
echo ""
echo "📋 Default Accounts:"
echo "   - Admin:    admin@gmail.com / 123456"
echo "   - Teacher:  teacher@gmail.com / 123456"
echo "   - Students: Register via signup"
echo ""
echo "🛑 To stop: docker-compose down"
echo "📜 View logs: docker-compose logs -f"
