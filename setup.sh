#!/bin/bash

# Development setup script for Hyperion
# This script helps set up the development environment and run tests

set -e  # Exit on any error

echo "🚀 Hyperion Development Setup"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check Docker
if command -v docker &> /dev/null; then
    print_status "Docker is installed"
    if docker info &> /dev/null; then
        print_status "Docker is running"
    else
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
else
    print_error "Docker is not installed. Please install Docker Desktop."
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    print_status "Docker Compose is available"
else
    print_error "Docker Compose is not available. Please update Docker Desktop."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src/app" ]; then
    print_error "Please run this script from the Hyperion project root directory."
    exit 1
fi

print_status "All prerequisites met!"

# Function to build and start services
start_services() {
    echo ""
    echo "🔨 Building and starting services..."
    
    cd deploy/docker
    
    # Stop any existing containers
    docker-compose down -v 2>/dev/null || true
    
    # Build and start services
    docker-compose up --build -d
    
    print_status "Services started!"
    
    # Wait for services to be healthy
    echo ""
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for app to be healthy
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec -T app curl -f http://localhost:8000/healthz &> /dev/null; then
            health_response=$(docker-compose exec -T app curl -s http://localhost:8000/healthz)
            if echo "$health_response" | grep -q '"ok":true'; then
                print_status "Hyperion is ready!"
                break
            else
                echo "  Attempt $((attempt + 1)): Model still loading..."
            fi
        else
            echo "  Attempt $((attempt + 1)): Service starting..."
        fi
        
        sleep 3
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Service failed to start within timeout"
        echo ""
        echo "Logs from the app container:"
        docker-compose logs app
        exit 1
    fi
    
    cd ../..
}

# Function to run tests
run_tests() {
    echo ""
    echo "🧪 Running API tests..."
    
    # Create test script if it doesn't exist
    cat > test_api.py << 'EOF'
#!/usr/bin/env python3
"""
Simple script to test the Hyperion API endpoints.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_basic_endpoints():
    """Test basic endpoints."""
    print("Testing basic endpoints...")
    
    # Root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
            data = response.json()
            print(f"  Service: {data.get('service')}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Root endpoint error: {str(e)}")
        return False
    
    # Health endpoint
    try:
        response = requests.get(f"{BASE_URL}/healthz")
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                print("✓ Health check passed")
                print(f"  Model loaded: {data.get('model_loaded')}")
            else:
                print("⚠ Service unhealthy but responding")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {str(e)}")
        return False
    
    return True

def test_chat_functionality():
    """Test chat endpoint."""
    print("\nTesting chat functionality...")
    
    test_data = {
        "prompt": "Hello! How are you?",
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v1/llm/chat", json=test_data, timeout=60)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Chat endpoint working")
            print(f"  Model: {data.get('model')}")
            print(f"  Response: '{data.get('response')[:50]}...'")
            print(f"  Tokens: {data.get('tokens_used')}")
            print(f"  Time: {elapsed_time:.2f}s")
            return True
        elif response.status_code == 503:
            print("⚠ Chat endpoint returned 503 - model may still be loading")
            return False
        else:
            print(f"✗ Chat endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Chat endpoint error: {str(e)}")
        return False

def main():
    if not test_basic_endpoints():
        return 1
    
    if not test_chat_functionality():
        return 1
    
    print("\n🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    # Run the tests
    if python3 test_api.py; then
        print_status "All tests passed!"
    else
        print_error "Some tests failed. Check the output above."
        return 1
    fi
    
    # Clean up test file
    rm -f test_api.py
}

# Function to show logs
show_logs() {
    echo ""
    echo "📋 Showing service logs..."
    cd deploy/docker
    docker-compose logs --tail=50 -f
}

# Function to stop services
stop_services() {
    echo ""
    echo "🛑 Stopping services..."
    cd deploy/docker
    docker-compose down -v
    print_status "Services stopped!"
    cd ../..
}

# Function to show status
show_status() {
    echo ""
    echo "📊 Service Status"
    echo "=================="
    
    cd deploy/docker
    
    if docker-compose ps | grep -q "Up"; then
        docker-compose ps
        
        echo ""
        echo "Service URLs:"
        echo "  • API: http://localhost:8000"
        echo "  • Health: http://localhost:8000/healthz"
        echo "  • Metrics: http://localhost:8000/metrics"
        echo "  • Redis: localhost:6379"
        echo ""
        echo "To view logs: ./setup.sh logs"
        echo "To run tests: ./setup.sh test"
        echo "To stop: ./setup.sh stop"
    else
        print_warning "Services are not running. Use './setup.sh start' to start them."
    fi
    
    cd ../..
}

# Function to show help
show_help() {
    echo ""
    echo "Hyperion Development Setup Script"
    echo "=================================="
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Build and start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  test      - Run API tests"
    echo "  logs      - Show service logs"
    echo "  status    - Show service status"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start services"
    echo "  $0 test     # Run tests"
    echo "  $0 logs     # View logs"
    echo ""
}

# Main script logic
case "${1:-start}" in
    "start")
        start_services
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        start_services
        show_status
        ;;
    "test")
        run_tests
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac