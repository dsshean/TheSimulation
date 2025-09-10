#!/bin/bash

# Full TheSimulation Startup Script - Backend + Tauri UI
echo "🤖 Starting Full TheSimulation (Backend + UI)"
echo "============================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ No .env file found in the current directory"
    echo "   Please ensure .env file exists with your API keys"
    exit 1
fi

# Start Docker backend
echo "🐳 Starting backend services (Redis + Simulation)..."
./run-docker.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to start backend services"
    exit 1
fi

# Wait for backend to be ready
echo "⏳ Waiting for backend services to be ready..."
sleep 10

# Check if Redis is responding
for i in {1..30}; do
    if nc -z localhost 6379; then
        echo "✅ Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Redis not responding after 30 seconds"
        echo "   Check backend logs: docker-compose logs thesimulation"
        exit 1
    fi
    sleep 1
done

# Check Node.js and Tauri dependencies
echo "🔧 Checking Tauri UI dependencies..."
cd simulation-dashboard

if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Node.js dependencies"
        exit 1
    fi
fi

# Check if Rust is installed for Tauri
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust/Cargo not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Check if Tauri CLI is installed
if ! command -v cargo-tauri &> /dev/null; then
    echo "📦 Installing Tauri CLI..."
    cargo install tauri-cli
fi

echo ""
echo "🚀 Starting Tauri Desktop UI..."
echo "   Backend services are running in Docker"
echo "   UI will connect to:"
echo "   - Redis: localhost:6379"
echo "   - Socket Server: localhost:8765"  
echo "   - WebSocket: localhost:8766"
echo ""

# Start Tauri in development mode
npm run tauri dev