#!/bin/bash

# TheSimulation Docker Runner Script
echo "🤖 TheSimulation Docker Setup and Runner"
echo "========================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ No .env file found in the current directory"
    echo "   Please ensure .env file exists with your API keys"
    echo "   The .env file should contain your GOOGLE_API_KEY and other configuration"
    exit 1
fi

# Source environment variables
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for required API key
if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "<YOUR_GOOGLE_API_KEY>" ]; then
    echo "❌ GOOGLE_API_KEY is not set or still has placeholder value"
    echo "   Please edit .env file and set a valid Google API key"
    exit 1
fi

echo "✅ Google API Key is configured"

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/states
mkdir -p data/life_summaries
mkdir -p data/narrative_images
mkdir -p logs/events

# Check if we have existing data
if [ "$(ls -A data/states 2>/dev/null)" ]; then
    echo "📂 Found existing simulation states"
else
    echo "📂 No existing states - will create new simulation"
fi

if [ "$(ls -A data/life_summaries 2>/dev/null)" ]; then
    echo "👥 Found existing life summaries"
else
    echo "👥 No existing personas - will generate new ones"
fi

# Build and run with Docker Compose
echo ""
echo "🐳 Building and starting TheSimulation..."
echo "   This may take several minutes on first run..."

# Stop any existing containers
docker-compose down

# Build and start
docker-compose up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TheSimulation backend is running!"
    echo ""
    echo "🗄️  Redis:        localhost:6379"
    echo "🔌 Socket Server: localhost:8765"  
    echo "📡 WebSocket:     localhost:8766"
    echo ""
    echo "🖥️  To start the Tauri UI:"
    echo "   cd simulation-dashboard"
    echo "   npm run tauri dev"
    echo ""
    echo "📋 To monitor backend logs:"
    echo "   docker-compose logs -f thesimulation"
    echo ""
    echo "🛑 To stop backend:"
    echo "   docker-compose down"
    echo ""
    echo "🔄 To restart backend:"
    echo "   docker-compose restart thesimulation"
    echo ""
    
    # Wait for services to start
    echo "⏳ Waiting for services to start..."
    sleep 5
    
    # Check container health
    if docker-compose ps | grep -q "Up"; then
        echo "✅ Container is running"
        
        # Show real-time logs for a few seconds
        echo "📝 Recent logs:"
        docker-compose logs --tail=20 thesimulation
        
        echo ""
        echo "🔍 For continuous monitoring run:"
        echo "   docker-compose logs -f thesimulation"
        
    else
        echo "❌ Container failed to start. Check logs:"
        docker-compose logs thesimulation
    fi
    
else
    echo "❌ Failed to start TheSimulation"
    echo "Check logs with: docker-compose logs thesimulation"
fi