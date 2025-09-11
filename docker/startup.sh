#!/bin/bash

echo "Starting TheSimulation Docker Container..."

# Load environment variables from mounted .env file if it exists
if [ -f "/app/.env" ]; then
    echo "🔧 Loading environment variables from mounted .env file..."
    set -a  # automatically export all variables
    source /app/.env
    set +a  # turn off automatic export
    echo "✅ Environment variables loaded"
else
    echo "ℹ️  No .env file found, using environment variables from docker-compose"
fi

# Verify required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set. Simulation will fail."
    echo "   Please ensure your .env file contains GOOGLE_API_KEY"
    exit 1
fi

echo "✅ Google API Key is configured"

# Set default values for optional environment variables
export INSTANCE_UUID=${INSTANCE_UUID:-""}
export OVERRIDE_LOCATION=${OVERRIDE_LOCATION:-""}
export OVERRIDE_MOOD=${OVERRIDE_MOOD:-""}

# Create data directories if they don't exist
mkdir -p /app/data/states
mkdir -p /app/data/life_summaries  
mkdir -p /app/data/narrative_images
mkdir -p /app/logs/events

# Check if we have existing data
if [ "$(ls -A /app/data/states)" ]; then
    echo "Found existing simulation states in /app/data/states"
else
    echo "No existing states found. Will create new simulation."
fi

if [ "$(ls -A /app/data/life_summaries)" ]; then
    echo "Found existing life summaries in /app/data/life_summaries"
else
    echo "No existing life summaries found. Will generate new personas."
fi

# Redis will be started by supervisor, just wait for it to be ready
echo "Waiting for Redis to be ready (managed by supervisor)..."
for i in {1..60}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Redis failed to start after 60 seconds"
        echo "Checking Redis logs..."
        cat /var/log/supervisor/redis_error.log 2>/dev/null || echo "No Redis error log found"
        exit 1
    fi
    sleep 1
done

# Test Python imports
echo "Testing Python dependencies..."
python -c "
import sys
try:
    import google.generativeai as genai
    import redis
    import asyncio
    from src.simulation_async import APP_NAME
    print('✅ All dependencies imported successfully')
    print(f'✅ Application name: {APP_NAME}')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Python dependency check failed"
    exit 1
fi

echo "Starting Supervisor to manage all services..."

# Start supervisor which will manage Redis, Simulation, and Dashboard
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf