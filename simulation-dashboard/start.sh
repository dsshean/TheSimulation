#!/bin/bash

echo "🚀 Starting TheSimulation Dashboard..."
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if Tauri CLI is available
if ! command -v tauri &> /dev/null; then
    echo "🔧 Installing Tauri CLI..."
    npm install -g @tauri-apps/cli@next
fi

echo "🎯 Starting development server..."
echo "The dashboard will open automatically."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

npm run tauri:dev