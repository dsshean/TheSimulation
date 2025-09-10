@echo off
echo 🤖 TheSimulation Docker Setup and Runner
echo ========================================

:: Check if .env file exists
if not exist ".env" (
    echo ❌ No .env file found in the current directory
    echo    Please ensure .env file exists with your API keys
    echo    The .env file should contain your GOOGLE_API_KEY and other configuration
    pause
    exit /b 1
)

:: Check for Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not in PATH
    echo    Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed or not in PATH
    echo    Please install Docker Compose
    pause
    exit /b 1
)

echo ✅ Docker is available

:: Create data directories
echo 📁 Creating data directories...
if not exist "data" mkdir data
if not exist "data\states" mkdir data\states
if not exist "data\life_summaries" mkdir data\life_summaries
if not exist "data\narrative_images" mkdir data\narrative_images
if not exist "logs" mkdir logs
if not exist "logs\events" mkdir logs\events

:: Check for existing data
dir /b "data\states" 2>nul | findstr . >nul
if %errorlevel% equ 0 (
    echo 📂 Found existing simulation states
) else (
    echo 📂 No existing states - will create new simulation
)

dir /b "data\life_summaries" 2>nul | findstr . >nul
if %errorlevel% equ 0 (
    echo 👥 Found existing life summaries
) else (
    echo 👥 No existing personas - will generate new ones
)

:: Build and run
echo.
echo 🐳 Building and starting TheSimulation...
echo    This may take several minutes on first run...

:: Stop any existing containers
docker-compose down

:: Build and start
docker-compose up --build -d

if %errorlevel% equ 0 (
    echo.
    echo 🎉 TheSimulation backend is running!
    echo.
    echo 🗄️  Redis:        localhost:6379
    echo 🔌 Socket Server: localhost:8765
    echo 📡 WebSocket:     localhost:8766
    echo.
    echo 🖥️  To start the Tauri UI:
    echo    cd simulation-dashboard
    echo    npm run tauri dev
    echo.
    echo 📋 To monitor backend logs:
    echo    docker-compose logs -f thesimulation
    echo.
    echo 🛑 To stop backend:
    echo    docker-compose down
    echo.
    echo 🔄 To restart backend:
    echo    docker-compose restart thesimulation
    echo.
    
    :: Wait for services to start
    echo ⏳ Waiting for services to start...
    timeout /t 5 /nobreak >nul
    
    :: Show recent logs
    echo 📝 Recent logs:
    docker-compose logs --tail=20 thesimulation
    
    echo.
    echo 🔍 For continuous monitoring run:
    echo    docker-compose logs -f thesimulation
    
) else (
    echo ❌ Failed to start TheSimulation
    echo Check logs with: docker-compose logs thesimulation
)

echo.
echo Press any key to exit...
pause >nul