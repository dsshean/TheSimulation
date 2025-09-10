@echo off
echo 🤖 Starting Full TheSimulation (Backend + UI)
echo =============================================

:: Check if .env file exists
if not exist ".env" (
    echo ❌ No .env file found in the current directory
    echo    Please ensure .env file exists with your API keys
    pause
    exit /b 1
)

:: Start Docker backend
echo 🐳 Starting backend services (Redis + Simulation)...
call run-docker.bat

if %errorlevel% neq 0 (
    echo ❌ Failed to start backend services
    pause
    exit /b 1
)

:: Wait for backend to be ready
echo ⏳ Waiting for backend services to be ready...
timeout /t 10 /nobreak >nul

:: Check if Redis is responding
echo 🔍 Checking if Redis is ready...
for /l %%i in (1,1,30) do (
    netstat -an | findstr ":6379" >nul 2>&1
    if not errorlevel 1 (
        echo ✅ Redis is ready
        goto redis_ready
    )
    if %%i==30 (
        echo ❌ Redis not responding after 30 seconds
        echo    Check backend logs: docker-compose logs thesimulation
        pause
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
)

:redis_ready

:: Change to dashboard directory
cd simulation-dashboard

:: Check Node.js dependencies
echo 🔧 Checking Tauri UI dependencies...
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Node.js dependencies
        pause
        exit /b 1
    )
)

:: Check if Rust is installed
echo 🔍 Checking for Rust/Cargo...
cargo --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Rust/Cargo not found
    echo    Please install Rust from: https://rustup.rs/
    echo    Then run this script again
    pause
    exit /b 1
)

:: Check if Tauri CLI is available
cargo tauri --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing Tauri CLI...
    cargo install tauri-cli
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Tauri CLI
        pause
        exit /b 1
    )
)

echo.
echo 🚀 Starting Tauri Desktop UI...
echo    Backend services are running in Docker
echo    UI will connect to:
echo    - Redis: localhost:6379
echo    - Socket Server: localhost:8765
echo    - WebSocket: localhost:8766
echo.

:: Start Tauri in development mode
npm run tauri dev

:: If we get here, Tauri has closed
echo.
echo 🛑 Tauri UI has closed
echo    Backend services are still running in Docker
echo    To stop backend: docker-compose down
echo.
pause