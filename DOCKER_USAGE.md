# TheSimulation Docker Setup

This Docker setup runs the TheSimulation backend services (Redis + Simulation) in containers, while the Tauri desktop UI runs on the host and connects to the containerized services.

## Prerequisites

1. **Docker & Docker Compose** installed
2. **Node.js** (for Tauri UI)
3. **Rust/Cargo** (for Tauri compilation)
4. **Existing .env file** in the root directory with your API keys

## Required .env File

Your `.env` file must exist in the root directory and contain:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional model settings
MODEL_GEMINI_PRO=gemini-2.0-flash
SEARCH_AGENT_MODEL_NAME=gemini-2.0-flash

# Optional feature flags
ENABLE_NARRATIVE_IMAGE_GENERATION=false
ENABLE_BLUESKY_POSTING=false
ENABLE_WEB_VISUALIZATION=true

# Optional social media (if ENABLE_BLUESKY_POSTING=true)
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

## Quick Start Options

### Option 1: Full Setup (Backend + UI) - Recommended

**Windows:**
```cmd
start-full-simulation.bat
```

**Linux/Mac:**
```bash
chmod +x start-full-simulation.sh
./start-full-simulation.sh
```

### Option 2: Backend Only (Manual UI Start)

**Windows:**
```cmd
run-docker.bat
:: Then in a separate terminal:
cd simulation-dashboard
npm run tauri dev
```

**Linux/Mac:**
```bash
./run-docker.sh
# Then in a separate terminal:
cd simulation-dashboard
npm run tauri dev
```

### Option 3: Manual Docker Compose
```bash
docker-compose up --build -d
cd simulation-dashboard
npm run tauri dev
```

## What Gets Started

### Docker Container (Backend):
- **Redis Server** (port 6379) - Message queuing
- **TheSimulation** - Main Python application

### Host System (UI):
- **Tauri Desktop App** - The actual user interface

## Data Persistence

Your simulation data is automatically preserved in:

- `./data/states/` - Simulation state files
- `./data/life_summaries/` - Agent personas
- `./data/narrative_images/` - Generated images
- `./logs/` - Application logs

These directories are mounted as Docker volumes, so data persists between container restarts.

## Access Points

- **Tauri Desktop UI**: Launches automatically
- **Redis**: localhost:6379
- **Socket Server**: localhost:8765
- **WebSocket**: localhost:8766

## Monitoring

```bash
# View logs
docker-compose logs -f thesimulation

# Check status
docker-compose ps

# Restart services
docker-compose restart thesimulation

# Stop everything
docker-compose down
```

## Troubleshooting

### Container won't start:
```bash
docker-compose logs thesimulation
```

### Missing API key error:
- Ensure `.env` file exists in root directory
- Verify `GOOGLE_API_KEY` is set in `.env`

### Redis connection issues:
```bash
docker-compose restart thesimulation
```

### Performance issues:
Edit `docker-compose.yml` resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 4G  # Increase memory
      cpus: '4.0' # Increase CPU
```

## Building for Production

For production deployment, create a `.env.prod` file and use:

```bash
docker-compose -f docker-compose.yml --env-file .env.prod up -d
```