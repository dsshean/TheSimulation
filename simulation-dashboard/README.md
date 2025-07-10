# TheSimulation Dashboard

A modern desktop application for monitoring TheSimulation - an LLM-driven autonomous agent simulation system.

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Backend**: Rust (Tauri 2.0)
- **UI Framework**: Tailwind CSS v4
- **Animations**: Framer Motion
- **State Management**: React hooks
- **Build Tool**: Vite

## Features

- 🌍 **Real-time World State** - Monitor simulation time, weather, and active agents
- 👥 **Agent Status** - Track individual simulacra status, actions, and remaining time
- 📍 **Location Info** - View current location details, objects, NPCs, and connections
- 📊 **Event Logs** - Real-time event monitoring with categorized views
- 🔄 **Auto-refresh** - Automatic data updates every 2 seconds
- 🎨 **Modern UI** - Clean, responsive design with smooth animations
- 💻 **Cross-platform** - Desktop app for Linux, macOS, and Windows

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Rust (for Tauri backend)
- A running TheSimulation instance

### Installation

1. **Navigate to dashboard directory**:
   ```bash
   cd simulation-dashboard
   ```

2. **Run the startup script** (recommended):
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   npm install
   npm install -g @tauri-apps/cli@next  # if not installed
   npm run tauri:dev
   ```

3. **Build for production**:
   ```bash
   npm run tauri:build
   ```

Note: The first run will take a few minutes as it downloads and compiles Rust dependencies.

## Usage

1. **Start TheSimulation**: Make sure your simulation is running with state files in `data/states/`
2. **Launch Dashboard**: Run `npm run tauri:dev` for development or use the built executable
3. **Monitor**: The dashboard will automatically find and load the latest simulation state

## Data Sources

The dashboard reads from:
- `data/states/simulation_state_*.json` - Current simulation state
- `logs/events/events_latest_*.jsonl` - Real-time event logs

## Features Overview

### World State Panel
- Current simulation time
- Weather conditions
- Active agent count
- World UUID and description

### Agent Status Panel
- Individual agent status (idle/busy/thinking)
- Current actions and remaining time
- Agent location and goals

### Location Info Panel
- Current location details
- Available objects and NPCs
- Connected locations and exits

### Event Log Tabs
- **Simulacra**: Agent thoughts and intents
- **World Engine**: Action resolutions and outcomes
- **Narrative**: Story generation and descriptions
- **All Events**: Combined view of all events

## Architecture

### Frontend (React)
- Modern React 18 with TypeScript
- Custom hooks for simulation data management
- Responsive component library
- Real-time updates via polling

### Backend (Rust/Tauri)
- File system operations for reading simulation data
- JSON parsing and state management
- Cross-platform desktop app framework
- Secure native API calls

## Development

### Project Structure
```
simulation-dashboard/
├── src/
│   ├── components/     # React components
│   ├── hooks/          # Custom React hooks
│   ├── types/          # TypeScript interfaces
│   ├── utils/          # Utility functions
│   ├── App.tsx         # Main app component
│   └── main.tsx        # React entry point
├── src-tauri/
│   ├── src/
│   │   └── main.rs     # Rust backend
│   ├── Cargo.toml      # Rust dependencies
│   └── tauri.conf.json # Tauri configuration
├── package.json
└── vite.config.ts
```

### Available Scripts

- `npm run dev` - Start Vite dev server
- `npm run build` - Build for production
- `npm run tauri:dev` - Start Tauri development mode
- `npm run tauri:build` - Build desktop application

## Migration from Textual

This dashboard replaces the previous Textual-based interface (`src/textual_dashboard.py`) with a modern desktop application offering:

- Better performance and responsiveness
- Modern UI with smooth animations
- Cross-platform compatibility
- Enhanced visualization capabilities
- Real-time event filtering and categorization

## License

Part of TheSimulation project - see main project license.