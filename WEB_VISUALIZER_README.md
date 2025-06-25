# TheSimulation Web Visualizer

A modern, real-time web-based visualization system that replaces the pygame visualizer with an interactive D3.js graph.

## Features

- **Real-time Updates**: Direct WebSocket connection to simulation state
- **Interactive Graph**: Force-directed layout similar to t-SNE for natural positioning
- **Agent Tracking**: Live agent movement, status, and activity monitoring
- **Location Mapping**: Visual representation of world locations and connections
- **Object Visualization**: Display of ephemeral objects in each location
- **Rich Information Panel**: Detailed agent info, location details, and recent events
- **Responsive Design**: Works on desktop and mobile browsers
- **No Installation Required**: Pure web-based, runs in any modern browser

## Quick Start

### 1. Start the Simulation
```bash
# Make sure websockets is installed
pip install websockets

# Start the main simulation
python main_async.py
```

### 2. Start the Web Visualizer
In a separate terminal:
```bash
# Start the HTTP server for the web interface
python start_visualizer.py
```

### 3. Open Your Browser
The visualizer will automatically open at: **http://localhost:8080**

Or manually navigate to: http://localhost:8080

## How It Works

### Architecture
```
TheSimulation (port 8766) ←WebSocket→ Web Browser (port 8080)
        ↓                                    ↓
   Real-time state               Interactive D3.js visualization
```

**Note**: Port 8765 is used for the main client API, port 8766 is for visualization WebSocket.

### Data Flow
1. **Simulation State**: The main simulation streams state updates via WebSocket
2. **Real-time Updates**: Browser receives live updates every 500ms
3. **Interactive Features**: Click nodes for details, drag to rearrange, zoom/pan
4. **Bidirectional**: Browser can request detailed info from simulation

## Visualization Elements

### Node Types
- **🟢 Locations** (Green circles): World locations with size based on object count
- **🔴 Agents** (Red circles): Simulacra with color indicating status
  - Green: Idle
  - Red: Busy
  - Orange: Thinking
- **🔵 Objects** (Blue circles): Ephemeral objects in the world

### Connections
- **Solid lines**: Location connections (doorways, paths)
- **Dashed lines**: Agent-location relationships (where agents are)

### Interactive Features
- **Mouse Controls**:
  - Drag nodes to reposition
  - Scroll to zoom in/out
  - Click nodes for detailed information
- **Control Buttons**:
  - Reset View: Return to default zoom/position
  - Pause/Resume: Stop/start live updates
  - Center on Agents: Focus view on simulacra

### Information Panel
- **Active Agents**: Live list with status and current actions
- **Locations**: All world locations with object/NPC counts
- **Recent Events**: Latest narrative events from the simulation
- **World Status**: Overall simulation statistics

## Configuration

### Environment Variables
```bash
# Enable/disable web visualization (default: True)
ENABLE_WEB_VISUALIZATION=true

# WebSocket port for real-time data (default: 8766)
VISUALIZATION_WEBSOCKET_PORT=8766

# HTTP port for web interface (default: 8080)
VISUALIZATION_HTTP_PORT=8080
```

### Customization
Edit these files to customize the visualization:
- `web_visualizer.html`: Layout and styling
- `simulation_viz.js`: Graph behavior and interactions
- `start_visualizer.py`: HTTP server configuration

## Troubleshooting

### Connection Issues
- **"Connecting..." stuck**: Make sure the simulation is running with `python main_async.py`
- **Port conflicts**: Change ports in `.env` or `start_visualizer.py`
- **WebSocket errors**: Check firewall settings for ports 8766 and 8080

### Performance
- **Slow updates**: Increase update interval in `simulation_viz.js` (line ~778)
- **Large simulations**: Visualization auto-limits object display for performance
- **Memory usage**: Browser handles up to hundreds of nodes efficiently

### Browser Support
- **Recommended**: Chrome, Firefox, Safari, Edge (modern versions)
- **Required**: WebSocket and D3.js support
- **Mobile**: Works on mobile devices with touch controls

## Advanced Features

### API Requests
The visualizer can request detailed information:
```javascript
// Request agent details
websocket.send(JSON.stringify({
    type: 'get_agent_details',
    agent_id: 'sim_sustdj'
}));

// Request location details  
websocket.send(JSON.stringify({
    type: 'get_location_details',
    location_id: 'Kitchen_Home_01'
}));
```

### Real-time Data Format
The WebSocket streams data in this format:
```json
{
    "type": "simulation_state",
    "world_time": 153.57,
    "simulacra": {
        "sim_sustdj": {
            "name": "Daniel Rodriguez",
            "current_location": "LivingRoom_Apartment_01",
            "status": "busy",
            "current_action": "Action: wait - Details: ..."
        }
    },
    "locations": {
        "Kitchen_Home_01": {
            "name": "Kitchen",
            "object_count": 8,
            "connections": ["LivingRoom_Apartment_01"]
        }
    }
}
```

## Comparison with Pygame Visualizer

| Feature | Pygame | Web Visualizer |
|---------|--------|----------------|
| Installation | pygame dependency | Browser only |
| Real-time | File polling | WebSocket streaming |
| Interactivity | Limited | Full mouse/touch |
| Accessibility | Desktop only | Any device |
| Customization | Python code | HTML/CSS/JS |
| Performance | Good for small sims | Scales better |
| Graph Layout | Basic positioning | Force-directed (t-SNE-like) |

## Development

### Extending the Visualizer
To add new features:
1. **Backend**: Modify `_prepare_visualization_data()` in `core_tasks.py`
2. **Frontend**: Update `simulation_viz.js` to handle new data
3. **Styling**: Edit CSS in `web_visualizer.html`

### Testing
```bash
# Test WebSocket connection
python -c "import websockets; print('WebSockets available')"

# Test HTTP server
python start_visualizer.py
# Should open browser automatically
```

---

**Note**: This web visualizer completely replaces the pygame system and provides a much more modern, accessible, and feature-rich visualization experience for TheSimulation.