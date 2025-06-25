#!/usr/bin/env python3
"""
Standalone visualization launcher for TheSimulation.

This script can be used to visualize simulation state files or connect to a running simulation.

Usage:
    python visualizer.py [state_file.json]
    python visualizer.py --live [--port PORT]
    python visualizer.py --help
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pygame_visualizer import SimulationVisualizer
from src.visualization_integration import SimulationVisualizationBridge


def setup_logging(level=logging.INFO):
    """Setup logging for the visualizer."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def find_latest_state_file():
    """Find the most recent simulation state file."""
    state_dir = Path("data/states")
    if not state_dir.exists():
        return None
    
    state_files = list(state_dir.glob("simulation_state_*.json"))
    if not state_files:
        return None
    
    # Sort by modification time, newest first
    state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return state_files[0]


def load_state_file(file_path):
    """Load simulation state from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file {file_path}: {e}")
        return None


def run_file_visualizer(state_file):
    """Run visualizer with a state file."""
    print(f"Loading state from: {state_file}")
    
    state_data = load_state_file(state_file)
    if not state_data:
        return False
    
    # Create and run visualizer
    visualizer = SimulationVisualizer(
        width=1200,
        height=800,
        title=f"TheSimulation Visualizer - {Path(state_file).name}"
    )
    
    visualizer.update_state(state_data)
    
    print("Visualization Controls:")
    print("  Mouse Wheel: Zoom in/out")
    print("  Left Click + Drag: Pan view")
    print("  Left Click on location: Select location")
    print("  Space: Reset camera view")
    print("  Tab: Toggle UI panel")
    print("  R: Re-layout locations")
    print("  Close window to exit")
    print()
    
    visualizer.run_standalone()
    return True


def run_live_visualizer(port=12345):
    """Run visualizer connected to live simulation."""
    print(f"Starting live visualizer (listening on port {port})")
    print("This feature requires the simulation to be running with visualization enabled.")
    print()
    
    # For now, just watch for state file changes
    print("Watching for state file changes...")
    
    visualizer = SimulationVisualizer(
        width=1200,
        height=800,
        title="TheSimulation - Live View"
    )
    
    last_state_file = None
    last_modified = 0
    
    visualizer.running = True
    
    try:
        while visualizer.running:
            # Check for latest state file
            latest_file = find_latest_state_file()
            
            if latest_file and latest_file != last_state_file:
                # New state file found
                state_data = load_state_file(latest_file)
                if state_data:
                    visualizer.update_state(state_data)
                    print(f"Updated from: {latest_file.name}")
                    last_state_file = latest_file
                    last_modified = latest_file.stat().st_mtime
            
            elif latest_file and latest_file.stat().st_mtime > last_modified:
                # Existing file was modified
                state_data = load_state_file(latest_file)
                if state_data:
                    visualizer.update_state(state_data)
                    last_modified = latest_file.stat().st_mtime
            
            # Handle events and draw
            import pygame
            for event in pygame.event.get():
                visualizer.handle_event(event)
            
            if visualizer.layout_dirty:
                visualizer._auto_layout_locations()
                visualizer.layout_dirty = False
            
            visualizer.draw()
            visualizer.clock.tick(30)  # Lower FPS for file watching
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        visualizer.cleanup()


def run_demo():
    """Run a demonstration with sample data."""
    print("Running visualization demo with sample data...")
    
    # Create sample state data
    sample_state = {
        "world_time": 150.5,
        "current_world_state": {
            "location_details": {
                "Home_01": {
                    "name": "Home",
                    "description": "A cozy living space with comfortable furniture",
                    "connected_locations": [
                        {"id": "Kitchen_01"},
                        {"id": "Garden_01"}
                    ]
                },
                "Kitchen_01": {
                    "name": "Kitchen", 
                    "description": "A modern kitchen with stainless steel appliances",
                    "connected_locations": [
                        {"id": "Home_01"},
                        {"id": "Dining_01"}
                    ]
                },
                "Garden_01": {
                    "name": "Garden",
                    "description": "A peaceful garden with flowers and trees",
                    "connected_locations": [
                        {"id": "Home_01"},
                        {"id": "Park_01"}
                    ]
                },
                "Dining_01": {
                    "name": "Dining Room",
                    "description": "An elegant dining room with a large table",
                    "connected_locations": [
                        {"id": "Kitchen_01"},
                        {"id": "Living_01"}
                    ]
                },
                "Living_01": {
                    "name": "Living Room", 
                    "description": "A spacious living room with comfortable seating",
                    "connected_locations": [
                        {"id": "Dining_01"}
                    ]
                },
                "Park_01": {
                    "name": "Public Park",
                    "description": "A large park with walking paths and benches",
                    "connected_locations": [
                        {"id": "Garden_01"}
                    ]
                }
            }
        },
        "simulacra_profiles": {
            "alice": {
                "persona_details": {
                    "Name": "Alice Smith"
                },
                "current_location": "Home_01",
                "status": "idle"
            },
            "bob": {
                "persona_details": {
                    "Name": "Bob Johnson"
                },
                "current_location": "Kitchen_01", 
                "status": "busy"
            },
            "charlie": {
                "persona_details": {
                    "Name": "Charlie Brown"
                },
                "current_location": "Garden_01",
                "status": "thinking"
            }
        }
    }
    
    visualizer = SimulationVisualizer(
        width=1200,
        height=800,
        title="TheSimulation Visualizer - Demo"
    )
    
    visualizer.update_state(sample_state)
    
    print("Demo visualization loaded!")
    print("This shows sample locations and simulacra to demonstrate the interface.")
    print()
    print("Visualization Controls:")
    print("  Mouse Wheel: Zoom in/out")
    print("  Left Click + Drag: Pan view") 
    print("  Left Click on location: Select location")
    print("  Space: Reset camera view")
    print("  Tab: Toggle UI panel")
    print("  R: Re-layout locations")
    print("  Close window to exit")
    print()
    
    visualizer.run_standalone()


def main():
    parser = argparse.ArgumentParser(
        description="TheSimulation Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualizer.py                          # Use latest state file or demo
  python visualizer.py state_file.json         # Visualize specific state file
  python visualizer.py --live                  # Watch for live updates
  python visualizer.py --demo                  # Run with sample data
        """
    )
    
    parser.add_argument(
        'state_file', 
        nargs='?', 
        help='Path to simulation state JSON file'
    )
    
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Watch for live simulation updates'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true', 
        help='Run demonstration with sample data'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=12345,
        help='Port for live connection (default: 12345)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Check if pygame is available
    try:
        import pygame
    except ImportError:
        print("Error: pygame is not installed.")
        print("Install it with: pip install pygame")
        return 1
    
    # Determine what to run
    if args.demo:
        run_demo()
    elif args.live:
        run_live_visualizer(args.port)
    elif args.state_file:
        if not os.path.exists(args.state_file):
            print(f"Error: State file '{args.state_file}' not found")
            return 1
        if not run_file_visualizer(args.state_file):
            return 1
    else:
        # Try to find latest state file, otherwise run demo
        latest_file = find_latest_state_file()
        if latest_file:
            print(f"Found latest state file: {latest_file}")
            if not run_file_visualizer(latest_file):
                print("Failed to load state file, running demo instead")
                run_demo()
        else:
            print("No state files found, running demo")
            run_demo()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())