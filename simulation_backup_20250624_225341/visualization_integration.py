# src/visualization_integration.py - Integration between simulation and visualization

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional
from pathlib import Path

from .pygame_visualizer import SimulationVisualizer
from .config import STATE_DIR
from .loop_utils import get_nested


class SimulationVisualizationBridge:
    """
    Bridge between the running simulation and the Pygame visualizer.
    Handles real-time state updates and visualization management.
    """
    
    def __init__(self, visualization_config: Optional[Dict[str, Any]] = None):
        self.config = visualization_config or {}
        self.visualizer: Optional[SimulationVisualizer] = None
        self.visualization_thread: Optional[threading.Thread] = None
        self.running = False
        self.state_update_interval = self.config.get('update_interval', 0.1)  # 100ms
        self.last_state_update = 0
        
        # Window configuration
        self.width = self.config.get('width', 1200)
        self.height = self.config.get('height', 800)
        self.title = self.config.get('title', 'TheSimulation - Live View')
        
        self.logger = logging.getLogger(__name__)
        
    def start_visualization(self, initial_state: Optional[Dict[str, Any]] = None):
        """Start the visualization in a separate thread."""
        if self.running:
            self.logger.warning("Visualization already running")
            return
        
        self.running = True
        self.visualization_thread = threading.Thread(
            target=self._run_visualization_thread,
            args=(initial_state,),
            daemon=True
        )
        self.visualization_thread.start()
        self.logger.info("Visualization started in separate thread")
    
    def _run_visualization_thread(self, initial_state: Optional[Dict[str, Any]]):
        """Run the visualization in its own thread."""
        try:
            self.visualizer = SimulationVisualizer(
                width=self.width,
                height=self.height,
                title=self.title
            )
            
            if initial_state:
                self.visualizer.update_state(initial_state)
            
            self.visualizer.running = True
            
            while self.running and self.visualizer.running:
                # Handle pygame events
                import pygame
                for event in pygame.event.get():
                    self.visualizer.handle_event(event)
                
                # Update layout if needed
                if self.visualizer.layout_dirty:
                    self.visualizer._auto_layout_locations()
                    self.visualizer.layout_dirty = False
                
                # Draw
                self.visualizer.draw()
                
                # Control framerate
                self.visualizer.clock.tick(self.visualizer.fps)
            
        except Exception as e:
            self.logger.error(f"Visualization thread error: {e}", exc_info=True)
        finally:
            if self.visualizer:
                self.visualizer.cleanup()
            self.running = False
    
    def update_state(self, state: Dict[str, Any], force_update: bool = False):
        """Update the visualization with new simulation state."""
        current_time = time.time()
        
        # Throttle updates to avoid overwhelming the visualization
        if not force_update and (current_time - self.last_state_update) < self.state_update_interval:
            return
        
        if self.visualizer and self.running:
            try:
                self.visualizer.update_state(state)
                self.last_state_update = current_time
            except Exception as e:
                self.logger.error(f"Error updating visualization state: {e}")
    
    def stop_visualization(self):
        """Stop the visualization."""
        if not self.running:
            return
        
        self.running = False
        if self.visualizer:
            self.visualizer.running = False
        
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
        
        self.logger.info("Visualization stopped")
    
    def is_running(self) -> bool:
        """Check if visualization is running."""
        return self.running and (self.visualization_thread is None or self.visualization_thread.is_alive())


class VisualizationTask:
    """
    Async task that can be added to the simulation to provide real-time visualization.
    """
    
    def __init__(self, bridge: SimulationVisualizationBridge):
        self.bridge = bridge
        self.logger = logging.getLogger(__name__)
    
    async def visualization_update_task(self, state: Dict[str, Any], logger_instance: logging.Logger):
        """
        Task that runs alongside the simulation to update visualization.
        Add this to your simulation's async tasks.
        """
        logger_instance.info("[Visualization] Update task started")
        
        update_interval = 0.1  # Update every 100ms
        
        try:
            while True:
                # Update visualization with current state
                if self.bridge.is_running():
                    self.bridge.update_state(state)
                
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            logger_instance.info("[Visualization] Update task cancelled")
        except Exception as e:
            logger_instance.error(f"[Visualization] Update task error: {e}", exc_info=True)


def create_visualization_config() -> Dict[str, Any]:
    """Create default visualization configuration."""
    return {
        'enabled': True,
        'width': 1200,
        'height': 800,
        'title': 'TheSimulation - Live View',
        'update_interval': 0.1,
        'auto_layout': True,
        'show_connections': True,
        'show_simulacra': True,
        'show_ui': True,
    }


# Integration functions for adding to simulation_async.py

def initialize_visualization(state: Dict[str, Any], logger: logging.Logger, 
                           config: Optional[Dict[str, Any]] = None) -> Optional[SimulationVisualizationBridge]:
    """
    Initialize visualization for the simulation.
    Call this in run_simulation() after state is loaded.
    """
    try:
        # Check if visualization is enabled
        visualization_enabled = config and config.get('enabled', False)
        if not visualization_enabled:
            logger.info("[Visualization] Disabled in configuration")
            return None
        
        # Create and start visualization
        bridge = SimulationVisualizationBridge(config)
        bridge.start_visualization(state)
        
        logger.info("[Visualization] Initialized and started")
        return bridge
        
    except Exception as e:
        logger.error(f"[Visualization] Failed to initialize: {e}", exc_info=True)
        return None


async def add_visualization_task(tasks: list, state: Dict[str, Any], 
                               bridge: Optional[SimulationVisualizationBridge], 
                               logger: logging.Logger):
    """
    Add visualization update task to the simulation task list.
    Call this when adding other async tasks.
    """
    if bridge and bridge.is_running():
        viz_task = VisualizationTask(bridge)
        viz_update_task = asyncio.create_task(
            viz_task.visualization_update_task(state, logger)
        )
        tasks.append(viz_update_task)
        logger.info("[Visualization] Update task added to simulation")


def cleanup_visualization(bridge: Optional[SimulationVisualizationBridge], logger: logging.Logger):
    """
    Clean up visualization resources.
    Call this in simulation cleanup.
    """
    if bridge:
        bridge.stop_visualization()
        logger.info("[Visualization] Cleaned up")


# Configuration loading from environment
def load_visualization_config_from_env() -> Dict[str, Any]:
    """Load visualization configuration from environment variables."""
    import os
    
    config = create_visualization_config()
    
    # Override with environment variables if set
    if os.getenv('ENABLE_VISUALIZATION', '').lower() == 'true':
        config['enabled'] = True
    elif os.getenv('ENABLE_VISUALIZATION', '').lower() == 'false':
        config['enabled'] = False
    
    if os.getenv('VISUALIZATION_WIDTH'):
        try:
            config['width'] = int(os.getenv('VISUALIZATION_WIDTH'))
        except ValueError:
            pass
    
    if os.getenv('VISUALIZATION_HEIGHT'):
        try:
            config['height'] = int(os.getenv('VISUALIZATION_HEIGHT'))
        except ValueError:
            pass
    
    if os.getenv('VISUALIZATION_UPDATE_INTERVAL'):
        try:
            config['update_interval'] = float(os.getenv('VISUALIZATION_UPDATE_INTERVAL'))
        except ValueError:
            pass
    
    return config