# src/pygame_visualizer.py - Pygame-based visualization for TheSimulation

import asyncio
import json
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import pygame
import pygame.font
import pygame.gfxdraw

from .config import (
    CURRENT_LOCATION_KEY, LOCATION_DETAILS_KEY, 
    SIMULACRA_KEY, WORLD_STATE_KEY
)
from .loop_utils import get_nested


@dataclass
class LocationNode:
    """Represents a location in the visualization."""
    id: str
    name: str
    description: str
    x: float
    y: float
    width: float = 120
    height: float = 80
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []
    
    def get_rect(self) -> pygame.Rect:
        """Get pygame rect for this location."""
        return pygame.Rect(self.x - self.width/2, self.y - self.height/2, self.width, self.height)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside this location."""
        return (self.x - self.width/2 <= x <= self.x + self.width/2 and
                self.y - self.height/2 <= y <= self.y + self.height/2)


@dataclass 
class SimulacraSprite:
    """Represents a simulacra in the visualization."""
    id: str
    name: str
    location_id: str
    status: str
    x: float
    y: float
    color: Tuple[int, int, int] = (0, 100, 255)  # Blue default
    radius: float = 8
    
    def get_status_color(self) -> Tuple[int, int, int]:
        """Get color based on status."""
        status_colors = {
            "idle": (0, 200, 0),      # Green
            "busy": (255, 165, 0),    # Orange  
            "thinking": (255, 255, 0), # Yellow
            "talking": (255, 0, 255), # Magenta
            "moving": (0, 255, 255),  # Cyan
            "unknown": (128, 128, 128) # Gray
        }
        return status_colors.get(self.status, self.color)


@dataclass
class Camera:
    """Camera for panning and zooming."""
    x: float = 0
    y: float = 0
    zoom: float = 1.0
    min_zoom: float = 0.1
    max_zoom: float = 5.0
    
    def world_to_screen(self, world_x: float, world_y: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = (world_x - self.x) * self.zoom + screen_width / 2
        screen_y = (world_y - self.y) * self.zoom + screen_height / 2
        return int(screen_x), int(screen_y)
    
    def screen_to_world(self, screen_x: int, screen_y: int, screen_width: int, screen_height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - screen_width / 2) / self.zoom + self.x
        world_y = (screen_y - screen_height / 2) / self.zoom + self.y
        return world_x, world_y
    
    def zoom_at(self, screen_x: int, screen_y: int, zoom_delta: float, screen_width: int, screen_height: int):
        """Zoom in/out at specific screen position."""
        # Get world position before zoom
        world_x, world_y = self.screen_to_world(screen_x, screen_y, screen_width, screen_height)
        
        # Apply zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * zoom_delta))
        
        # Adjust camera position to keep world point under cursor
        new_screen_x, new_screen_y = self.world_to_screen(world_x, world_y, screen_width, screen_height)
        self.x += (new_screen_x - screen_x) / self.zoom
        self.y += (new_screen_y - screen_y) / self.zoom


class SimulationVisualizer:
    """Main visualization class for TheSimulation."""
    
    def __init__(self, width: int = 1200, height: int = 800, title: str = "TheSimulation Visualizer"):
        self.width = width
        self.height = height
        self.title = title
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Colors
        self.colors = {
            'background': (20, 20, 30),
            'location_box': (60, 60, 80),
            'location_border': (100, 100, 120),
            'location_text': (255, 255, 255),
            'connection_line': (80, 80, 100),
            'grid': (40, 40, 50),
            'ui_panel': (40, 40, 50),
            'ui_text': (200, 200, 200),
            'selected': (255, 200, 0),
        }
        
        # Camera and interaction
        self.camera = Camera()
        self.dragging = False
        self.drag_start = (0, 0)
        self.selected_location = None
        
        # Data structures
        self.locations: Dict[str, LocationNode] = {}
        self.simulacra: Dict[str, SimulacraSprite] = {}
        self.state_data: Dict[str, Any] = {}
        self.layout_dirty = True
        
        # Threading and state
        self.running = False
        self.state_lock = threading.Lock()
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # UI elements
        self.show_ui = True
        self.ui_panel_width = 250
        
        self.logger = logging.getLogger(__name__)
        
    def update_state(self, new_state: Dict[str, Any]):
        """Update visualization with new simulation state."""
        with self.state_lock:
            self.state_data = new_state.copy()
            self._parse_locations()
            self._parse_simulacra()
            if self.layout_dirty:
                self._auto_layout_locations()
                self.layout_dirty = False
    
    def _parse_locations(self):
        """Parse locations from state data."""
        world_state = self.state_data.get(WORLD_STATE_KEY, {})
        location_details = world_state.get(LOCATION_DETAILS_KEY, {})
        
        current_location_ids = set(location_details.keys())
        
        # Remove locations that no longer exist
        for loc_id in list(self.locations.keys()):
            if loc_id not in current_location_ids:
                del self.locations[loc_id]
        
        # Add or update locations
        for loc_id, loc_data in location_details.items():
            if loc_id not in self.locations:
                # New location - needs positioning
                self.locations[loc_id] = LocationNode(
                    id=loc_id,
                    name=loc_data.get('name', loc_id),
                    description=loc_data.get('description', ''),
                    x=0, y=0  # Will be positioned by auto_layout
                )
                self.layout_dirty = True
            else:
                # Update existing location
                self.locations[loc_id].name = loc_data.get('name', loc_id)
                self.locations[loc_id].description = loc_data.get('description', '')
            
            # Update connections
            connections = loc_data.get('connected_locations', [])
            if isinstance(connections, list):
                self.locations[loc_id].connections = [
                    conn.get('id', conn) if isinstance(conn, dict) else conn 
                    for conn in connections
                ]
    
    def _parse_simulacra(self):
        """Parse simulacra from state data."""
        simulacra_data = self.state_data.get(SIMULACRA_KEY, {})
        
        current_sim_ids = set(simulacra_data.keys())
        
        # Remove simulacra that no longer exist
        for sim_id in list(self.simulacra.keys()):
            if sim_id not in current_sim_ids:
                del self.simulacra[sim_id]
        
        # Add or update simulacra
        for sim_id, sim_data in simulacra_data.items():
            location_id = sim_data.get(CURRENT_LOCATION_KEY, 'unknown')
            status = sim_data.get('status', 'unknown')
            name = get_nested(sim_data, 'persona_details', 'Name', default=sim_id)
            
            if sim_id not in self.simulacra:
                # New simulacra
                self.simulacra[sim_id] = SimulacraSprite(
                    id=sim_id,
                    name=name,
                    location_id=location_id,
                    status=status,
                    x=0, y=0  # Will be positioned relative to location
                )
            else:
                # Update existing simulacra
                self.simulacra[sim_id].name = name
                self.simulacra[sim_id].location_id = location_id
                self.simulacra[sim_id].status = status
            
            # Position simulacra relative to their location
            self._position_simulacra_in_location(sim_id)
    
    def _position_simulacra_in_location(self, sim_id: str):
        """Position a simulacra within their current location."""
        sim = self.simulacra[sim_id]
        location = self.locations.get(sim.location_id)
        
        if not location:
            # Location doesn't exist yet, position at origin
            sim.x, sim.y = 0, 0
            return
        
        # Find all simulacra in the same location
        sims_in_location = [
            s for s in self.simulacra.values() 
            if s.location_id == sim.location_id
        ]
        
        # Position them around the location box
        count = len(sims_in_location)
        if count == 1:
            # Single simulacra - center of location
            sim.x = location.x
            sim.y = location.y
        else:
            # Multiple simulacra - arrange in circle around location
            index = list(sims_in_location).index(sim)
            angle = (2 * math.pi * index) / count
            radius = max(location.width, location.height) * 0.7
            sim.x = location.x + radius * math.cos(angle)
            sim.y = location.y + radius * math.sin(angle)
    
    def _auto_layout_locations(self):
        """Automatically layout locations using force-directed algorithm."""
        if not self.locations:
            return
        
        # Initialize positions if needed
        for i, location in enumerate(self.locations.values()):
            if location.x == 0 and location.y == 0:
                # Place in a circle initially
                angle = (2 * math.pi * i) / len(self.locations)
                location.x = 300 * math.cos(angle)
                location.y = 300 * math.sin(angle)
        
        # Force-directed layout with multiple iterations
        for iteration in range(100):
            forces = defaultdict(lambda: [0, 0])
            
            # Repulsive forces between all locations
            locations_list = list(self.locations.values())
            for i, loc1 in enumerate(locations_list):
                for j, loc2 in enumerate(locations_list):
                    if i >= j:
                        continue
                    
                    dx = loc2.x - loc1.x
                    dy = loc2.y - loc1.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 1:
                        distance = 1
                    
                    # Repulsive force (inverse square)
                    force = 5000 / (distance * distance)
                    fx = force * dx / distance
                    fy = force * dy / distance
                    
                    forces[loc1.id][0] -= fx
                    forces[loc1.id][1] -= fy
                    forces[loc2.id][0] += fx
                    forces[loc2.id][1] += fy
            
            # Attractive forces for connected locations
            for location in self.locations.values():
                for conn_id in location.connections:
                    if conn_id in self.locations:
                        other = self.locations[conn_id]
                        dx = other.x - location.x
                        dy = other.y - location.y
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance > 1:
                            # Attractive force (spring)
                            force = 0.01 * distance
                            fx = force * dx / distance
                            fy = force * dy / distance
                            
                            forces[location.id][0] += fx
                            forces[location.id][1] += fy
                            forces[other.id][0] -= fx
                            forces[other.id][1] -= fy
            
            # Apply forces with damping
            damping = 0.9
            for location in self.locations.values():
                fx, fy = forces[location.id]
                location.x += fx * damping
                location.y += fy * damping
        
        # Center the layout
        if self.locations:
            avg_x = sum(loc.x for loc in self.locations.values()) / len(self.locations)
            avg_y = sum(loc.y for loc in self.locations.values()) / len(self.locations)
            for location in self.locations.values():
                location.x -= avg_x
                location.y -= avg_y
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.QUIT:
            self.running = False
        
        elif event.type == pygame.VIDEORESIZE:
            self.width, self.height = event.w, event.h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Reset camera
                self.camera.x = 0
                self.camera.y = 0
                self.camera.zoom = 1.0
            elif event.key == pygame.K_TAB:
                # Toggle UI
                self.show_ui = not self.show_ui
            elif event.key == pygame.K_r:
                # Re-layout locations
                self.layout_dirty = True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Check if clicking on a location
                world_x, world_y = self.camera.screen_to_world(event.pos[0], event.pos[1], self.width, self.height)
                clicked_location = None
                
                for location in self.locations.values():
                    if location.contains_point(world_x, world_y):
                        clicked_location = location
                        break
                
                if clicked_location:
                    self.selected_location = clicked_location.id
                else:
                    self.selected_location = None
                    # Start dragging
                    self.dragging = True
                    self.drag_start = event.pos
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Pan camera
                dx = event.pos[0] - self.drag_start[0]
                dy = event.pos[1] - self.drag_start[1]
                self.camera.x -= dx / self.camera.zoom
                self.camera.y -= dy / self.camera.zoom
                self.drag_start = event.pos
        
        elif event.type == pygame.MOUSEWHEEL:
            # Zoom
            zoom_factor = 1.1 if event.y > 0 else 1/1.1
            mouse_pos = pygame.mouse.get_pos()
            self.camera.zoom_at(mouse_pos[0], mouse_pos[1], zoom_factor, self.width, self.height)
    
    def draw_grid(self):
        """Draw background grid."""
        grid_size = 100 * self.camera.zoom
        if grid_size < 10:
            return  # Too small to see
        
        start_x = int(-self.camera.x * self.camera.zoom + self.width/2) % int(grid_size)
        start_y = int(-self.camera.y * self.camera.zoom + self.height/2) % int(grid_size)
        
        for x in range(int(start_x), self.width, int(grid_size)):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.height))
        
        for y in range(int(start_y), self.height, int(grid_size)):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.width, y))
    
    def draw_connections(self):
        """Draw connection lines between locations."""
        with self.state_lock:
            for location in self.locations.values():
                for conn_id in location.connections:
                    if conn_id in self.locations:
                        other = self.locations[conn_id]
                        
                        # Convert to screen coordinates
                        x1, y1 = self.camera.world_to_screen(location.x, location.y, self.width, self.height)
                        x2, y2 = self.camera.world_to_screen(other.x, other.y, self.width, self.height)
                        
                        # Only draw if on screen
                        if (0 <= x1 <= self.width or 0 <= x2 <= self.width) and \
                           (0 <= y1 <= self.height or 0 <= y2 <= self.height):
                            pygame.draw.line(self.screen, self.colors['connection_line'], (x1, y1), (x2, y2), 2)
    
    def draw_locations(self):
        """Draw location boxes with labels."""
        with self.state_lock:
            for location in self.locations.values():
                # Convert to screen coordinates
                screen_x, screen_y = self.camera.world_to_screen(location.x, location.y, self.width, self.height)
                
                # Scale size with zoom
                width = location.width * self.camera.zoom
                height = location.height * self.camera.zoom
                
                # Skip if too small or off-screen
                if width < 5 or height < 5:
                    continue
                if (screen_x + width/2 < 0 or screen_x - width/2 > self.width or
                    screen_y + height/2 < 0 or screen_y - height/2 > self.height):
                    continue
                
                # Location rectangle
                rect = pygame.Rect(screen_x - width/2, screen_y - height/2, width, height)
                
                # Choose colors
                box_color = self.colors['selected'] if location.id == self.selected_location else self.colors['location_box']
                border_color = self.colors['location_border']
                
                # Draw box
                pygame.draw.rect(self.screen, box_color, rect)
                pygame.draw.rect(self.screen, border_color, rect, 2)
                
                # Draw label if big enough
                if width > 50 and height > 20:
                    text = self.font_medium.render(location.name, True, self.colors['location_text'])
                    text_rect = text.get_rect(center=(screen_x, screen_y))
                    self.screen.blit(text, text_rect)
    
    def draw_simulacra(self):
        """Draw simulacra sprites."""
        with self.state_lock:
            for sim in self.simulacra.values():
                # Convert to screen coordinates
                screen_x, screen_y = self.camera.world_to_screen(sim.x, sim.y, self.width, self.height)
                
                # Scale radius with zoom
                radius = max(3, sim.radius * self.camera.zoom)
                
                # Skip if off-screen
                if (screen_x + radius < 0 or screen_x - radius > self.width or
                    screen_y + radius < 0 or screen_y - radius > self.height):
                    continue
                
                # Draw simulacra circle
                color = sim.get_status_color()
                pygame.draw.circle(self.screen, color, (int(screen_x), int(screen_y)), int(radius))
                pygame.draw.circle(self.screen, (255, 255, 255), (int(screen_x), int(screen_y)), int(radius), 2)
                
                # Draw name if zoomed in enough
                if radius > 8:
                    text = self.font_small.render(sim.name, True, (255, 255, 255))
                    text_rect = text.get_rect(center=(screen_x, screen_y + radius + 10))
                    self.screen.blit(text, text_rect)
    
    def draw_ui(self):
        """Draw UI panels and information."""
        if not self.show_ui:
            return
        
        # UI panel background
        panel_rect = pygame.Rect(self.width - self.ui_panel_width, 0, self.ui_panel_width, self.height)
        pygame.draw.rect(self.screen, self.colors['ui_panel'], panel_rect)
        pygame.draw.line(self.screen, self.colors['ui_text'], 
                        (self.width - self.ui_panel_width, 0), 
                        (self.width - self.ui_panel_width, self.height), 2)
        
        # UI content
        y_offset = 10
        margin = 10
        
        # Title
        title_text = self.font_large.render("Simulation View", True, self.colors['ui_text'])
        self.screen.blit(title_text, (self.width - self.ui_panel_width + margin, y_offset))
        y_offset += 40
        
        # Camera info
        cam_text = [
            f"Zoom: {self.camera.zoom:.2f}",
            f"Camera: ({self.camera.x:.0f}, {self.camera.y:.0f})",
            f"Locations: {len(self.locations)}",
            f"Simulacra: {len(self.simulacra)}"
        ]
        
        for text in cam_text:
            rendered = self.font_small.render(text, True, self.colors['ui_text'])
            self.screen.blit(rendered, (self.width - self.ui_panel_width + margin, y_offset))
            y_offset += 20
        
        y_offset += 20
        
        # Controls
        controls_text = [
            "Controls:",
            "Mouse wheel: Zoom",
            "Left drag: Pan",
            "Left click: Select location",
            "Space: Reset camera",
            "Tab: Toggle UI",
            "R: Re-layout"
        ]
        
        for text in controls_text:
            color = self.colors['ui_text'] if not text.endswith(':') else (255, 255, 255)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (self.width - self.ui_panel_width + margin, y_offset))
            y_offset += 16
        
        # Selected location info
        if self.selected_location and self.selected_location in self.locations:
            y_offset += 20
            location = self.locations[self.selected_location]
            
            # Location details
            details_text = [
                f"Selected: {location.name}",
                f"ID: {location.id}",
                f"Connections: {len(location.connections)}"
            ]
            
            for text in details_text:
                rendered = self.font_small.render(text, True, (255, 255, 255))
                self.screen.blit(rendered, (self.width - self.ui_panel_width + margin, y_offset))
                y_offset += 16
            
            # Simulacra in location
            sims_here = [s for s in self.simulacra.values() if s.location_id == self.selected_location]
            if sims_here:
                y_offset += 10
                sim_title = self.font_small.render("Simulacra here:", True, (255, 255, 255))
                self.screen.blit(sim_title, (self.width - self.ui_panel_width + margin, y_offset))
                y_offset += 16
                
                for sim in sims_here:
                    sim_text = f"• {sim.name} ({sim.status})"
                    rendered = self.font_small.render(sim_text, True, sim.get_status_color())
                    self.screen.blit(rendered, (self.width - self.ui_panel_width + margin, y_offset))
                    y_offset += 16
    
    def draw(self):
        """Main drawing function."""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw connections
        self.draw_connections()
        
        # Draw locations
        self.draw_locations()
        
        # Draw simulacra
        self.draw_simulacra()
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def run_standalone(self, state_file: str = None):
        """Run visualizer in standalone mode."""
        self.running = True
        
        # Load initial state if provided
        if state_file:
            try:
                with open(state_file, 'r') as f:
                    initial_state = json.load(f)
                self.update_state(initial_state)
            except Exception as e:
                self.logger.error(f"Failed to load state file {state_file}: {e}")
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                self.handle_event(event)
            
            # Update layout if needed
            if self.layout_dirty:
                self._auto_layout_locations()
                self.layout_dirty = False
            
            # Draw
            self.draw()
            
            # Control framerate
            self.clock.tick(self.fps)
        
        pygame.quit()
    
    def cleanup(self):
        """Clean up resources."""
        pygame.quit()