
"""
Improved Textual dashboard for TheSimulation with better layout and agent focus.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import glob

from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Log, Static, DataTable, Label, RichLog, TabbedContent, TabPane
)


class WorldStatePanel(Static):
    """Panel showing world state with live running time."""
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.state_data = {}
        self.last_update_time = datetime.now()
        self.activity_indicator = "⏸️"
        self.last_sim_time = 0.0
        self.sim_time_increment = 0.0
        self.time_start = datetime.now()
    
    def update_state(self, state_data: Dict[str, Any]):
        """Update with new state data."""
        self.state_data = state_data
        current_time = datetime.now()
        self.last_update_time = current_time
        self.activity_indicator = "🔄"  # Show activity
        
        current_sim_time = state_data.get('world_time', 0.0)
        # DEBUG: Always update the last_sim_time to show current value
        self.last_sim_time = current_sim_time
        self.time_start = current_time
        
        self.set_timer(1.0, self._reset_activity)
        self._update_display()
    
    def _update_display(self):
        """Update the display with current information."""
        if not self.state_data:
            self.update("🌍 World State: Loading simulation...")
            return
        
        # Directly use the sim time from state - no interpolation needed
        interpolated_time = self.last_sim_time
        
        weather = self.state_data.get('world_feeds', {}).get('weather', {}).get('condition', 'Unknown')
        active_count = len(self.state_data.get('active_simulacra_ids', []))
        
        if interpolated_time < 60:
            time_display = f"{interpolated_time:.1f}s"
        elif interpolated_time < 3600:
            minutes = int(interpolated_time // 60)
            seconds = interpolated_time % 60
            time_display = f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(interpolated_time // 3600)
            minutes = int((interpolated_time % 3600) // 60)
            seconds = interpolated_time % 60
            time_display = f"{hours}h {minutes}m {seconds:.0f}s"
        
        content = f"🌍 World Time: {time_display} | 🌡️ {weather} | 👥 {active_count} Agents | {self.activity_indicator}"
        self.update(content)
    
    def _reset_activity(self):
        """Reset activity indicator."""
        self.activity_indicator = "⏸️"
        self._update_display()


class DetailedStatusPanel(Static):
    """Panel showing detailed simulation status."""
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.state_data = {}
    
    def update_state(self, state_data: Dict[str, Any]):
        self.state_data = state_data
        self._update_display()
    
    def _update_display(self):
        if not self.state_data:
            self.update("📊 Status: Loading...")
            return
        
        content = ["[bold]🕐 Time & Status[/bold]"]
        content.append("─" * 20)
        
        sim_time = self.state_data.get('world_time', 0.0)
        time_display = f"{sim_time:.1f}s"
        content.append(f"[bold]Time:[/bold] {time_display}")
        
        world_uuid = self.state_data.get('world_instance_uuid', 'N/A')
        content.append(f"[bold]UUID:[/bold] {str(world_uuid)[:8]}...")
        
        # World description
        world_template = self.state_data.get('world_template_details', {})
        world_desc = world_template.get('description', 'No description')
        content.append(f"[bold]Desc:[/bold] {world_desc}")
        
        # Get current location
        simulacra = self.state_data.get('simulacra_profiles', {})
        active_sim_ids = self.state_data.get('active_simulacra_ids', [])
        if active_sim_ids and simulacra:
            primary_sim = simulacra.get(active_sim_ids[0], {})
            location_id = primary_sim.get('current_location', 'Unknown')
            location_details = self.state_data.get('current_world_state', {}).get('location_details', {})
            location_name = location_details.get(location_id, {}).get('name', location_id)
            content.append(f"[bold]Location:[/bold] {location_name}")
        
        content.append("\n[bold]🌐 World Feeds[/bold]")
        world_feeds = self.state_data.get('world_feeds', {})
        weather = world_feeds.get('weather', {}).get('condition', 'N/A')
        content.append(f"🌡️ [bold]Weather:[/bold] {weather}")
        
        # Add news feeds
        local_news = world_feeds.get('local_news', {}).get('headlines', [])
        if local_news and len(local_news) > 0:
            content.append(f"📰 [bold]Local:[/bold] {local_news[0]}")
        
        world_news = world_feeds.get('world_news', {}).get('headlines', [])
        if world_news and len(world_news) > 0:
            content.append(f"🌍 [bold]World:[/bold] {world_news[0]}")
        
        regional_news = world_feeds.get('regional_news', {}).get('headlines', [])
        if regional_news and len(regional_news) > 0:
            content.append(f"🗺️ [bold]Regional:[/bold] {regional_news[0]}")
        
        self.update("\n".join(content))


class AgentStatusTable(DataTable):
    """Table showing agent status."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_column("Agent", key="agent")
        self.add_column("Status", key="status") 
        self.add_column("Action", key="action")
        self.add_column("Duration", key="duration")
    
    def update_state(self, state_data: Dict[str, Any]):
        self.clear()
        simulacra = state_data.get('simulacra_profiles', {})
        active_sim_ids = state_data.get('active_simulacra_ids', [])
        current_time = state_data.get('world_time', 0.0)
        
        if not simulacra or not active_sim_ids:
            self.add_row("No Agents", "Loading", "...", "")
            return
        
        for sim_id in active_sim_ids:
            sim_data = simulacra.get(sim_id, {})
            name = sim_data.get('persona_details', {}).get('Name', sim_id)
            status = sim_data.get('status', 'Unknown')
            action_desc = sim_data.get('current_action_description', 'Idle')
            
            # Calculate duration for busy actions
            duration_text = ""
            if status == "busy":
                action_end_time = sim_data.get('current_action_end_time', 0.0)
                if action_end_time > current_time:
                    remaining = action_end_time - current_time
                    duration_text = f"{remaining:.1f}s left"
                else:
                    duration_text = "Completing..."
            
            # No truncation needed for action description
            status_emoji = "🟢" if status == "idle" else "🟡" if status == "busy" else "🔴"
            self.add_row(name, f"{status_emoji} {status}", action_desc, duration_text)


class LocationDataTable(DataTable):
    """Table for objects and NPCs in the current location."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_column("Type", key="type", width=4)
        self.add_column("Name", key="name", width=18)
        self.add_column("Description", key="desc", width=30)
    
    def update_location_data(self, state_data: Dict[str, Any]):
        self.clear()
        simulacra = state_data.get('simulacra_profiles', {})
        active_sim_ids = state_data.get('active_simulacra_ids', [])
        
        if not active_sim_ids or not simulacra:
            self.add_row("...")
            return
        
        primary_sim = simulacra.get(active_sim_ids[0], {})
        primary_location_id = primary_sim.get('current_location')
        
        if not primary_location_id:
            self.add_row("N/A", "No Location", "...")
            return

        location_details = state_data.get('world_state', {}).get('location_details', {})
        location_info = location_details.get(primary_location_id, {})
        location_name = location_info.get('name', primary_location_id)
        
        self.add_row("📍", f"[bold]{location_name}[/bold]", "")
        
        for obj in location_info.get('ephemeral_objects', [])[:4]:
            self.add_row("📦", obj.get('name', '?'), obj.get('description', ''))
        for npc in location_info.get('ephemeral_npcs', [])[:3]:
            self.add_row("👤", npc.get('name', '?'), npc.get('description', ''))

class AgentOutputLog(RichLog):
    """A simple RichLog widget for displaying formatted agent output."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_cache = []
        self.max_events = 50
        
    def add_event(self, event_data: Dict[str, Any]):
        """Add an event to the log with proper formatting, latest first."""
        sim_time = event_data.get('sim_time_s', 0.0)
        agent_id = event_data.get('agent_id', 'Unknown')
        event_type = event_data.get('event_type', 'unknown')
        data = event_data.get('data', {})
        
        time_str = f"T{sim_time:.1f}s"
        panel = None
        
        if event_type == 'monologue':
            monologue = data.get('monologue', '')
            panel = Panel(
                monologue, 
                title=f"💭 {agent_id} - {time_str}",
                title_align="left",
                border_style="blue"
            )
            
        elif event_type == 'intent':
            action_type = data.get('action_type', 'unknown')
            details = data.get('details', '')
            target_id = data.get('target_id', '')
            
            intent_text = f"Action: {action_type}"
            if target_id:
                intent_text += f"\nTarget: {target_id}"
            if details:
                intent_text += f"\nDetails: {details}"
                
            panel = Panel(
                intent_text,
                title=f"🎯 {agent_id} Intent - {time_str}",
                title_align="left", 
                border_style="yellow"
            )
            
        elif event_type == 'resolution':
            valid_action = data.get('valid_action', False)
            duration = data.get('duration', 0.0)
            outcome = data.get('outcome_description', 'No description')
            
            # Add connections and ephemeral objects info
            results = data.get('results', {})
            discovered_objects = results.get('discovered_objects', [])
            discovered_connections = results.get('discovered_connections', [])
            
            status_emoji = "✅" if valid_action else "❌"
            resolution_text = f"Duration: {duration}s\n{outcome}"
            
            if discovered_objects:
                resolution_text += f"\n\n📦 Objects Found: {len(discovered_objects)}"
                for obj in discovered_objects[:3]:  # Show first 3
                    resolution_text += f"\n- {obj.get('name', 'Unknown')}: {obj.get('description', '')}"
                    
            if discovered_connections:
                resolution_text += f"\n\n🚪 Connections Found: {len(discovered_connections)}"
                for conn in discovered_connections[:3]:  # Show first 3
                    resolution_text += f"\n- To {conn.get('to_location_id_hint', 'Unknown')}: {conn.get('description', '')}"
                
            panel = Panel(
                resolution_text,
                title=f"{status_emoji} World Engine - {time_str}",
                title_align="left",
                border_style="green" if valid_action else "red"
            )
            
        elif event_type == 'narrative':
            narrative_text = data.get('narrative_text', '')
            panel = Panel(
                narrative_text,
                title=f"📖 Narrator - {time_str}",
                title_align="left",
                border_style="purple"
            )
        
        if panel:
            # Add to cache and keep latest first
            self.events_cache.insert(0, panel)
            if len(self.events_cache) > self.max_events:
                self.events_cache = self.events_cache[:self.max_events]
            
            # Rebuild display with latest first
            self.clear()
            for cached_panel in self.events_cache:
                self.write(cached_panel)
            # Keep scroll at top for latest content
            self.scroll_home()

class SystemLog(RichLog):
    """A log for system messages."""
    def add_event(self, message: str, level: str):
        color = {"INFO": "green", "WARNING": "yellow", "ERROR": "red"}.get(level, "white")
        self.write(f"[[{color}]{level}[/{color}]] {message}")

class SimulationDashboard(App):
    """The main application class for the dashboard."""

    CSS = """
    .top-section { height: 8; margin-bottom: 1; }
    .world-state { width: 1fr; height: 1fr; border: thick $primary; margin-right: 1; }
    .agent-status-section { width: 2fr; height: 1fr; border: thick $secondary; }
    .upper-section { height: 12; margin-bottom: 1; }
    .status-panel { width: 1fr; border: thick $accent; margin-right: 1; }
    .objects-panel { width: 1fr; border: thick $warning; margin-right: 1; }
    .connections-panel { width: 1fr; border: thick $primary; }
    .agent-outputs { height: 1fr; border: thick $success; min-height: 30; }
    .system-section { display: none; height: 8; border: thick $warning; margin-top: 1; }
    .agent-panel { width: 1fr; border: thick $secondary; margin: 0 1; }
    DataTable { height: 1fr; width: 100%; }
    #agent_table { width: 100%; }
    RichLog { height: 1fr; border: none; }
    """
    
    BINDINGS = [("q", "quit", "Quit"), ("s", "toggle_system", "Toggle System Log")]
    
    def __init__(self, state_file: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.state_file = state_file
        self.events_file = None
        self.last_event_position = 0
        self.agent_logs: Dict[str, AgentOutputLog] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(classes="top-section"):
                yield WorldStatePanel(classes="world-state")
                with Vertical(classes="agent-status-section"):
                    yield Label("👥 Agent Status Overview")
                    yield AgentStatusTable(id="agent_table")
            with Horizontal(classes="upper-section"):
                with Vertical(classes="status-panel"):
                    yield Label("🕐 Time & Location Status")
                    yield DetailedStatusPanel(id="status_panel")
                    yield Label("⏳ Queued Tasks")
                    yield DataTable(id="tasks_table")
                with Vertical(classes="objects-panel"):
                    yield Label("📦 Objects & NPCs")
                    yield DataTable(id="objects_table")
                with Vertical(classes="connections-panel"):
                    yield Label("🚪 Connections")
                    yield DataTable(id="connections_table")
            with Vertical(classes="agent-outputs"):
                yield Label("🤖 Agent Outputs")
                with TabbedContent(id="agent_tabs"):
                    with TabPane("👥 Primary Agents", id="primary_tab"):
                        with Horizontal():
                            with Vertical(classes="agent-panel"):
                                yield Label("💭 Simulacra (Thoughts & Intents)")
                                yield AgentOutputLog(id="sim_log")
                            with Vertical(classes="agent-panel"):
                                yield Label("📖 Narration (Story Generation)")
                                yield AgentOutputLog(id="narrator_log")
                    with TabPane("⚙️ World Engine", id="world_engine_tab"):
                        yield AgentOutputLog(id="world_log")
            yield SystemLog(id="system_log", classes="system-section")
        yield Footer()
    
    def on_ready(self) -> None:
        self.agent_logs["Simulacra"] = self.query_one("#sim_log", AgentOutputLog)
        self.agent_logs["WorldEngine"] = self.query_one("#world_log", AgentOutputLog)
        self.agent_logs["Narrator"] = self.query_one("#narrator_log", AgentOutputLog)
        
        # Initialize tasks table
        tasks_table = self.query_one("#tasks_table", DataTable)
        tasks_table.clear(columns=True)
        tasks_table.add_column("Task", key="task", width=15)
        tasks_table.add_column("Status", key="task_status", width=10)
        
        # Initialize objects table
        objects_table = self.query_one("#objects_table", DataTable)
        objects_table.clear(columns=True)
        objects_table.add_column("Name", key="obj_name")
        objects_table.add_column("Description", key="obj_desc")
        
        # Initialize connections table
        connections_table = self.query_one("#connections_table", DataTable)
        connections_table.clear(columns=True)
        connections_table.add_column("To", key="conn_to")
        connections_table.add_column("Description", key="conn_desc")
        
        self.set_interval(0.5, self.update_display)
        self.set_interval(0.5, self.read_new_events)

    def read_new_events(self) -> None:
        """Read new events from the JSONL events file and route them to appropriate panels."""
        if not self.events_file:
            # Find the latest events file
            events_pattern = "logs/events/events_latest_*.jsonl"
            events_files = glob.glob(events_pattern)
            if events_files:
                self.events_file = max(events_files, key=lambda f: Path(f).stat().st_mtime)
        
        if not self.events_file or not Path(self.events_file).exists():
            return
            
        try:
            with open(self.events_file, 'r') as f:
                # Seek to last known position
                f.seek(self.last_event_position)
                new_lines = f.readlines()
                self.last_event_position = f.tell()
                
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    event_data = json.loads(line)
                    self.route_event_to_panel(event_data)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.query_one(SystemLog).add_event(f"Error reading events: {e}", "ERROR")

    def route_event_to_panel(self, event_data: Dict[str, Any]) -> None:
        """Route an event to the appropriate panel based on event type."""
        event_type = event_data.get("event_type")
        
        if event_type in ['monologue', 'intent']:
            panel = self.agent_logs.get("Simulacra")
        elif event_type == 'resolution':
            panel = self.agent_logs.get("WorldEngine")
        elif event_type == 'narrative':
            panel = self.agent_logs.get("Narrator")
        else:
            # Unknown event type, send to system log
            self.query_one(SystemLog).add_event(f"Unknown event: {event_data}", "INFO")
            return
            
        if panel:
            panel.add_event(event_data)

    def action_toggle_system(self) -> None:
        self.query_one(SystemLog).display = not self.query_one(SystemLog).display
    
    def update_live_simulation_data(self, state_data: dict, event_bus_size: int = 0, narration_queue_size: int = 0):
        """Update dashboard with live simulation data from the running simulation."""
        self.simulation_state_ref = state_data
        # Force immediate update with live data
        self.update_display()

    def update_display(self) -> None:
        # Use live simulation state if available, otherwise fall back to file
        if hasattr(self, 'simulation_state_ref') and self.simulation_state_ref:
            state_data = self.simulation_state_ref
        else:
            # Fallback to file reading if no live state available
            if not self.state_file:
                state_dir = Path("data/states")
                if state_dir.exists():
                    state_files = list(state_dir.glob("simulation_state_*.json"))
                    if state_files:
                        self.state_file = str(max(state_files, key=lambda p: p.stat().st_mtime))
            
            if self.state_file and Path(self.state_file).exists():
                try:
                    with open(self.state_file, 'r') as f:
                        state_data = json.load(f)
                except Exception as e:
                    self.query_one(SystemLog).add_event(f"Failed to load state: {e}", "ERROR")
                    return
            else:
                return
        
        # Update all panels with state data
        try:
            self.query_one(WorldStatePanel).update_state(state_data)
            self.query_one(AgentStatusTable).update_state(state_data)
            self.query_one(DetailedStatusPanel).update_state(state_data)
            self._update_tasks_table(state_data)
            self._update_objects_table(state_data)
            self._update_connections_table(state_data)
        except Exception as e:
            self.query_one(SystemLog).add_event(f"Error updating display: {e}", "ERROR")
    
    def _update_tasks_table(self, state_data: Dict[str, Any]):
        """Update the queued tasks table with pending simulation tasks."""
        tasks_table = self.query_one("#tasks_table", DataTable)
        tasks_table.clear()
        
        simulacra = state_data.get('simulacra_profiles', {})
        active_sim_ids = state_data.get('active_simulacra_ids', [])
        
        # Show current agent tasks
        for sim_id in active_sim_ids:
            sim_data = simulacra.get(sim_id, {})
            name = sim_data.get('persona_details', {}).get('Name', sim_id)
            status = sim_data.get('status', 'Unknown')
            
            if status == "busy":
                action_desc = sim_data.get('current_action_description', 'Unknown action')[:15]
                tasks_table.add_row(f"{name[:10]}", "Active")
            elif status == "thinking":
                tasks_table.add_row(f"{name[:10]}", "Planning")
        
        # Add placeholder for future queue items
        if len(active_sim_ids) == 0:
            tasks_table.add_row("No tasks", "Idle")
    
    def _update_objects_table(self, state_data: Dict[str, Any]):
        objects_table = self.query_one("#objects_table", DataTable)
        objects_table.clear()
        
        simulacra = state_data.get('simulacra_profiles', {})
        active_sim_ids = state_data.get('active_simulacra_ids', [])
        
        if active_sim_ids and simulacra:
            primary_sim = simulacra.get(active_sim_ids[0], {})
            location_id = primary_sim.get('current_location')
            if location_id:
                location_details = state_data.get('current_world_state', {}).get('location_details', {})
                location_info = location_details.get(location_id, {})
                
                # Add objects
                for obj in location_info.get('ephemeral_objects', [])[:5]:
                    name = obj.get('name', '?')
                    desc = obj.get('description', '')
                    objects_table.add_row(name, desc)
                
                # Add NPCs
                for npc in location_info.get('ephemeral_npcs', [])[:3]:
                    name = npc.get('name', '?')
                    desc = npc.get('description', '')
                    objects_table.add_row(f"👤 {name}", desc)
    
    def _update_connections_table(self, state_data: Dict[str, Any]):
        connections_table = self.query_one("#connections_table", DataTable)
        connections_table.clear()
        
        simulacra = state_data.get('simulacra_profiles', {})
        active_sim_ids = state_data.get('active_simulacra_ids', [])
        
        if active_sim_ids and simulacra:
            primary_sim = simulacra.get(active_sim_ids[0], {})
            location_id = primary_sim.get('current_location')
            if location_id:
                location_details = state_data.get('current_world_state', {}).get('location_details', {})
                location_info = location_details.get(location_id, {})
                
                for conn in location_info.get('connected_locations', [])[:5]:
                    to_loc = conn.get('to_location_id_hint', '?')
                    desc = conn.get('description', '')
                    connections_table.add_row(to_loc, desc)



def run_dashboard(state_file: Optional[str] = None):
    app = SimulationDashboard(state_file=state_file)
    app.run()

if __name__ == "__main__":
    run_dashboard()
