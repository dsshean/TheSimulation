import socket
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.table import Table
from rich import box

# Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
BUFFER_SIZE = 8192

console = Console()
history = []  # Store command/response history

class SimulationClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.buffer = b""
    
    def connect(self) -> bool:
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(0.1)  # Short timeout for non-blocking reads
            self.connected = True
            console.print(f"[bold green]Connected to simulation at {self.host}:{self.port}[/]")
            return True
        except Exception as e:
            console.print(f"[bold red]Connection failed: {e}[/]")
            self.connected = False
            return False
    
    def send_command(self, command_obj: Dict) -> Optional[Dict]:
        if not self.connected:
            if not self.connect():
                return None
                
        try:
            message = json.dumps(command_obj) + "\n"
            self.socket.sendall(message.encode('utf-8'))
            
            # Wait for response with timeout
            response_data = b""
            start_time = time.time()
            
            while True:
                try:
                    chunk = self.socket.recv(BUFFER_SIZE)
                    if not chunk:
                        break
                    response_data += chunk
                    if b'\n' in response_data:
                        break
                except socket.timeout:
                    if time.time() - start_time > 10:  # 10-second overall timeout
                        console.print("[yellow]Response timed out[/]")
                        break
                    continue
                except Exception as e:
                    console.print(f"[red]Error receiving data: {e}[/]")
                    break
            
            if response_data:
                response_str = response_data.decode('utf-8').strip()
                try:
                    response = json.loads(response_str)
                    return response
                except json.JSONDecodeError:
                    console.print(f"[red]Received invalid JSON response:[/]")
                    console.print(Syntax(response_str, "json", theme="monokai", line_numbers=True))
                    return {"success": False, "message": "Invalid JSON response", "raw": response_str}
            return {"success": False, "message": "No response"}
            
        except Exception as e:
            console.print(f"[bold red]Error sending command: {e}[/]")
            self.connected = False
            return None
    
    def close(self):
        if self.socket:
            try:
                self.socket.close()
                console.print("[bold yellow]Connection closed[/]")
            except:
                pass
            self.socket = None
            self.connected = False

def display_response(command: Dict, response: Dict):
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Record in history
    history.append((command, response, timestamp))
    
    # Display different panels based on command type
    command_type = command.get("command", "unknown")
    
    if command_type == "narrate":
        narrative_text = command.get("text", "")
        console.print(Panel(
            f"[bold cyan]Narrative Injection:[/]\n\n[italic]{narrative_text}[/]\n\n[green]Response:[/] {response.get('message', 'No message')}",
            title=f"[{timestamp}] Narrative",
            border_style="cyan",
            expand=False
        ))
    
    elif command_type == "inject_event":
        agent_id = command.get("agent_id", "unknown")
        description = command.get("description", "")
        console.print(Panel(
            f"[bold magenta]Agent:[/] {agent_id}\n\n[bold cyan]Event Injected:[/]\n[italic]{description}[/]\n\n[green]Response:[/] {response.get('message', 'No message')}",
            title=f"[{timestamp}] Agent Event",
            border_style="magenta",
            expand=False
        ))
    
    elif command_type == "world_info":
        category = command.get("category", "unknown")
        info = command.get("info", "")
        console.print(Panel(
            f"[bold yellow]Category:[/] {category}\n\n[bold cyan]Info Updated:[/]\n[italic]{info}[/]\n\n[green]Response:[/] {response.get('message', 'No message')}",
            title=f"[{timestamp}] World Info Update",
            border_style="yellow",
            expand=False
        ))
    
    elif command_type == "fix_json":
        if response.get("fixed_json"):
            console.print(Panel(
                Syntax(response.get("fixed_json", "{}"), "json", theme="monokai", line_numbers=True),
                title=f"[{timestamp}] Fixed JSON",
                border_style="green",
                expand=False
            ))
        else:
            console.print(Panel(
                f"[bold red]JSON Fix Failed:[/]\n\n[green]Response:[/] {response.get('message', 'No message')}",
                title=f"[{timestamp}] JSON Fix",
                border_style="red",
                expand=False
            ))
    elif command_type == "teleport_agent":
        agent_id = command.get("agent_id", "unknown")
        new_location_id = command.get("new_location_id", "unknown")
        location_description = command.get("location_description", "N/A")
        console.print(Panel(
            f"[bold yellow]Agent Teleported:[/]\nAgent ID: {agent_id}\nNew Location: {new_location_id}\nDescription: {location_description}\n\n[green]Response:[/] {response.get('message', 'No message')}",
            title=f"[{timestamp}] Teleport Agent",
            border_style="yellow",
            expand=False
        ))


    
    else:
        # Generic response display
        console.print(Panel(
            Syntax(json.dumps(response, indent=2), "json", theme="monokai"),
            title=f"[{timestamp}] Response",
            border_style="blue",
            expand=False
        ))

def show_history():
    if not history:
        console.print("[yellow]No command history yet[/]")
        return
    
    table = Table(title="Command History", box=box.ROUNDED)
    table.add_column("#", style="cyan")
    table.add_column("Time", style="magenta")
    table.add_column("Command", style="green")
    table.add_column("Status", style="yellow")
    
    for i, (cmd, resp, time) in enumerate(history, 1):
        cmd_type = cmd.get("command", "unknown")
        cmd_preview = ""
        if cmd_type == "narrate":
            cmd_preview = f"narrate: {cmd.get('text', '')[:30]}..."
        elif cmd_type == "inject_event":
            cmd_preview = f"event to {cmd.get('agent_id', 'unknown')}"
        elif cmd_type == "world_info":
            cmd_preview = f"{cmd.get('category', 'unknown')} update"
        elif cmd_type == "fix_json":
            cmd_preview = "JSON fix"
        elif cmd_type == "teleport_agent":
            desc_preview = f" w/ desc: {cmd.get('location_description', '')[:15]}..." if cmd.get('location_description') and cmd.get('location_description') != "No specific description provided" else ""
            cmd_preview = f"teleport {cmd.get('agent_id', 'N/A')} to {cmd.get('new_location_id', 'N/A')}{desc_preview}"
            
        status = "[green]Success[/]" if resp.get("success") else "[red]Failed[/]"
        table.add_row(str(i), time, cmd_preview, status)
    
    console.print(table)

def _fetch_and_parse_agent_details(client: SimulationClient) -> List[Dict[str, str]]:
    """
    Fetches simulation state from the server and extracts agent IDs and their statuses.
    Returns a list of dictionaries, each containing 'id' and 'status', 
    or an empty list if an error occurs or no agents are found.
    """
    command = {"command": "get_state"}
    response = client.send_command(command)
    
    # Debug: Print full response
    console.print("[yellow]DEBUG: Full response from server:[/]")
    console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
    
    if not response or not response.get("success", False):
        console.print(f"[red]Failed to get simulation state.[/red]")
        return []
    
    agent_ids = []
    agent_details_list = []

    # Primary way to get agent_ids and their details
    sim_profiles = response.get("data", {}).get("state_summary", {}).get("simulacra_profiles", {})
    if sim_profiles and isinstance(sim_profiles, dict):
        for agent_id, details in sim_profiles.items():
            agent_ids.append(agent_id) # Keep a list of IDs for selection logic later
            agent_details_list.append({
                "id": agent_id,
                "status": details.get("status", "unknown")
            })
    
    # Fallback if primary way fails, try to get just IDs from various known paths
    if not agent_ids:
        if response.get("data", {}).get("agent_ids"):
            agent_ids = response["data"]["agent_ids"]
        elif response.get("data", {}).get("data", {}).get("agent_ids"):
            agent_ids = response["data"]["data"]["agent_ids"]
        elif response.get("state", {}).get("simulacra_profiles"):
            agent_ids = list(response["state"]["simulacra_profiles"].keys())
        elif response.get("simulacra_profiles"):
            agent_ids = list(response["simulacra_profiles"].keys())
        elif response.get("data", {}).get("simulacra"):
            agent_ids = list(response["data"]["simulacra"].keys())
        elif response.get("agents"):
            agent_ids = response["agents"]
        
        if not agent_ids: # Try broader search if still no IDs
            for key, value in response.items():
                if isinstance(value, list) and value and isinstance(value[0], str) and value[0].startswith("sim_"):
                    agent_ids = value
                    console.print(f"[yellow]Found potential agent IDs in field '{key}'[/]")
                    break
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], str) and subvalue[0].startswith("sim_"):
                            agent_ids = subvalue
                            console.print(f"[yellow]Found potential agent IDs in field '{key}.{subkey}'[/]")
                            break
        
        if not agent_ids: # Regex scan as last resort for IDs
            console.print("[yellow]Scanning response for agent IDs...[/]")
            response_str = json.dumps(response)
            import re
            potential_ids = re.findall(r'"(sim_[a-zA-Z0-9_]+)"', response_str)
            if potential_ids:
                agent_ids = list(set(potential_ids))
                console.print(f"[yellow]Found {len(agent_ids)} potential agent IDs by regex search[/]")

        # If we found IDs through fallback, create basic detail entries
        if agent_ids and not agent_details_list:
            for agent_id in agent_ids:
                agent_details_list.append({"id": agent_id, "status": "unknown (fallback)"})

    return agent_details_list

def _display_active_agents(client: SimulationClient) -> List[Dict[str, str]]:
    """
    Fetches and displays the list of active simulacra.
    Returns the list of agent details (dict with 'id' and 'status'), or an empty list if none are found.
    """
    console.rule("[bold cyan]Active Simulacra[/]")
    agent_details_list = _fetch_and_parse_agent_details(client)
    if not agent_details_list:
        console.print("[red]No active simulacra found in the simulation.[/red]")
        return []
    
    console.print("[bold green]Currently active simulacra:[/]")
    for i, agent_detail in enumerate(agent_details_list, 1):
        console.print(f"{i}. ID: {agent_detail['id']} - Status: {agent_detail['status']}")
    return agent_details_list

def _select_agent_from_server_state(client: SimulationClient, action_description: str) -> Optional[str]:
    """
    Fetches simulation state, lists agent IDs, and prompts user to select one.
    """
    agent_details_list = _fetch_and_parse_agent_details(client)
    
    if not agent_details_list:
        console.print(f"[red]No active simulacra found for {action_description}[/]")
        use_manual = Prompt.ask(f"[yellow]Enter agent ID manually for {action_description}?[/]", choices=["y", "n"], default="y")
        if use_manual.lower() == "y":
            agent_id = Prompt.ask("Enter agent ID")
            return agent_id
        else:
            return None 
    
    console.print("[bold cyan]Available simulacra:[/]")
    if not agent_details_list: # Should be redundant due to check above, but safe
        console.print("[red]No agents available to select.[/red]")
        return None
        
    for i, agent_detail in enumerate(agent_details_list, 1):
        console.print(f"{i}. ID: {agent_detail['id']} - Status: {agent_detail['status']}")
    
    agent_idx = IntPrompt.ask(f"Select agent for {action_description}", choices=[str(i) for i in range(1, len(agent_details_list)+1)])
    return agent_details_list[int(agent_idx)-1]['id'] # Return only the ID

def interactive_session(client: SimulationClient):
    console.rule("[bold blue]Interactive Mode with Simulacra[/]")
    
    selected_agent = _select_agent_from_server_state(client, "interactive mode")
    if not selected_agent:
        return
    
    # Start interaction mode
    start_cmd = {"command": "start_interaction_mode", "agent_id": selected_agent}
    response = client.send_command(start_cmd)
    
    if not response or not response.get("success", False):
        console.print("[red]Failed to start interaction mode[/]")
        return
    
    console.print(f"[bold green]Interactive mode started with {selected_agent}[/]")
    console.print("[italic]Type one of these commands: [/]")
    console.print("- [bold cyan]text:[/] [cyan]<message>[/] - Send a text message")
    console.print("- [bold cyan]call:[/] [cyan]<message>[/] - Make a phone call")
    console.print("- [bold cyan]voice:[/] [cyan]<message>[/] - Speak directly (voice of god)")
    console.print("- [bold cyan]event:[/] [cyan]<description>[/] - Custom event")
    console.print("- [bold cyan]doorbell:[/] [cyan]<details>[/] - Someone at the door")
    console.print("- [bold cyan]noise:[/] [cyan]<description>[/] - Make a noise")
    console.print("- [bold cyan]exit[/] - End interaction mode")
    
    last_check_time = 0.0
    stop_polling = threading.Event()
    
    def poll_for_responses():
        nonlocal last_check_time
        while not stop_polling.is_set():
            time.sleep(1.0)
            try:
                poll_cmd = {
                    "command": "get_agent_responses", 
                    "since_timestamp": last_check_time,
                    "agent_id": selected_agent
                }
                poll_response = client.send_command(poll_cmd)
                
                if poll_response and poll_response.get("success", False):
                    # Access responses from response.get("data", {}).get("responses", [])
                    responses = poll_response.get("data", {}).get("responses", [])
                    if responses:
                        for resp in responses:
                            content = resp.get("content", "")
                            if "[InteractionMode]" in content:
                                continue
                            if selected_agent in content:
                                if "decides to" in content:
                                    parts = content.split("decides to", 1)
                                    action_part = parts[1].strip() if len(parts) > 1 else content
                                    console.print(f"[bold magenta]{selected_agent}:[/] [italic cyan]{action_part}[/]")
                                elif "says" in content and selected_agent in content.split("says")[0]:
                                    parts = content.split("says", 1)
                                    speech_part = parts[1].strip() if len(parts) > 1 else content
                                    console.print(f"[bold magenta]{selected_agent}:[/] [italic cyan]{speech_part}[/]")
                                elif ":" in content and content.index(":") > content.index(selected_agent):
                                    parts = content.split(":", 1)
                                    speech_part = parts[1].strip() if len(parts) > 1 else content
                                    console.print(f"[bold magenta]{selected_agent}:[/] [italic cyan]{speech_part}[/]")
                                elif "responds" in content and selected_agent in content.split("responds")[0]:
                                    parts = content.split("responds", 1)
                                    second_part = parts[1].strip() if len(parts) > 1 else content
                                    if '"' in second_part:
                                        quote_parts = second_part.split('"')
                                        speech_part = quote_parts[1] if len(quote_parts) > 1 else second_part
                                    else:
                                        speech_part = second_part
                                    console.print(f"[bold magenta]{selected_agent}:[/] [italic cyan]{speech_part}[/]")
                                else:
                                    console.print(f"[dim]{content}[/]")
                        # Update timestamp from response.get("data", {}).get("current_time", ...)
                        last_check_time = poll_response.get("data", {}).get("current_time", last_check_time)
            except Exception as e:
                print(f"Error polling for responses: {e}")
    
    polling_thread = threading.Thread(target=poll_for_responses)
    polling_thread.daemon = True
    polling_thread.start()
    
    try:
        while True:
            user_input = Prompt.ask("[bold green]You[/]")
            if user_input.lower() == 'exit':
                break
            event_type = "text_message"
            content = user_input
            if user_input.startswith("text:"):
                event_type = "text_message"
                content = user_input[5:].strip()
            elif user_input.startswith("call:"):
                event_type = "phone_call"
                content = user_input[5:].strip()
            elif user_input.startswith("voice:"):
                event_type = "voice"
                content = user_input[6:].strip()
            elif user_input.startswith("event:"):
                event_type = "custom"
                content = user_input[6:].strip()
            elif user_input.startswith("doorbell:"):
                event_type = "doorbell"
                content = user_input[9:].strip()
            elif user_input.startswith("noise:"):
                event_type = "noise"
                content = user_input[6:].strip()
            event_cmd = {
                "command": "interaction_event", 
                "agent_id": selected_agent,
                "event_type": event_type,
                "content": content
            }
            event_response = client.send_command(event_cmd)
            if not event_response or not event_response.get("success", False):
                console.print("[red]Failed to send interaction event[/]")
    finally:
        stop_polling.set()
        console.print("[yellow]Ending interactive mode...[/]")
        end_cmd = {"command": "end_interaction_mode", "agent_id": selected_agent}
        client.send_command(end_cmd)
        console.print(f"[green]{selected_agent} has returned to the simulation[/]")

def main_menu() -> bool:
    console.rule("[bold green]TheSimulation Control Panel[/]")
    console.print("[bold cyan]Available Commands:[/]")
    console.print("1. Inject narrative")
    console.print("2. Send event to agent")
    console.print("3. Update world info")
    console.print("4. Fix broken JSON")
    console.print("5. View command history")
    console.print("6. Interactive mode with simulacra")
    console.print("7. Teleport Simulacrum")
    console.print("8. List Active Simulacra") # New option
    console.print("0. Exit") 
    
    choice = IntPrompt.ask("Select command", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"], default=1)
    
    return process_menu_choice(choice)

def process_menu_choice(choice: int) -> bool:
    client = SimulationClient(SERVER_HOST, SERVER_PORT)
    
    try:
        if choice == 0:
            return False
        
        elif choice == 1:
            console.rule("[bold cyan]Inject Narrative[/]")
            text = Prompt.ask("Enter narrative text")
            if text:
                command = {"command": "narrate", "text": text}
                response = client.send_command(command)
                if response:
                    display_response(command, response)
        
        elif choice == 2:
            console.rule("[bold magenta]Send Event to Agent[/]")
            agent_id = Prompt.ask("Enter agent ID (e.g., sim_xxt27r)")
            description = Prompt.ask("Enter event description")
            if agent_id and description:
                command = {"command": "inject_event", "agent_id": agent_id, "description": description}
                response = client.send_command(command)
                if response:
                    display_response(command, response)
        
        elif choice == 3:
            console.rule("[bold yellow]Update World Info[/]")
            console.print("[cyan]Categories:[/] weather, news")
            category = Prompt.ask("Enter category", choices=["weather", "news"])
            info = Prompt.ask("Enter new information")
            if category and info:
                command = {"command": "world_info", "category": category, "info": info}
                response = client.send_command(command)
                if response:
                    display_response(command, response)
        
        elif choice == 4:
            console.rule("[bold green]Fix Broken JSON[/]")
            console.print("Enter or paste the broken JSON (Ctrl+D or Ctrl+Z on empty line to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            text = "\n".join(lines)
            if text:
                command = {"command": "fix_json", "text": text}
                response = client.send_command(command)
                if response:
                    display_response(command, response)
        
        elif choice == 5:
            console.rule("[bold blue]Command History[/]")
            show_history()
        
        elif choice == 6:
            interactive_session(client)

        elif choice == 7:
            console.rule("[bold yellow]Teleport Simulacrum[/]")
            selected_agent_id = _select_agent_from_server_state(client, "teleportation")
            if selected_agent_id:
                new_location_id = Prompt.ask(f"Enter new location ID for {selected_agent_id} (e.g., Home_01, Park_Central)")
                if new_location_id.strip(): # Ensure new_location_id is not just whitespace
                    location_description = Prompt.ask(
                        f"Enter description for '{new_location_id}' (optional)",
                        default="No specific description provided.",
                        show_default=True # Show the default value in the prompt
                    )
                    command = {"command": "teleport_agent", "agent_id": selected_agent_id, "new_location_id": new_location_id.strip(), "location_description": location_description}
                    response = client.send_command(command)
                    if response:
                        display_response(command, response)
                else:
                    console.print("[yellow]Teleportation cancelled: No location ID provided.[/yellow]")
        
        elif choice == 8: # List Active Simulacra
            _display_active_agents(client)

        
    finally:
        if client:
            client.close()
    
    return True

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold green]TheSimulation[/] [bold yellow]Interactive Client[/]",
        border_style="green"
    ))
    
    running = True
    while running:
        try:
            running = main_menu()
            if running:
                console.print("\nPress Enter to continue...", end="")
                input()
                console.clear()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Exiting...[/]")
            running = False
    
    console.print("\n[bold green]Goodbye![/]")
