# src/agents/narration.py (Using AgentTool for WE & Simulacra)
from google.adk.agents import Agent
# --- Add AgentTool Import ---
from google.adk.tools.agent_tool import AgentTool
# --- End Add Import ---
from src.config import settings
from src.tools import narration_tools
from src.prompts import narration_instructions # Use UPDATED instruction below
from rich.console import Console

console = Console()

def create_agent(simulacra_agent_instance, world_engine_agent_instance, npc_agent_instance):
    """Factory function to create the Narration agent."""
    if not all([simulacra_agent_instance, world_engine_agent_instance, npc_agent_instance]):
         console.print("[bold red]Cannot create Narration Agent: Agent instances missing.[/bold red]")
         return None
    try:
        # --- Define AgentTools ---
        world_engine_tool = AgentTool(agent=world_engine_agent_instance)
        simulacra_tool = AgentTool(agent=simulacra_agent_instance) # Define Simulacra as Tool
        # --- End Define AgentTools ---

        agent = Agent(
            name="narration_agent",
            model=settings.MODEL_GEMINI_PRO,
            description="The Master Narrator. Orchestrates the simulation turn-by-turn, mediating all interactions.",
            instruction=narration_instructions.NARRATION_AGENT_INSTRUCTION, # Use UPDATED instruction below
            tools=[
                narration_tools.set_simulacra_daily_goal,
                narration_tools.get_current_simulation_state_summary,
                narration_tools.get_last_simulacra_action_details,
                world_engine_tool, # World Engine tool
                simulacra_tool     # Simulacra tool
            ],
            sub_agents=[
                # Only NPC remains as sub_agent for explicit delegation later
                npc_agent_instance
            ],
            output_key="last_narration"
        )

        console.print(f"Agent '[bold yellow]{agent.name}[/bold yellow]' created (WE & Simulacra as Tools).")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating narration_agent:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None