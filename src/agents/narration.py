# src/agents/narration.py
from google.adk.agents import Agent
from src.config import settings
from src.tools import narration_tools
from src.prompts import narration_instructions # Import instructions
from rich.console import Console

console = Console()

def create_agent(simulacra_agent_instance, world_engine_agent_instance, npc_agent_instance):
    """Factory function to create the Narration agent."""
    if not all([simulacra_agent_instance, world_engine_agent_instance, npc_agent_instance]):
         console.print("[bold red]Cannot create Narration Agent because one or more sub-agent instances are missing.[/bold red]")
         return None
    try:
        agent = Agent(
            name="narration_agent",
            model=settings.MODEL_GEMINI_PRO,
            description="The Master Narrator. Orchestrates the simulation turn-by-turn, mediating all interactions.",
            instruction=narration_instructions.NARRATION_AGENT_INSTRUCTION, # Use imported instruction
            tools=[
                narration_tools.set_simulacra_daily_goal,
                narration_tools.get_current_simulation_state_summary
            ],
            sub_agents=[
                simulacra_agent_instance,
                world_engine_agent_instance,
                npc_agent_instance
            ],
            output_key="last_narration"
        )
        console.print(f"Agent '[bold yellow]{agent.name}[/bold yellow]' created with sub-agents.")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating narration_agent:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None