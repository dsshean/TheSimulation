# src/agents/simulacra.py
from google.adk.agents import Agent
from src.config import settings
from src.tools import simulacra_tools
from src.prompts import simulacra_instructions # Import instructions
from rich.console import Console

console = Console()

def create_agent():
    """Factory function to create the Simulacra agent."""
    try:
        agent = Agent(
            name="simulacra",
            model=settings.MODEL_GEMINI_PRO,
            description="Decides the player character's action intent based on context provided by the Narrator.",
            instruction=simulacra_instructions.SIMULACRA_AGENT_INSTRUCTION, # Use imported instruction
            tools=[
                simulacra_tools.attempt_move_to,
                simulacra_tools.attempt_talk_to,
                simulacra_tools.check_self_status
            ],
        )
        console.print(f"Agent '[bold blue]{agent.name}[/bold blue]' created.")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating simulacra_agent:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None