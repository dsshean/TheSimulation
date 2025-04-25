# src/agents/simulacra.py
from google.adk.agents import Agent
from google.adk.tools import FunctionTool # Import FunctionTool
from src.config import settings
from src.tools import simulacra_tools
from src.prompts import simulacra_instructions # Import instructions
from rich.console import Console

console = Console()

def create_agent():
    """Factory function to create the Simulacra agent."""
    try:
        agent = Agent(
            name="simulacra", # Base name, will be overridden in main.py
            model=settings.MODEL_GEMINI_PRO,
            description="Reflects internally and then decides the player character's action intent (move or talk).", # Updated description
            instruction=simulacra_instructions.SIMULACRA_AGENT_INSTRUCTION,
            tools=[
                # --- Correct FunctionTool usage ---
                FunctionTool(simulacra_tools.generate_internal_monologue),
                FunctionTool(simulacra_tools.attempt_move_to),
                FunctionTool(simulacra_tools.attempt_talk_to),
                FunctionTool(simulacra_tools.check_self_status)
                # --- End Correct Usage ---
            ],
            # output_key=None # Ensure no default output key if not needed
        )
        console.print(f"Agent '[bold blue]{agent.name}[/bold blue]' template created (with reflection tool).")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating simulacra_agent template:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None # Return None on error