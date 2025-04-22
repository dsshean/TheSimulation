# src/agents/npc.py
from google.adk.agents import Agent
from src.config import settings
from src.tools import npc_tools
from src.prompts import npc_instructions # Import instructions
from rich.console import Console

console = Console()

def create_agent():
    """Factory function to create the NPC agent."""
    try:
        agent = Agent(
            name="npc_agent",
            model=settings.MODEL_GEMINI_PRO,
            description="Generates NPC dialogue responses when instructed by the Narrator.",
            instruction=npc_instructions.NPC_AGENT_INSTRUCTION, # Use imported instruction
            tools=[npc_tools.generate_npc_response],
        )
        console.print(f"Agent '[bold magenta]{agent.name}[/bold magenta]' created.")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating npc_agent:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None