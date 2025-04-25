# src/agents/npc.py (Interaction Resolver Agent)

from google.adk.agents import Agent # Alias for LlmAgent
from src.config import settings
# Import the NEW tools
from src.tools import npc_tools
# Import the NEW instructions
from src.prompts import npc_instructions
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

# --- Define the NPC Agent Instance ---
npc_agent = None
try:
    # This agent resolves talk/interact actions after validation
    npc_agent = Agent(
        name="NpcInteractionAgent", # More descriptive name
        # Model needs good reasoning & dialogue generation
        model=settings.MODEL_GEMINI_PRO,
        description=(
            "Resolves the outcome of validated social (talk) and object (interact) "
            "actions based on character states, personas, and world context."
        ),
        instruction=npc_instructions.NPC_AGENT_INSTRUCTION,
        tools=[
            npc_tools.get_validated_interactions,
            npc_tools.update_interaction_results,
        ],
        # Output not strictly needed, as results are written to state by the tool
        # output_key="npc_agent_confirmation",
    )
    console.print(f"Agent '[bold green]{npc_agent.name}[/bold green]' defined (Interaction Resolver Role).")
    logger.info(f"NpcInteractionAgent initialized with tools: {[t.__name__ for t in npc_agent.tools]}")

except Exception as e:
    console.print(f"[bold red]Error creating NpcInteractionAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)