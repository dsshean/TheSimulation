from google.adk.agents import Agent # Alias for LlmAgent
from src.config import settings
from src.tools import npc_tools
from src.prompts import npc_instructions
from google.adk.tools import FunctionTool
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

npc_agent = None
try:
    npc_agent = Agent(
        name="NpcInteractionAgent", # More descriptive name
        model=settings.MODEL_GEMINI_PRO,
        description=(
            "Resolves the outcome of validated social (talk) and object (interact) "
            "actions based on character states, personas, and world context."
        ),
        instruction=npc_instructions.NPC_AGENT_INSTRUCTION,
        tools=[
            FunctionTool(func=npc_tools.get_validated_interactions),
            FunctionTool(func=npc_tools.update_interaction_results),
        ],
    )
    console.print(f"Agent '[bold green]{npc_agent.name}[/bold green]' defined (Interaction Resolver Role).")
    logger.info(f"NpcInteractionAgent initialized with tools: {[t.func.__name__ for t in npc_agent.tools]}")

except Exception as e:
    console.print(f"[bold red]Error creating NpcInteractionAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)