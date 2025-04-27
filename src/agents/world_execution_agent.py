import logging
from google.adk.agents import Agent
from src.config import settings
from src.prompts import world_execution_instructions
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

world_execution_agent = None
try:
    world_execution_agent = Agent(
        name="WorldExecutionAgent",
        model=settings.MODEL_GEMINI_PRO, # Or another suitable model
        description=(
            "Processes approved 'move' actions, infers details like specific locations "
            "and travel time, and generates a descriptive narrative for the execution phase (Phase 4b)."
        ),
        instruction=world_execution_instructions.WORLD_EXECUTION_INSTRUCTION,
        tools=[],
        output_key="execution_results_json", # Key to store the final JSON output
    )
    console.print(f"Agent '[bold magenta]{world_execution_agent.name}[/bold magenta]' defined (Movement Execution Role).")

except Exception as e:
    console.print(f"[bold red]Error creating WorldExecutionAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    world_execution_agent = None # Ensure it's None on error
