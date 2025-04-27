from google.adk.agents import Agent # Agent is alias for LlmAgent
from src.config import settings
from src.prompts import world_engine_instructions
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

world_engine_agent = None
try:
    world_engine_agent = Agent( # Using Agent (alias for LlmAgent)
        name="WorldEngineAgent", # Keep the clear name
        model=settings.MODEL_GEMINI_PRO, # Or another suitable model
        description=(
            "Acts as the simulation's physics and rules engine. Validates "
            "proposed actions based on world state context and rules, "
            "outputting a validation JSON."
        ),
        instruction=world_engine_instructions.WORLD_ENGINE_INSTRUCTION,
        tools=[],
        output_key="validation_result", # Save the validation output dictionary
    )
    console.print(f"Agent '[bold blue]{world_engine_agent.name}[/bold blue]' defined (Validator Role).") # Correct Role Desc

except Exception as e:
    console.print(f"[bold red]Error creating world_engine_agent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    world_engine_agent = None