from google.adk.agents import Agent # Agent is alias for LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types # Import types for Content/Part
from google.generativeai.client import get_default_model_client # Needed to get model client
from src.config import settings
from src.prompts import world_engine_instructions
from src.tools.world_engine_tools import save_single_validation_result
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

# --- Factory function remains largely the same, but uses the simplified tool list ---
def create_world_engine_validator(sim_id: str) -> Agent:
    """
    Factory function to create a specialized World Engine agent instance
    for validating a SINGLE simulacrum's action.
    Relies on context being available implicitly for evaluation.

    Args:
        sim_id: The ID of the simulacrum this agent instance will validate.
        model_name: The name of the language model to use.

    Returns:
        An configured Agent (LLMAgent) instance for single validation.
    """
    if not sim_id:
        raise ValueError("sim_id cannot be empty when creating a world engine validator.")

    try:
        specific_instruction = world_engine_instructions.WORLD_ENGINE_INSTRUCTION.format(target_simulacra_id=sim_id)
        # Create the Agent (LLMAgent) instance
        agent = Agent(
            name=f"WorldEngineValidator_{sim_id}",
            model=settings.MODEL_GEMINI_PRO,
            # --- Use system_instruction for consistency ---
            instruction=specific_instruction,
            # instruction=specific_instruction, # Comment out or remove
            # --- End modification ---
            description=f"World Engine Validator for {sim_id}.",
            tools=[FunctionTool(save_single_validation_result)],
        )
        logger.debug(f"Created WorldEngineValidator instance for {sim_id}")
        return agent

    except KeyError as ke: # Catch KeyError specifically
        logger.exception(f"KeyError during .format() for {sim_id}: {ke}. Placeholder likely missing in instruction string.")
        console.print(f"[bold red]KeyError formatting instruction for {sim_id}: '{ke}'. Check world_engine_instructions.py template variables.[/bold red]")
        # Also print the raw string again to help debug
        print("--- Raw instruction string that caused KeyError ---")
        print(world_engine_instructions.WORLD_ENGINE_INSTRUCTION)
        print("--- End raw string ---")
        return None # Or raise
    except Exception as e:
        logger.exception(f"Error creating world_engine_validator for {sim_id}: {e}")
        # console.print("Factory 'create_world_engine_validator' defined in world_engine.py (Simplified Tools).") # Redundant info
        console.print(f"[bold red]Error creating world_engine_validator for {sim_id}:[/bold red] {e}")
        return None # Return None on other errors as well