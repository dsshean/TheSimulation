# src/agents/simulacra.py
from google.adk.agents import Agent
from src.config import settings
from src.tools import simulacra_tools
from src.prompts import simulacra_instructions # Import instructions
from rich.console import Console

# Assuming tools are imported correctly from their respective modules
# Adjust imports based on your actual file structure
from src.tools.world_state_tools import (
    get_full_world_state
)
from src.prompts.world_state_instructions import WORLD_STATE_INSTRUCTION

# Define the World State Agent
world_state_agent = Agent(
    name="world_state_agent",
    model=settings.MODEL_GEMINI_PRO, # Choose an appropriate model
    instruction=WORLD_STATE_INSTRUCTION,
    description="Determines and reports the current state of the simulation world based on configuration and real-time data (if applicable).",
    tools=[
        get_full_world_state
    ],
    # output_key="current_world_state" # Optional: Automatically save output to state
)

# print("world_state_agent defined.")