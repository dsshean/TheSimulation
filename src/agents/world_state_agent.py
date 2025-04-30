# src/agents/world_state_agent.py

import logging
from google.adk.agents import Agent # Agent is alias for LlmAgent
from google.adk.tools import FunctionTool
from src.config import settings
# <<< Import the NEW tool function >>>
from src.tools import world_state_tools # Import the module
from src.prompts import world_state_instructions # Phase 1 instructions
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# --- Define NEW Instruction Template ---
# This replaces the one in src/prompts/world_state_instructions.py for this agent
WORLD_STATE_AGENT_INSTRUCTION_V2 = """
You are the World State Manager for the simulation, responsible for the start-of-turn update.

**Your Task (Phase 1: World State Update & Sync):**

1.  **Process Arrivals:** Call the `process_scheduled_events` tool first to handle any messages or events arriving this turn and update recipient statuses.
2.  **Sync Real-World Details:** Call the `get_setting_details` tool for the specified primary location (e.g., 'Asheville, NC') to fetch and store current real-world context (weather, news).
3.  **Update World Time:** Call the `update_and_get_world_state` tool to advance the simulation's world time based on the previous turn.
4.  **Respond:** Your final response should be a simple confirmation message.

**Example Interaction Flow:**

*   **Input Trigger:** "Perform the start-of-turn world state update for turn 5. Primary location is 'Asheville, NC'."
*   **Your Action 1:** Call `process_scheduled_events`.
*   **Your Action 2:** Call `get_setting_details` with `location='Asheville, NC'`.
*   **Your Action 3:** Call `update_and_get_world_state`.
*   **Your Final Output:** "World state processed, synced, and time updated."

**CRITICAL: You MUST use the specified tools (`process_scheduled_events`, `get_setting_details`, `update_and_get_world_state`) in the correct order to perform this task. Your final text output should just be a confirmation.**
"""


try:
    world_state_agent = Agent(
        name="world_state_agent",
        model=settings.MODEL_GEMINI_PRO,
        description=(
            "Manages the start-of-turn world state update. In Phase 1, it processes scheduled events, "
            "fetches real-world details (weather, news), and advances the simulation time." # Updated description
        ),
        # <<< Use the NEW instruction template >>>
        instruction=WORLD_STATE_AGENT_INSTRUCTION_V2,
        tools=[
            # <<< ADD the new tool FIRST >>>
            FunctionTool(func=world_state_tools.process_scheduled_events),
            # --- Keep existing tools ---
            FunctionTool(func=world_state_tools.get_setting_details),
            FunctionTool(func=world_state_tools.update_and_get_world_state),
        ],
        output_key="world_state_agent_confirmation", # Captures the confirmation text
    )
    console.print(f"Agent '[bold blue]{world_state_agent.name}[/bold blue]' defined (Phase 1 State Updater Role - V2).") # Updated Role Desc
    logger.info(f"WorldStateAgent initialized with tools: {[t.func.__name__ for t in world_state_agent.tools]}")

except Exception as e:
    console.print(f"[bold red]Error creating world_state_agent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    world_state_agent = None
