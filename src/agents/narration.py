# src/agents/narration.py

import asyncio
import json
import logging
import os
from typing import Dict, Optional

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
# <<< Import FunctionTool and ToolContext >>>
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

# Rich Console
from rich.console import Console

# Local Imports
from src.config import settings
# Assuming narration_instructions.py exists and has NARRATION_INSTRUCTION
from src.prompts import narration_instructions
# Import BOTH tools from the module
from src.tools import narration_tools

# Setup
console = Console()
logger = logging.getLogger(__name__)
# Configure logging if running standalone
# Moved configuration inside if __name__ == "__main__" block

# --- State Keys (Define or import for test) ---
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
# Add other keys needed for mock state in the test
WORLD_STATE_KEY = "current_world_state"
TURN_INTERACTION_LOG_KEY = "turn_interaction_log"
TURN_EXECUTION_NARRATIVES_KEY = "turn_execution_narratives"
# Keys read by get_narration_context
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_MONOLOGUE_KEY_FORMAT = "last_simulacra_{}_monologue"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result"
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"


# --- Agent Definition ---
narration_agent = None
try:
    # Explicitly wrap BOTH tool functions
    get_context_tool_obj = FunctionTool(func=narration_tools.get_narration_context)
    save_narration_tool_obj = FunctionTool(func=narration_tools.save_narration)

    # --- Use actual instruction from prompts ---
    agent_instruction = getattr(narration_instructions, 'NARRATION_AGENT_INSTRUCTION', None)
    if not agent_instruction:
        logger.error("CRITICAL: NARRATION_INSTRUCTION not found in src.prompts.narration_instructions. Agent cannot be created.")
        raise ImportError("NARRATION_INSTRUCTION not found.")

    narration_agent = LlmAgent(
        name="NarrationAgent",
        model=settings.MODEL_GEMINI_FLASH or "gemini-1.5-flash-latest",
        instruction=agent_instruction, # Use the imported instruction
        description="Describes the events of the simulation turn based on the final state and interaction log.",
        # Pass BOTH wrapped tool objects
        tools=[
            get_context_tool_obj,
            save_narration_tool_obj,
        ],
    )
    console.print(f"Agent '[bold yellow]{narration_agent.name}[/bold yellow]' defined (Narrator Role).")

    # Log initialized tools
    if narration_agent.tools:
        tool_names = []
        for tool_obj in narration_agent.tools:
            if hasattr(tool_obj, 'func') and hasattr(tool_obj.func, '__name__'):
                tool_names.append(tool_obj.func.__name__)
            else:
                tool_names.append(type(tool_obj).__name__) # Fallback to class name
        logger.info(f"NarrationAgent initialized with tools: {tool_names}")
    else:
        logger.warning("NarrationAgent initialized with no tools.")

except ImportError as imp_err:
     console.print(f"[bold red]Error importing dependencies for NarrationAgent: {imp_err}[/bold red]")
     logger.error(f"Failed to import dependencies for NarrationAgent: {imp_err}")
except Exception as e:
    console.print(f"[bold red]Error creating NarrationAgent:[/bold red] {e}")
    logger.exception("Error creating NarrationAgent") # Log the full traceback


# --- Test Function ---
async def _test_agent():
    """Tests the Narration agent creation and basic execution."""
    if not narration_agent:
        print("Narration agent not initialized. Skipping test.")
        return

    print("\nTesting Narration Agent Execution...")
    # Mock data
    test_sim_id = "sim_narr_test"
    test_app_name = "test_app"
    test_user_id = "test_user"
    test_narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(test_sim_id)

    # Mock session with relevant state keys
    mock_state = {
        ACTIVE_SIMULACRA_IDS_KEY: [test_sim_id, "sim_other"], # Include target sim
        WORLD_STATE_KEY: {
            "world_time": "2024-01-01T10:15:00Z",
            "location_details": {"TestLocation": {"description": "A test room after actions."}},
            # ... other world state ...
        },
        # Mock data that get_narration_context might read
        SIMULACRA_LOCATION_KEY_FORMAT.format(test_sim_id): "TestLocation",
        SIMULACRA_GOAL_KEY_FORMAT.format(test_sim_id): "Test the narrator.",
        SIMULACRA_PERSONA_KEY_FORMAT.format(test_sim_id): {"Name": "NarrTest", "Personality_Traits": ["Observant"]},
        SIMULACRA_STATUS_KEY_FORMAT.format(test_sim_id): {"mood": "Content"},
        SIMULACRA_MONOLOGUE_KEY_FORMAT.format(test_sim_id): "Thinking about narration...",
        SIMULACRA_INTENT_KEY_FORMAT.format(test_sim_id): {"action_type": "wait"},
        ACTION_VALIDATION_KEY_FORMAT.format(test_sim_id): {"validation_status": "approved"},
        INTERACTION_RESULT_KEY_FORMAT.format(test_sim_id): None, # No interaction this turn
        TURN_EXECUTION_NARRATIVES_KEY: {test_sim_id: "NarrTest waited patiently."}, # From hypothetical Phase 4b
        test_narration_key: None # Ensure the target key is initially None
    }
    mock_session = Session(id="test_narr_session", user_id=test_user_id, app_name=test_app_name, state=mock_state)

    # Configure API Key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set.")
        return
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"Agent created: {narration_agent.name}")
    # Verify tools are FunctionTool instances
    print(f"Agent Tools: {[type(t).__name__ for t in narration_agent.tools]}")

    # Mock Runner and SessionService
    mock_session_service = InMemorySessionService()
    mock_session_service.create_session(
        user_id=mock_session.user_id,
        app_name=mock_session.app_name,
        session_id=mock_session.id,
        state=mock_session.state
    )
    runner = Runner(agent=narration_agent, app_name=test_app_name, session_service=mock_session_service)

    print("\n--- Running Test ---")
    # Trigger text should align with agent's instructions
    trigger = types.Content(parts=[types.Part(text="Generate and save the narrative for the turn.")])

    async for event in runner.run_async(user_id=mock_session.user_id, session_id=mock_session.id, new_message=trigger):
        print(f"Event from {event.author}:")
        if event.content and event.content.parts and event.content.parts[0].text:
             print(f"  Content: {event.content.parts[0].text}")
        if event.get_function_calls():
            call = event.get_function_calls()[0]
            # Truncate potentially long args for printing
            args_repr = repr(call.args)
            args_display = args_repr[:200] + ('...' if len(args_repr) > 200 else '')
            print(f"  Tool Call -> {call.name}({args_display})")
        if event.get_function_responses():
            resp = event.get_function_responses()[0]
            # Truncate potentially long response for printing
            resp_repr = repr(resp.response)
            resp_display = resp_repr[:200] + ('...' if len(resp_repr) > 200 else '')
            print(f"  Tool Response <- {resp.name} = {resp_display}")
        # Check for state delta applied by the save_narration tool
        if event.actions and event.actions.state_delta:
             print(f"  State Delta Applied: {event.actions.state_delta}")
        if event.error_message: print(f"  Error: {event.error_message}")

    # Check final state for the saved narration
    final_session = mock_session_service.get_session(
        app_name=test_app_name, user_id=test_user_id, session_id="test_narr_session"
    )
    print("\n--- Final State ---")
    saved_narration = final_session.state.get(test_narration_key)
    print(f"Saved Narration for {test_sim_id} (Key: {test_narration_key}):")
    if saved_narration:
        print(f"  >>> {saved_narration}")
    else:
        print("  >>> [Narration not found in final state]")


if __name__ == "__main__":
    # Basic setup to run the test function
    logging.basicConfig(level=logging.INFO) # Use INFO level for less noise
    asyncio.run(_test_agent())
