# src/agents/world_engine.py
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
# <<< Import FunctionTool >>>
from google.adk.tools import FunctionTool
from google.genai import types

# Config and Prompts
from src.config import settings
from src.prompts import world_engine_instructions
# <<< Import only read_state_key >>>
from src.tools.world_engine_tools import read_state_key

logger = logging.getLogger(__name__)

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{sim_id}_intent"
# <<< Define the CONSISTENT format string for individual results >>>
VALIDATION_RESULT_KEY_FORMAT = "simulacra_{sim_id}_validation_result" # Correct format

# <<< REMOVE combined key definition (not used in this approach) >>>
# TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"


def create_world_engine_validator(
    sim_id: str,
    model_name: Optional[str] = None
) -> LlmAgent:
    """
    Factory function to create a specialized World Engine agent instance
    for validating a SINGLE simulacrum's action.
    Uses the read_state_key tool to fetch the intent.
    Outputs the result dictionary under the unique key 'simulacra_{sim_id}_validation_result'.
    """
    if not sim_id:
        raise ValueError("sim_id cannot be empty when creating a world engine validator.")

    llm_model = model_name or settings.MODEL_GEMINI_PRO or "gemini-1.5-flash-latest"
    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=sim_id)
    # <<< Define the specific output key using the CONSISTENT format >>>
    output_key_for_agent = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id) # Uses the corrected format

    try:
        # Use the updated instruction from the prompts module
        specific_instruction = world_engine_instructions.WORLD_ENGINE_INSTRUCTION.format(
            target_simulacra_id=sim_id,
            intent_key_for_prompt=intent_key,
            _WORLD_STATE_KEY=WORLD_STATE_KEY,
            # <<< Pass the specific output key (using consistent format) to the prompt >>>
            _OUTPUT_KEY=output_key_for_agent
        )

        # Explicitly wrap the read tool
        read_tool_obj = FunctionTool(func=read_state_key)

        agent = LlmAgent(
            name=f"WorldEngineValidator_{sim_id}",
            model=llm_model,
            instruction=specific_instruction,
            description=f"World Engine Validator for {sim_id}.",
            # <<< Only include the read tool >>>
            tools=[read_tool_obj],
            # <<< Use the specific output key for this agent >>>
            output_key=output_key_for_agent
        )
        # <<< Update log message >>>
        logger.info(f"Created WorldEngineValidator instance for {sim_id} with read tool and output key '{output_key_for_agent}'.")
        return agent

    except KeyError as ke:
        logger.exception(f"KeyError formatting instruction for {sim_id}: {ke}.")
        raise ValueError(f"Failed to format instruction for validator {sim_id}: {ke}") from ke
    except Exception as e:
        logger.exception(f"Error creating world_engine_validator for {sim_id}: {e}")
        raise RuntimeError(f"Failed to create validator agent for {sim_id}") from e

# --- Optional: Add a check for the instruction template content during import ---
try:
    if "{target_simulacra_id}" not in world_engine_instructions.WORLD_ENGINE_INSTRUCTION:
         logger.warning("world_engine_instructions.WORLD_ENGINE_INSTRUCTION might be missing the '{target_simulacra_id}' placeholder.")
    # <<< ADD BACK check for _OUTPUT_KEY >>>
    if "{_OUTPUT_KEY}" not in world_engine_instructions.WORLD_ENGINE_INSTRUCTION:
         logger.warning("world_engine_instructions.WORLD_ENGINE_INSTRUCTION might be missing the '{_OUTPUT_KEY}' placeholder.")
except AttributeError:
     logger.error("Could not access world_engine_instructions.WORLD_ENGINE_INSTRUCTION.")


# --- Test Function (Adjusted for individual output_key) ---
async def _test_agent():
    """Tests the World Engine Validator agent creation and basic execution using individual output_key."""
    print("Testing World Engine Validator Agent Creation & Execution (consistent individual output_key)...")
    # Mock data
    test_sim_id = "sim_test_validator"
    test_intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=test_sim_id)
    # <<< Define the specific key for the test using the CONSISTENT format >>>
    test_output_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=test_sim_id)
    test_app_name = "test_app"
    test_user_id = "test_user"

    # --- Test Case 1: Intent is present and valid ---
    print("\n--- Test Case 1: Valid Intent Present ---")
    mock_state_valid = {
        WORLD_STATE_KEY: {
            "world_time": "2024-01-01T10:00:00Z",
            "location_details": {"TestLocation": {"description": "A test room."}},
            "world_rules": {"gravity": True}
        },
        test_intent_key: { # Valid dictionary intent
            "action_type": "move",
            "destination": "TestLocation2"
        },
        # <<< Initialize the specific key >>>
        test_output_key: None
    }
    mock_session_valid = Session(id="test_session_valid", user_id=test_user_id, app_name=test_app_name, state=mock_state_valid)

    # --- Test Case 2: Intent is missing ---
    print("\n--- Test Case 2: Intent Missing ---")
    mock_state_missing = {
        WORLD_STATE_KEY: {
            "world_time": "2024-01-01T10:00:00Z",
            "location_details": {"TestLocation": {"description": "A test room."}},
            "world_rules": {"gravity": True}
        },
        # test_intent_key is missing
        # <<< Initialize the specific key >>>
        test_output_key: None
    }
    mock_session_missing = Session(id="test_session_missing", user_id=test_user_id, app_name=test_app_name, state=mock_state_missing)

    # Configure API Key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set.")
        return
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Create the agent instance
    try:
        agent = create_world_engine_validator(sim_id=test_sim_id)
        print(f"Agent created: {agent.name}")
        print(f"Agent Tools: {[type(t).__name__ for t in agent.tools]}") # Should show FunctionTool
        print(f"Agent Output Key: {agent.output_key}") # Verify specific output key
    except Exception as e:
        print(f"Agent creation failed: {e}")
        return

    # --- Run Test Case 1 ---
    print("\n--- Running Test Case 1 (Valid Intent) ---")
    mock_session_service_1 = InMemorySessionService()
    mock_session_service_1.create_session(
        user_id=mock_session_valid.user_id,
        app_name=mock_session_valid.app_name,
        session_id=mock_session_valid.id,
        state=mock_session_valid.state
    )
    runner1 = Runner(agent=agent, app_name=test_app_name, session_service=mock_session_service_1)
    trigger1 = types.Content(parts=[types.Part(text=f"Validate intent for {test_sim_id}.")])
    final_agent_output_1 = None

    async for event in runner1.run_async(user_id=mock_session_valid.user_id, session_id=mock_session_valid.id, new_message=trigger1):
        print(f"Event from {event.author}:")
        if event.content and event.content.parts and event.content.parts[0].text:
             print(f"  Content: {event.content.parts[0].text}")
             if event.is_final_response(): final_agent_output_1 = event.content.parts[0].text # Capture final output JSON string
        if event.get_function_calls():
            call = event.get_function_calls()[0]
            print(f"  Tool Call -> {call.name}({call.args})") # Should only be read_state_key
        if event.get_function_responses():
            resp = event.get_function_responses()[0]
            print(f"  Tool Response <- {resp.name} = {resp.response}")
        # <<< Check for state delta from the AGENT's final event (due to output_key) >>>
        if event.is_final_response() and event.actions and event.actions.state_delta:
             print(f"  Final State Delta Applied (via output_key): {event.actions.state_delta}")
        elif event.actions and event.actions.state_delta:
             print(f"  Intermediate State Delta: {event.actions.state_delta}") # Should be empty
        if event.error_message: print(f"  Error: {event.error_message}")

    # <<< Remove post-processing block >>>

    final_session_1 = mock_session_service_1.get_session(
        app_name=test_app_name, user_id=test_user_id, session_id="test_session_valid"
    )
    print("\n--- Final State (Test Case 1) ---")
    # <<< Check the specific state key directly >>>
    validation_result_1 = final_session_1.state.get(test_output_key)
    print(f"Validation Result (from state key '{test_output_key}'): {validation_result_1}")
    # Expected: Should be the dictionary { "validation_status": "approved", ... } or the raw string if parsing fails

    # --- Run Test Case 2 ---
    print("\n--- Running Test Case 2 (Missing Intent) ---")
    mock_session_service_2 = InMemorySessionService()
    mock_session_service_2.create_session(
        user_id=mock_session_missing.user_id,
        app_name=mock_session_missing.app_name,
        session_id=mock_session_missing.id,
        state=mock_session_missing.state
    )
    runner2 = Runner(agent=agent, app_name=test_app_name, session_service=mock_session_service_2)
    trigger2 = types.Content(parts=[types.Part(text=f"Validate intent for {test_sim_id}.")])
    final_agent_output_2 = None

    async for event in runner2.run_async(user_id=mock_session_missing.user_id, session_id=mock_session_missing.id, new_message=trigger2):
        print(f"Event from {event.author}:")
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"  Content: {event.content.parts[0].text}")
            if event.is_final_response(): final_agent_output_2 = event.content.parts[0].text
        if event.get_function_calls():
            call = event.get_function_calls()[0]
            print(f"  Tool Call -> {call.name}({call.args})") # Should only be read_state_key
        if event.get_function_responses():
            resp = event.get_function_responses()[0]
            print(f"  Tool Response <- {resp.name} = {resp.response}")
        if event.is_final_response() and event.actions and event.actions.state_delta:
             print(f"  Final State Delta Applied (via output_key): {event.actions.state_delta}")
        elif event.actions and event.actions.state_delta:
             print(f"  Intermediate State Delta: {event.actions.state_delta}")
        if event.error_message: print(f"  Error: {event.error_message}")

    # <<< Remove post-processing block >>>

    final_session_2 = mock_session_service_2.get_session(
        app_name=test_app_name, user_id=test_user_id, session_id="test_session_missing"
    )
    print("\n--- Final State (Test Case 2) ---")
    validation_result_2 = final_session_2.state.get(test_output_key)
    print(f"Validation Result (from state key '{test_output_key}'): {validation_result_2}")
    # Expected: Should be the dictionary { "validation_status": "no_intent_found", ... } or the raw string

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Need to import parse_json_output for the test function
    try:
        from src.loop_utils import parse_json_output
        from rich.console import Console # Needed by parse_json_output
    except ImportError:
        print("Could not import loop_utils.parse_json_output for test.")
        # Define a dummy function if needed
        def parse_json_output(*args, **kwargs): return None
    asyncio.run(_test_agent())

