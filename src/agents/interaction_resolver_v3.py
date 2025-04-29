# src/agents/interaction_resolver_v3.py

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

# --- State Keys (Import or define centrally) ---
# Assuming these keys are defined and accessible
WORLD_STATE_KEY = "current_world_state"
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
# Keys for NPC state (assuming structure)
NPC_STATUS_KEY_FORMAT = "npc_{}_status"
NPC_LOCATION_KEY_FORMAT = "npc_{}_location"
# Keys for Object state (assuming structure)
OBJECT_STATE_KEY_FORMAT = "object_{}_state" # e.g., object_computer_livingroom_state
# Key for storing interaction outcomes (optional, for narration)
TURN_INTERACTION_LOG_KEY = "turn_interaction_log"

logger = logging.getLogger(__name__)

# --- Agent Definition ---

def create_agent(
    model_name: Optional[str] = None # Allow overriding model
) -> LlmAgent:
    """
    Factory function to create the V3 Interaction Resolution agent.

    This agent processes validated 'talk' and 'use' intents, determines
    outcomes based on the target (NPC or object), and generates state updates.

    Args:
        model_name: Optional override for the LLM model name.

    Returns:
        An LlmAgent responsible for resolving interactions.
    """
    llm_model = model_name or os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest")

    # Note: This agent processes MULTIPLE interactions in one go.
    # It needs the full validation results and current world state.
    instruction = f"""You are the Interaction Resolution engine for the simulation.
Your task is to process a batch of validated actions ('talk' and 'use') intended by Simulacra for this turn and determine their outcomes.
You must update the state of the target entities (NPCs or objects) and generate brief narrative snippets describing what happened.

Input State Keys:
- Validation Results: {{{TURN_VALIDATION_RESULTS_KEY}}} (Contains validated intents like {{'sim_id': {{'is_valid': True, 'validated_intent': {{'action_type': 'talk', 'target': 'npc_bob', ...}}}}}})
- Current World State: {{{WORLD_STATE_KEY}}} (Contains NPC statuses under keys like '{NPC_STATUS_KEY_FORMAT.format('<npc_id>')}', and object states under keys like '{OBJECT_STATE_KEY_FORMAT.format('<object_id>')}')

Processing Steps for EACH valid 'talk' or 'use' intent in the Validation Results:
1. Identify the acting Simulacrum (`sim_id`) and the `validated_intent` (action_type, target, content/details).
2. Determine the target type (NPC or object) based on the `target` ID format (e.g., 'npc_*' vs 'object_*').
3. Retrieve the current state of the `target` from the World State.
4. Based on the `action_type`, `target` state, and `content`/`details`:
    a. **If 'talk' (target is NPC):**
        - Generate a realistic response from the NPC based on their persona/status and the incoming message.
        - Determine any change in the NPC's status (mood, relationship towards speaker).
        - Create a state update delta for the NPC's status key (e.g., `{NPC_STATUS_KEY_FORMAT.format('bob')}: {{'mood': 'annoyed', ...}}`).
        - Create a narrative snippet (e.g., "Bob frowned and replied curtly to Alice.").
    b. **If 'use' (target is object):**
        - Determine how the action changes the object's state (e.g., computer power 'on', TV channel changed, item consumed).
        - Create a state update delta for the object's state key (e.g., `{OBJECT_STATE_KEY_FORMAT.format('computer_livingroom')}: {{'power': 'on', 'current_app': 'email'}}`).
        - Create a narrative snippet (e.g., "Alice turned on the computer and opened her email.").
    c. **If action is impossible/invalid based on target state** (e.g., using an 'off' computer):
        - Generate no state change for the target.
        - Create a narrative snippet indicating failure (e.g., "Bob tried to use the computer, but it was off.").
5. Collect ALL state update deltas generated across all processed intents into a single JSON object.
6. Collect ALL narrative snippets into a list of strings.

Output Format:
Produce a single JSON object containing two keys:
- "state_updates": A JSON object containing ALL the state deltas calculated in step 5 (e.g., {{ "{NPC_STATUS_KEY_FORMAT.format('bob')}": {{...}}, "{OBJECT_STATE_KEY_FORMAT.format('computer_livingroom')}": {{...}} }}).
- "narrative_log": A list of all the narrative snippet strings generated (e.g., ["Alice turned on the computer...", "Bob frowned..."]).

Example Output:
```json
{{
  "state_updates": {{
    "npc_bob_status": {{ "mood": "neutral", "last_interaction_with": "sim_alice" }},
    "object_computer_livingroom_state": {{ "power": "on", "current_app": "web_browser" }}
  }},
  "narrative_log": [
    "Alice said 'Hi' to Bob, who nodded in return.",
    "Alice then turned on the computer in the living room and opened the web browser."
  ]
}}
Ensure the "state_updates" object contains the full state key names as strings. Ensure the "narrative_log" is a list of strings. Output ONLY the final JSON object. """

    interaction_resolver_agent = LlmAgent(
        name="InteractionResolverAgent_V3",
        model=llm_model,
        instruction=instruction,
        description="Resolves validated 'talk' and 'use' interactions, updating target states and logging outcomes.",
    )

    logger.info(f"Created V3 LlmAgent '{interaction_resolver_agent.name}'")
    return interaction_resolver_agent

async def _test_agent(): # Need to define these here for the test scope global test_sim_id_alice, test_sim_id_bob, test_npc_id, test_object_id

    print("Testing Interaction Resolver V3 Agent Creation & Execution...")
    # Mock data
    test_sim_id_alice = "sim_alice"
    test_sim_id_bob = "sim_bob"
    test_npc_id = "npc_charlie"
    test_object_id = "object_coffee_machine_kitchen"

    # Mock session with some initial state
    mock_state = {
        WORLD_STATE_KEY: {
            "world_time": "2024-01-01T09:05:00Z",
            NPC_STATUS_KEY_FORMAT.format(test_npc_id): {"mood": "neutral", "location": "Kitchen"},
            OBJECT_STATE_KEY_FORMAT.format(test_object_id): {"power": "off", "water_level": "full", "beans_level": "full"},
            # Include other necessary world state parts
        },
        # Include Simulacra states if needed by resolver (unlikely directly)
        # ...
        TURN_VALIDATION_RESULTS_KEY: {
            test_sim_id_alice: {
                "is_valid": True,
                "validated_intent": {
                    "action_type": "talk",
                    "target": test_npc_id,
                    "content": "Morning Charlie, making coffee?"
                },
                "reasoning": "Valid talk action."
            },
            test_sim_id_bob: {
                "is_valid": True,
                "validated_intent": {
                    "action_type": "use",
                    "target": test_object_id,
                    "details": "make one cup"
                },
                "reasoning": "Valid use action."
            }
        },
        TURN_INTERACTION_LOG_KEY: [] # Initialize log key
    }
    mock_session = Session(id="test_session", user_id="test_user", app_name="test_app", state=mock_state)

    # Configure API Key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set.")
        return
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Create the agent
    agent = create_agent()
    print(f"Agent created: {agent.name}")

    # Mock Runner and SessionService
    mock_session_service = InMemorySessionService()
    mock_session_service.create_session(
        user_id=mock_session.user_id,
        app_name=mock_session.app_name,
        session_id=mock_session.id,
        state=mock_session.state
    )
    runner = Runner(agent=agent, app_name="test_app", session_service=mock_session_service)

    print("\n--- Running Test ---")
    trigger = types.Content(parts=[types.Part(text="Resolve interactions based on validation results.")])
    final_agent_output = None
    async for event in runner.run_async(user_id=mock_session.user_id, session_id=mock_session.id, new_message=trigger):
        print(f"Event from {event.author}:")
        if event.is_final_response() and event.content:
            print(f"  Final Content: {event.content.parts[0].text}")
            final_agent_output = event.content.parts[0].text # Capture final JSON output
        elif event.content:
            print(f"  Intermediate Content: {event.content.parts[0].text}") # Should ideally be empty
        if event.error_message: print(f"  Error: {event.error_message}")

    # Process the output (as the simulation loop would)
    # Process the output (as the simulation loop would)
    print("\n--- Processing Agent Output ---")
    if final_agent_output:
        # --- MODIFICATION START: Improved Cleaning ---
        logger.debug(f"Raw agent output: {repr(final_agent_output)}") # Log raw output
        cleaned_output = final_agent_output.strip() # Initial strip

        # Remove potential ```json prefix and strip again
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:].strip()
        # Remove potential ``` prefix (if ```json wasn't found) and strip again
        elif cleaned_output.startswith("```"):
             cleaned_output = cleaned_output[3:].strip()

        # Remove potential ``` suffix and strip again
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()

        logger.debug(f"Cleaned agent output for parsing: {repr(cleaned_output)}") # Log cleaned output
        # --- MODIFICATION END ---

        try:
            # Use the cleaned string for parsing
            output_data = json.loads(cleaned_output)
            state_updates = output_data.get("state_updates", {})
            narrative_log = output_data.get("narrative_log", [])

            # ... (rest of the processing logic remains the same) ...
            print("Applying State Updates:")
            # ... etc ...

        except json.JSONDecodeError as json_err:
            # ... (error handling remains the same, printing repr(cleaned_output)) ...
            error_message = f"Failed to decode JSON. Error: {json_err}. repr(cleaned_output):\n>>>\n{repr(cleaned_output)}\n<<<"
            logger.error(error_message)
            try:
                from rich.console import Console
                console = Console()
                console.print(f"[bold red]Error: Cleaned agent output was not valid JSON:[/bold red]\nError: {json_err}\nAttempted to parse (repr):\n>>>\n{repr(cleaned_output)}\n<<<")
            except ImportError:
                print(f"Error: Cleaned agent output was not valid JSON:\nError: {json_err}\nAttempted to parse (repr):\n>>>\n{repr(cleaned_output)}\n<<<")
        except Exception as e:
            try:
                from rich.console import Console
                console = Console()
                console.print(f"[bold red]Error processing agent output: {e}[/bold red]")
            except ImportError:
                print(f"Error processing agent output: {e}")
    else:
        try:
            from rich.console import Console
            console = Console()
            console.print("[yellow]Warning: No final output received from agent.[/yellow]")
        except ImportError:
            print("Warning: No final output received from agent.")


    # Check final state
    final_session = mock_session_service.get_session(app_name="test_app", user_id="test_user", session_id="test_session")
    print("\n--- Final Mock State (After Updates) ---")
    print(f"NPC Status ({test_npc_id}): {final_session.state.get(NPC_STATUS_KEY_FORMAT.format(test_npc_id))}")
    print(f"Object State ({test_object_id}): {final_session.state.get(OBJECT_STATE_KEY_FORMAT.format(test_object_id))}")
    print(f"Interaction Log: {final_session.state.get(TURN_INTERACTION_LOG_KEY)}")

if __name__ == "__main__":
    # Basic setup to run the test function
    logging.basicConfig(level=logging.DEBUG)
    # Define mock variables needed for the test scope if not defined globally
    # (These are already defined within _test_agent now, but keeping them here
    # doesn't hurt if you prefer defining them globally for the test script)
    test_sim_id_alice = "sim_alice"
    test_sim_id_bob = "sim_bob"
    test_npc_id = "npc_charlie"
    test_object_id = "object_coffee_machine_kitchen"
    asyncio.run(_test_agent())