# src/agents/interaction_resolver_v3.py

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

# ADK Imports
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

# --- State Keys (Import or define centrally) ---
# Assuming these keys are defined and accessible
WORLD_STATE_KEY = "current_world_state"
VALIDATION_RESULT_KEY_FORMAT = "simulacra_{sim_id}_validation_result"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
PROFILE_STATUS_KEY = "status"
PROFILE_LOCATION_KEY = "current_location"
STATUS_INTERACTION_STATUS_KEY = "interaction_status"
STATUS_INTERACTION_PARTNER_KEY = "interaction_partner_id"
STATUS_INTERACTION_MEDIUM_KEY = "interaction_medium"
STATUS_LAST_INTERACTION_SNIPPET_KEY = "last_interaction_snippet"
NPC_STATUS_KEY_FORMAT = "npc_{}_status"
NPC_LOCATION_KEY_FORMAT = "npc_{}_location" # Assuming NPCs have locations
OBJECT_STATE_KEY_FORMAT = "object_{}_state" # e.g., object_computer_livingroom_state
TURN_INTERACTION_LOG_KEY = "turn_interaction_log"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids" # Needed to identify Sim targets
SCHEDULED_EVENTS_KEY = "scheduled_events" # List of events to be processed in future turns

logger = logging.getLogger(__name__)

# --- Agent Definition ---

def create_agent(
    model_name: Optional[str] = None # Allow overriding model
) -> LlmAgent:
    """
    Factory function to create the V3 Interaction Resolution agent.

    This agent processes validated 'talk' and 'use' intents, determines
    outcomes based on the target (Simulacra, NPC, or object) and interaction state,
    schedules latent events, and generates state updates and narrative snippets.

    Args:
        model_name: Optional override for the LLM model name.

    Returns:
        An LlmAgent responsible for resolving interactions.
    """
    llm_model = model_name or os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest")

    # --- Define instruction template using descriptive text for LLM placeholders ---
    instruction_template = """You are the Interaction Resolution engine for the simulation.
Your task is to process a batch of validated actions ('talk', 'use', 'end_interaction') intended by Simulacra for this turn and determine their outcomes.
You must update the state of the involved entities (Simulacra, NPCs, or objects) OR schedule future events for latent actions, and generate brief narrative snippets describing what happened.

Input State Keys:
- Individual Validation Results: Keys like `{_VALIDATION_RESULT_KEY_FORMAT_EXAMPLE_}` (Contains validated intents like {{'validation_status': 'approved', 'original_intent': {{'action_type': 'talk', 'target': 'sim_bob', ...}}}})
- Current World State: `{_WORLD_STATE_KEY_}` (Contains NPC statuses, object states, time, current_turn, etc.)
- Simulacra Profiles: `{_SIMULACRA_PROFILES_KEY_}` (Contains profiles for all simulacra, including their status under `{_PROFILE_STATUS_KEY_}`)
- Active Simulacra IDs: `{_ACTIVE_SIMULACRA_IDS_KEY_}` (List of active sim IDs)
- Scheduled Events List: `{_SCHEDULED_EVENTS_KEY_}` (List of pending future events)

Processing Steps for EACH valid ('approved' or 'modified') 'talk', 'use', or 'end_interaction' intent found in the validation results:
1. Identify the acting Simulacrum (get their ID from the validation result, let's call this the [ACTOR_ID]) and their `original_intent` (action_type, target, content/details).
2. Retrieve the actor's current profile and status from `{_SIMULACRA_PROFILES_KEY_}.[ACTOR_ID].{_PROFILE_STATUS_KEY_}`. Note their `{_STATUS_INTERACTION_STATUS_KEY_}`, `{_STATUS_INTERACTION_PARTNER_KEY_}`, `{_STATUS_INTERACTION_MEDIUM_KEY_}`.
3. Determine the target type and the communication medium (if applicable).
    - Is the `target` ID in the `{_ACTIVE_SIMULACRA_IDS_KEY_}` list? -> Target is a Simulacrum (let's call their ID [TARGET_SIM_ID]).
    - Does the `target` ID start with 'npc_'? -> Target is an NPC (let's call their ID [TARGET_NPC_ID]).
    - Does the `target` ID start with 'object_'? -> Target is an Object (let's call its ID [TARGET_OBJECT_ID]).
    - Is the `target` a device like 'phone' or 'computer' used for communication? -> Target is Object (Device).
4. Retrieve the current state of the `target` (Simulacrum profile/status, NPC status, or Object state) from the state.
5. Based on the `action_type`, actor status, target type, target state, medium, and `content`/`details`:

    # <<< Check for Latent Actions FIRST >>>
    h. **If Action has Latency (e.g., 'use carrier_pigeon', 'cast telepathy_spell'):**
        - Determine Latency: Based on the medium (pigeon, spell type) and context (distance, rules), estimate latency in turns (e.g., `latency_turns = 3`). Assume 1 turn minimum if unsure.
        - Calculate Arrival Turn: Get `current_turn` number (from `{_WORLD_STATE_KEY_}.current_turn`). `arrival_turn = current_turn + latency_turns`.
        - Create Scheduled Event: Prepare a dictionary like: `{{"type": "message_arrival", "recipient_id": "[TARGET_SIM_ID]", "sender_id": "[ACTOR_ID]", "content": "message content", "medium": "carrier_pigeon", "arrival_turn": arrival_turn}}`.
        - **Schedule Event:** Add this event dictionary to the list under the `{_SCHEDULED_EVENTS_KEY_}` state key. (Your output `state_updates` should reflect adding this item to the list).
        - Update Sender State: Update the [ACTOR_ID]'s status to 'awaiting_delivery' or 'spell_in_transit', partner=[TARGET_SIM_ID], medium='carrier_pigeon'/'telepathy_spell'.
        - Narrative: Describe the *sending* action (e.g., "Alice attaches the note to the pigeon and releases it.").
        - **Do NOT update the recipient's state directly.**

    # <<< Existing logic becomes 'else if' or follows after latency check >>>
    a. **Else if 'talk' (Target is Simulacrum [TARGET_SIM_ID]):**
        - Check Actor Status: Are they `in_conversation` or `waiting_response` with the target Simulacrum ID?
        - **If Ongoing Remote:** (Actor status matches target & medium is 'phone'/'text' etc.)
            - Update the target Simulacrum's status: set `{_STATUS_LAST_INTERACTION_SNIPPET_KEY_}` to actor's `content`, set `{_STATUS_INTERACTION_STATUS_KEY_}` to 'waiting_response'.
            - Update actor's status: set `{_STATUS_INTERACTION_STATUS_KEY_}` to 'waiting_response'.
            - Narrative: Describe the exchange over the remote medium.
        - **If Ongoing In-Person:** (Actor status matches target & medium is 'in_person')
            - Check Location: Are actor (`{_PROFILE_LOCATION_KEY_}`) and target (`{_SIMULACRA_PROFILES_KEY_}.[TARGET_SIM_ID].{_PROFILE_LOCATION_KEY_}`) in the same location?
            - If Yes: Update the target Simulacrum's status (snippet, 'waiting_response'). Update actor's status ('waiting_response'). Narrative: Describe face-to-face exchange.
            - If No: Action fails. Narrative: "[Actor] tried to talk to [Target], but they weren't there."
        - **Else (Initiating Talk):**
            - Check Location: Must be same location.
            - If Yes: Initiate `in_person` interaction. Update actor status ('waiting_response', partner=target, medium='in_person'). Update target status ('being_spoken_to', partner=[ACTOR_ID], medium='in_person', snippet=content). Narrative: "[Actor] approaches [Target] and says..."
            - If No: Action fails. Narrative: "[Actor] looked for [Target] to talk, but couldn't find them."

    b. **Else if 'talk' (Target is NPC [TARGET_NPC_ID]):**
        - Check Location: Are actor and NPC (`{_NPC_LOCATION_KEY_FORMAT_EXAMPLE_}`) in the same location?
        - If Yes: Generate NPC response based on NPC persona/status and actor's message. Determine NPC status change (mood etc.). Create state update delta for `{_NPC_STATUS_KEY_FORMAT_EXAMPLE_}`. Narrative: Describe the exchange.
        - If No: Action fails. Narrative: "[Actor] tried to talk to [NPC Name], but they weren't there."

    c. **Else if 'use' (Target is Object - Device for IMMEDIATE Communication):**
        - **If `details` specify contacting a Simulacrum [TARGET_SIM_ID] (e.g., "call sim_bob", "text sim_bob"):**
            - Check the target Simulacrum's status (busy, idle?).
            - If available: Update actor status ('calling'/'sending_text', partner=[TARGET_SIM_ID], medium='phone'/'text'). Update the target Simulacrum's status ('receiving_call'/'receiving_text', partner=[ACTOR_ID], medium='phone'/'text', snippet="Incoming call/text from [Actor]"). Narrative: "[Actor] dials [Target]'s number." / "[Actor] sends a text to [Target]."
            - If unavailable: Narrative: "[Actor] tries calling/texting [Target], but gets no answer/fails to send."
        - **If `details` specify answering/responding (e.g., "answer call", "read text"):**
            - Check actor's status (`receiving_call` or `receiving_text`).
            - If `receiving_call`: Update actor status ('in_conversation', medium='phone'). Update caller's status ('in_conversation', medium='phone'). Narrative: "[Actor] answers the call from [Caller]."
            - If `receiving_text`: Update actor status ('idle' or 'waiting_response', clear snippet). Narrative: "[Actor] reads the text from [Sender]."

    d. **Else if 'use' (Target is Simulacrum [TARGET_SIM_ID] - Physical Interaction):**
        - Check Location: Must be same location.
        - Determine effect (e.g., using item on them). Create state update delta for `{_SIMULACRA_PROFILES_KEY_}.[TARGET_SIM_ID].{_PROFILE_STATUS_KEY_}` (e.g., update condition). Narrative: Describe the physical action.

    e. **Else if 'use' (Target is Object - General [TARGET_OBJECT_ID]):**
        - Determine how the action changes the object's state (e.g., computer power 'on', item consumed).
        - Create state update delta for the object's state key (e.g., `{_OBJECT_STATE_KEY_FORMAT_EXAMPLE_}: {{'power': 'on'}}`).
        - Narrative: Describe the action and outcome (e.g., "Alice turned on the computer.").

    f. **Else if 'end_interaction' (Target is Simulacrum [TARGET_SIM_ID]):**
        - Check Actor Status: Are they interacting with the target Simulacrum ID?
        - If Yes: Update actor status ('idle', partner=None, medium=None, snippet=None). Update the target Simulacrum's status ('idle', partner=None, medium=None, snippet=None). Narrative: "[Actor] ends the conversation/call with [Target]."
        - If No: Action has no effect. Narrative: "[Actor] wasn't interacting with [Target]."

    g. **Else if Action Fails based on Target State:**
        - Generate no state change for the target.
        - Narrative: Indicate failure (e.g., "Bob tried to use the computer, but it was off.").

6. Collect ALL state update deltas generated across all processed intents into a single JSON object. This includes updates to Simulacra/NPC/Object states AND potentially adding items to the `{_SCHEDULED_EVENTS_KEY_}` list.
    # <<< NOTE on List Update: How to represent adding to a list depends on ADK/state service.
    #     Assume for now you output the key with the *new, complete list* if you modified it,
    #     OR use a special syntax if supported (e.g., `{{"{_SCHEDULED_EVENTS_KEY_}": {{"_append": [new_event_dict]}}}}`).
    #     Let's assume outputting the new complete list for simplicity in the prompt. >>>
    Read the current `{_SCHEDULED_EVENTS_KEY_}` list from state, append any new scheduled events from step 5h, and include the updated list in your `state_updates` output if it changed.
7. Collect ALL narrative snippets into a list of strings.

Output Format:
Produce a single JSON object containing two keys:
- "state_updates": A JSON object containing ALL the state deltas calculated in step 6 (e.g., {{ "{_SIMULACRA_PROFILES_KEY_}.sim_bob.{_PROFILE_STATUS_KEY_}": {{...}}, "{_SCHEDULED_EVENTS_KEY_}": [existing_event, new_event] }}).
- "narrative_log": A list of all the narrative snippet strings generated.

Ensure the "state_updates" object contains the full state key names as strings. Ensure the "narrative_log" is a list of strings. Output ONLY the final JSON object. """

    # Format the template with actual Python constants using .format()
    instruction = instruction_template.format(
        _VALIDATION_RESULT_KEY_FORMAT_EXAMPLE_=VALIDATION_RESULT_KEY_FORMAT.format(sim_id='<sim_id>'),
        _WORLD_STATE_KEY_=WORLD_STATE_KEY,
        _SIMULACRA_PROFILES_KEY_=SIMULACRA_PROFILES_KEY,
        _PROFILE_STATUS_KEY_=PROFILE_STATUS_KEY,
        _ACTIVE_SIMULACRA_IDS_KEY_=ACTIVE_SIMULACRA_IDS_KEY,
        _STATUS_INTERACTION_STATUS_KEY_=STATUS_INTERACTION_STATUS_KEY,
        _STATUS_INTERACTION_PARTNER_KEY_=STATUS_INTERACTION_PARTNER_KEY,
        _STATUS_INTERACTION_MEDIUM_KEY_=STATUS_INTERACTION_MEDIUM_KEY,
        _STATUS_LAST_INTERACTION_SNIPPET_KEY_=STATUS_LAST_INTERACTION_SNIPPET_KEY,
        _PROFILE_LOCATION_KEY_=PROFILE_LOCATION_KEY,
        # Use descriptive text in examples instead of placeholders needing format
        _NPC_LOCATION_KEY_FORMAT_EXAMPLE_=NPC_LOCATION_KEY_FORMAT.format('[TARGET_NPC_ID]'),
        _NPC_STATUS_KEY_FORMAT_EXAMPLE_=NPC_STATUS_KEY_FORMAT.format('[TARGET_NPC_ID]'),
        _OBJECT_STATE_KEY_FORMAT_EXAMPLE_=OBJECT_STATE_KEY_FORMAT.format('[TARGET_OBJECT_ID]'),
        # <<< ADDED KEY >>>
        _SCHEDULED_EVENTS_KEY_=SCHEDULED_EVENTS_KEY
    )

    interaction_resolver_agent = LlmAgent(
        name="InteractionResolverAgent_V3",
        model=llm_model,
        instruction=instruction, # Use the formatted string
        description="Resolves validated 'talk', 'use', 'end_interaction' actions, schedules latent events, updates states, and logs outcomes.",
        # No tools needed, relies on state access via prompt context
    )

    logger.info(f"Created V3 LlmAgent '{interaction_resolver_agent.name}' with latency handling.")
    return interaction_resolver_agent

# --- Test Function (Needs update for latency) ---
async def _test_agent():
    print("Testing Interaction Resolver V3 Agent Creation & Execution (with Latency)...")
    # Mock data
    test_sim_id_alice = "sim_alice"
    test_sim_id_bob = "sim_bob"
    test_pigeon_id = "object_carrier_pigeon_alice" # Example object ID

    # --- Assume current turn is 5 for latency calculation ---
    current_turn_for_test = 5

    # Mock session with initial state
    mock_state = {
        WORLD_STATE_KEY: {
            "world_time": "2024-01-01T09:05:00Z",
            # <<< ADD current_turn to world state for agent context >>>
            "current_turn": current_turn_for_test,
            # ... other world state ...
        },
        ACTIVE_SIMULACRA_IDS_KEY: [test_sim_id_alice, test_sim_id_bob],
        SIMULACRA_PROFILES_KEY: {
            test_sim_id_alice: {
                "persona_details": {"Name": "Alice"},
                "current_goal": "Send message",
                PROFILE_STATUS_KEY: { "mood": "Neutral", STATUS_INTERACTION_STATUS_KEY: "idle", },
                PROFILE_LOCATION_KEY: "Tower"
            },
            test_sim_id_bob: {
                "persona_details": {"Name": "Bob"},
                "current_goal": "Wait for news",
                 PROFILE_STATUS_KEY: { "mood": "Waiting", STATUS_INTERACTION_STATUS_KEY: "idle", },
                PROFILE_LOCATION_KEY: "Castle"
            }
        },
        # --- Add validation result for latent action ---
        VALIDATION_RESULT_KEY_FORMAT.format(sim_id=test_sim_id_alice): {
            "validation_status": "approved",
            "original_intent": {
                "action_type": "use",
                "target": test_pigeon_id,
                "details": f"Send message 'Meet tonight' via carrier_pigeon to {test_sim_id_bob}"
            },
            "reasoning": "Valid use action."
        },
        # --- Initialize scheduled events list ---
        SCHEDULED_EVENTS_KEY: [],
        TURN_INTERACTION_LOG_KEY: []
    }
    mock_session = Session(id="test_session_latency", user_id="test_user", app_name="test_app", state=mock_state)

    # Configure API Key
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

    print("\n--- Running Test (Latency Scenario) ---")
    trigger = types.Content(parts=[types.Part(text=f"Resolve interactions for turn {current_turn_for_test} based on individual validation results.")])
    final_agent_output = None
    async for event in runner.run_async(user_id=mock_session.user_id, session_id=mock_session.id, new_message=trigger):
        print(f"Event from {event.author}:")
        if event.is_final_response() and event.content:
            print(f"  Final Content: {event.content.parts[0].text}")
            final_agent_output = event.content.parts[0].text
        elif event.content:
            print(f"  Intermediate Content: {event.content.parts[0].text}")
        if event.error_message: print(f"  Error: {event.error_message}")

    # Process the output
    print("\n--- Processing Agent Output ---")
    if final_agent_output:
        # ... (cleaning logic remains the same) ...
        logger.debug(f"Raw agent output: {repr(final_agent_output)}")
        cleaned_output = final_agent_output.strip()
        if cleaned_output.startswith("```json"): cleaned_output = cleaned_output[7:].strip()
        elif cleaned_output.startswith("```"): cleaned_output = cleaned_output[3:].strip()
        if cleaned_output.endswith("```"): cleaned_output = cleaned_output[:-3].strip()
        logger.debug(f"Cleaned agent output for parsing: {repr(cleaned_output)}")

        try:
            output_data = json.loads(cleaned_output)
            state_updates = output_data.get("state_updates", {})
            narrative_log = output_data.get("narrative_log", [])

            print("Applying State Updates:")
            if state_updates:
                # Simulate update
                for key, value in state_updates.items():
                    print(f"  - {key}: {value}")
                    # Simulate nested update for profiles
                    if key.startswith(f"{SIMULACRA_PROFILES_KEY}."):
                        parts = key.split('.')
                        if len(parts) >= 3:
                            sim_id = parts[1]
                            current_profile = mock_session.state.get(SIMULACRA_PROFILES_KEY, {}).get(sim_id, {})
                            # Dive into nested structure (simplified simulation)
                            target = current_profile
                            for part in parts[2:-1]:
                                target = target.setdefault(part, {})
                            if isinstance(target, dict):
                                target[parts[-1]] = value
                            else: # Handle direct update like simulacra_profiles.sim_alice.status = {...}
                                if len(parts) == 3:
                                     mock_session.state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[parts[2]] = value
                                else: print(f"    [Warning] Cannot simulate nested update for non-dict target: {key}")
                        else: mock_session.state[key] = value # Fallback direct update
                    else:
                        mock_session.state[key] = value # Direct update for non-profile keys
            else:
                print("  (No state updates generated)")

            print("\nNarrative Log:")
            if narrative_log:
                # Simulate adding to log
                if TURN_INTERACTION_LOG_KEY not in mock_session.state: mock_session.state[TURN_INTERACTION_LOG_KEY] = []
                mock_session.state[TURN_INTERACTION_LOG_KEY].extend(narrative_log)
                for entry in narrative_log:
                    print(f"  - {entry}")
            else:
                print("  (No narrative entries generated)")

        except json.JSONDecodeError as json_err:
            # ... (error handling remains the same) ...
            error_message = f"Failed to decode JSON. Error: {json_err}. repr(cleaned_output):\n>>>\n{repr(cleaned_output)}\n<<<"
            logger.error(error_message)
            try: from rich.console import Console; console = Console(); console.print(f"[bold red]Error: Cleaned agent output was not valid JSON:[/bold red]\nError: {json_err}\nAttempted to parse (repr):\n>>>\n{repr(cleaned_output)}\n<<<")
            except ImportError: print(f"Error: Cleaned agent output was not valid JSON:\nError: {json_err}\nAttempted to parse (repr):\n>>>\n{repr(cleaned_output)}\n<<<")
        except Exception as e:
            try: from rich.console import Console; console = Console(); console.print(f"[bold red]Error processing agent output: {e}[/bold red]")
            except ImportError: print(f"Error processing agent output: {e}")
    else:
        try: from rich.console import Console; console = Console(); console.print("[yellow]Warning: No final output received from agent.[/yellow]")
        except ImportError: print("Warning: No final output received from agent.")


    # Check final state
    print("\n--- Final Mock State (After Updates) ---")
    print(f"Alice Status: {mock_session.state.get(SIMULACRA_PROFILES_KEY, {}).get(test_sim_id_alice, {}).get(PROFILE_STATUS_KEY)}")
    print(f"Bob Status: {mock_session.state.get(SIMULACRA_PROFILES_KEY, {}).get(test_sim_id_bob, {}).get(PROFILE_STATUS_KEY)}")
    # <<< Check the scheduled events list >>>
    print(f"Scheduled Events: {mock_session.state.get(SCHEDULED_EVENTS_KEY)}")
    print(f"Interaction Log: {mock_session.state.get(TURN_INTERACTION_LOG_KEY)}")

if __name__ == "__main__":
    # Basic setup to run the test function
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see detailed logs
    asyncio.run(_test_agent())

