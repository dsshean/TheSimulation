# src/agents/simulacra_v3.py

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

# ADK Imports
from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent # Added BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import FunctionTool, ToolContext # Added ToolContext
from google.genai import types

# --- State Keys (Define or import centrally) ---
# Assuming these might be defined elsewhere, but defining here for completeness
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
LOCATION_DETAILS_KEY = "location_details" # Used in Observation
WORLD_TIME_KEY = "world_time" # Used in Observation
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details" # Used in Observation
OBSERVE_OUTPUT_KEY = "sim_observation_output" # Define if not imported
REFLECTION_OUTPUT_KEY = "sim_reflection_output" # Define if not imported
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{sim_id}_intent" # Define if not imported

# Keys within the profile dictionary (assuming structure like state['simulacra_profiles'][sim_id])
PROFILE_PERSONA_KEY = "persona_details"
PROFILE_GOAL_KEY = "current_goal"
PROFILE_STATUS_KEY = "status" # Key holding status dict within profile
PROFILE_LOCATION_KEY = "current_location"

# --- NEW: Keys within the status dictionary ---
STATUS_INTERACTION_STATUS_KEY = "interaction_status" # e.g., 'idle', 'calling', 'in_conversation'
STATUS_INTERACTION_PARTNER_KEY = "interaction_partner_id" # ID of Sim/NPC
STATUS_INTERACTION_MEDIUM_KEY = "interaction_medium" # e.g., 'in_person', 'phone', 'text'
STATUS_LAST_INTERACTION_SNIPPET_KEY = "last_interaction_snippet" # Last message received
STATUS_MOOD_KEY = "mood" # Existing example
STATUS_CONDITION_KEY = "condition" # Existing example
# ---

logger = logging.getLogger(__name__)

# --- Agent Factory ---

def create_agent(
    sim_id: str,
    persona: Dict[str, Any],
    session: Session, # Session object is passed for context if needed by tools/agents
    model_name: Optional[str] = None
) -> SequentialAgent:
    """
    Factory function to create the V3 Simulacra agent using a SequentialAgent.
    Handles detailed interaction status checks.
    """
    llm_model = model_name or os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest")
    persona_name = persona.get("Name", sim_id)
    persona_traits = persona.get("Personality_Traits", [])
    long_term_aspirations = persona.get("Long_Term_Aspirations", "None specified")

    # Construct state key paths used in prompts
    profile_base_key = f"{SIMULACRA_PROFILES_KEY}.{sim_id}"
    status_state_key = f"{profile_base_key}.{PROFILE_STATUS_KEY}" # Path to the status dict
    persona_state_key = f"{profile_base_key}.{PROFILE_PERSONA_KEY}"
    goal_state_key = f"{profile_base_key}.{PROFILE_GOAL_KEY}"
    location_state_key = f"{profile_base_key}.{PROFILE_LOCATION_KEY}"

    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=sim_id)

    # --- Sub-Agent Definitions ---

    # 1. Observation Agent (Modify instructions to include interaction status)
    observation_instruction = f"""
You are {persona_name} ({sim_id}). Your task is to observe your immediate surroundings and internal state with detail, internalizing who and where you are right now.

Access the current simulation state to understand your context:
1. Your ID is '{sim_id}'.
2. Find your current location key: `state['{location_state_key}']` -> `my_location_key`.
3. Look up details for this location: `state['{WORLD_STATE_KEY}']['{LOCATION_DETAILS_KEY}'].get(my_location_key)` -> `location_data`.
4. Get the description if available: `location_data.get('description') if location_data else None` -> `location_description`.
5. Get the current world time: `state['{WORLD_STATE_KEY}']['{WORLD_TIME_KEY}']`.
6. Note the general setting: `state['{WORLD_TEMPLATE_DETAILS_KEY}']['description']`.
7. **Check your internal status:** `state['{status_state_key}']` which includes:
    - `{STATUS_MOOD_KEY}`: Your current mood.
    - `{STATUS_CONDITION_KEY}`: Your physical condition.
    - **`{STATUS_INTERACTION_STATUS_KEY}`: Your current interaction state (e.g., 'idle', 'in_conversation', 'receiving_call').**
    - **`{STATUS_INTERACTION_PARTNER_KEY}`: Who you are interacting with (if any).**
    - **`{STATUS_INTERACTION_MEDIUM_KEY}`: How you are interacting (if any).**
    - **`{STATUS_LAST_INTERACTION_SNIPPET_KEY}`: The last message/action received (if any).**
    - Other status fields like social need, recent events.
8. Consider who else is nearby and what they are doing (from `state['{WORLD_STATE_KEY}']` if available).
9. Note any significant weather or global events (from `state['{WORLD_STATE_KEY}']`).

**Location Handling Logic:**
*   If `location_description` is available, use it.
*   Otherwise, use `my_location_key` as the location description.
*   Let the final description be `Final Location Description`.

Synthesize these pieces of information into a brief, factual observation (4-6 sentences) summarizing these points. **Crucially, include your current interaction status if not 'idle'.**

Example Output (Idle): "It's [Time Description] in [Final Location Description]. I'm currently feeling [Your Status/Mood]."
Example Output (In Call): "It's [Time Description] in [Final Location Description]. I'm currently feeling [Your Status/Mood] and am on a phone call with [Partner ID]. The last thing they said was '[Snippet]'."
Example Output (Receiving Call): "It's [Time Description] in [Final Location Description]. I'm currently feeling [Your Status/Mood]. My phone is ringing - it's a call from [Partner ID]."

Output ONLY the observation text.
"""
    observation_agent = LlmAgent(
        name=f"Sim_{sim_id}_Observe",
        model=llm_model,
        instruction=observation_instruction,
        description=f"Observes the current state, location, and interaction status for {persona_name}.",
        output_key=OBSERVE_OUTPUT_KEY
    )

    # 2. Reflection Agent (Modify instructions to react to interaction status)
    reflection_instruction = f"""
You are {persona_name} ({sim_id}). You just made an observation about your situation. Now, reflect deeply like a real person would, connecting it to your personality, goals, feelings, and recent experiences. **Embody your persona.**

Your Core Info (from state):
- Persona: `state['{persona_state_key}']` (Traits: {persona_traits}, Aspirations: {long_term_aspirations})
- Current Goal: `state['{goal_state_key}']`
- Current Status: `state['{status_state_key}']` (Includes mood, condition, **interaction_status, interaction_partner_id, interaction_medium, last_interaction_snippet**)
- Your Observation: {{{OBSERVE_OUTPUT_KEY}}}
- World Context (Time/Weather/News): `state['{WORLD_STATE_KEY}']`

**Think Step-by-Step (Internal Monologue):**
1.  **React to Observation & Interaction Status:**
    *   How does your current status (mood, physical state, social need) influence your reaction to the observation?
    *   **IF** your `{STATUS_INTERACTION_STATUS_KEY}` is NOT 'idle':
        *   **IF** 'in_conversation' or 'waiting_response': Focus on the `{STATUS_LAST_INTERACTION_SNIPPET_KEY}`. What did `{STATUS_INTERACTION_PARTNER_KEY}` just say/do? How should you respond based on your persona, goal, and relationship? Should you continue, change the subject, or end the interaction?
        *   **IF** 'receiving_call' or 'receiving_text': Do you want to answer/respond? Who is it from (`{STATUS_INTERACTION_PARTNER_KEY}`)? Are you busy? Does your mood/goal make you want to engage or ignore it?
        *   **IF** 'calling': Are you getting impatient? Should you wait longer or hang up?
    *   **ELSE** (interaction_status is 'idle'): How do recent events/interactions (if any mentioned in status) color your thoughts?
2.  **Evaluate Goal:** Is your current goal (`state['{goal_state_key}']`) still relevant or achievable given the situation (time, weather, your status, **ongoing interaction**)? Does it align with your long-term aspirations (`{long_term_aspirations}`)? Do you feel motivated to pursue it *now*, or does the interaction take priority?
3.  **Consider Alternatives/Needs:** Based on your persona traits (`{persona_traits}`), feelings, social needs, or reactions to the situation/news/weather/interaction, are there other things you *want* or *need* to do right now?
4.  **Prioritize:** What feels most important *right now*? **Responding to the interaction?** Your goal? An immediate need? A social impulse? Reacting to the environment? Briefly weigh the options.
5.  **Possible Pivot?:** If you strongly feel like changing your short-term goal based on this reflection (e.g., because the interaction sparked a new idea, or your original goal is blocked), clearly state the *new goal* you're considering.

Output ONLY your realistic, first-person detailed internal monologue (6-10 sentences) reflecting this thought process, **especially how you're handling the current interaction status**, prioritization, and potential goal shift.
"""
    reflection_agent = LlmAgent(
        name=f"Sim_{sim_id}_Reflect",
        model=llm_model,
        instruction=reflection_instruction,
        description=f"Reflects on observations and interaction status for {persona_name}.",
        output_key=REFLECTION_OUTPUT_KEY
    )

    # 3. Intent Formation Agent (Modify instructions to prioritize interaction response)
    intent_formation_instruction = f"""
You are {persona_name} ({sim_id}). Based on your observation and reflection, decide your single, primary action/intent for this turn. Be plausible and in-character.

Your Reflection/Thoughts: {{{REFLECTION_OUTPUT_KEY}}}
Your Observation: {{{OBSERVE_OUTPUT_KEY}}}
World Context (Time/Weather/Location/Nearby): `state['{WORLD_STATE_KEY}']`
Your Location: `state['{location_state_key}']`
Your Status: `state['{status_state_key}']` (Includes `{STATUS_INTERACTION_STATUS_KEY}`, `{STATUS_INTERACTION_PARTNER_KEY}`, `{STATUS_INTERACTION_MEDIUM_KEY}`)

**Decision Factors:**
- Your prioritized need/desire from your reflection.
- **Interaction Priority:** If your `{STATUS_INTERACTION_STATUS_KEY}` is 'in_conversation', 'waiting_response', 'receiving_call', or 'receiving_text', your primary action should usually be related to handling that interaction (responding, answering, ending, ignoring) unless your reflection strongly justifies prioritizing something else immediately.
- Plausibility: Is the action feasible given the time, location, weather, social context, your status, **and the interaction medium**?
- Goal Alignment: Does it progress your stated goal, address a temporary priority, or represent a shift in focus?

**Action Formatting:**
Choose ONE action type ('talk', 'move', 'use', 'wait', 'think', 'update_goal', 'end_interaction'). Format it as a JSON object.
- `{{ "action_type": "talk", "target": "partner_id_from_status", "content": "Your response" }}` (If responding in conversation, use partner ID from status)
- `{{ "action_type": "move", "destination": "Location Name or ID" }}`
- `{{ "action_type": "use", "target": "object_id", "details": "Specific action, e.g., 'answer call', 'read text', 'hang up', 'call Sarah', 'check news'" }}` (Use for phone/computer interactions, including answering/ending calls/texts)
- `{{ "action_type": "wait", "duration_seconds": 60, "reason": "Brief reason, e.g., 'Waiting for call response'" }}`
- `{{ "action_type": "think", "details": "Briefly what you are thinking about/planning" }}` (If no external action)
- `{{ "action_type": "update_goal", "new_goal": "Your new short-term goal" }}`
- `{{ "action_type": "end_interaction", "target": "partner_id_from_status", "reason": "Optional reason" }}` (To explicitly end a conversation/call)

**CRITICAL: Ensure the chosen action is the *single most sensible next step* based on your reflection, giving appropriate priority to ongoing interactions.**

Output ONLY the JSON object representing your final intent for this turn. This output will directly update the simulation state under the key '{intent_key}'.
"""
    intent_formation_agent = LlmAgent(
        name=f"Sim_{sim_id}_DecideIntent",
        model=llm_model,
        instruction=intent_formation_instruction,
        description=f"Forms the final intent for {persona_name}, considering interaction status.",
        output_key=intent_key,
        # response_mime_type="application/json" # Consider enabling if output is consistently JSON
    )

    # --- Create the Sequential Agent ---
    simulacra_sequential_agent = SequentialAgent(
        name=f"Simulacra_{sim_id}_ThinkingProcess",
        sub_agents=[
            observation_agent,
            reflection_agent,
            intent_formation_agent,
        ],
        description=f"Manages the internal planning sequence for {persona_name} ({sim_id})."
    )

    logger.info(f"Created V3 SequentialAgent '{simulacra_sequential_agent.name}' for {sim_id} with interaction status awareness.")
    return simulacra_sequential_agent

# --- Example Usage (_test_agent function needs update for new status keys) ---
async def _test_agent():
    # Need to define these here for the test scope
    # Make them global if needed by other functions in this file
    test_sim_id = "sim_test123"
    test_persona = {
        "Name": "Tester", "Age": 30, "Occupation": "Debugger",
        "Personality_Traits": ["Curious", "Logical", "Introverted"],
        "Long_Term_Aspirations": "Understand complex systems",
        "Background": "A test simulacrum.",
    }
    test_location_key = "TestLab"

    print("Testing Simulacra V3 Agent Creation & Execution (with Interaction Status)...")

    # Mock session with new status keys
    mock_state = {
        WORLD_STATE_KEY: {
            WORLD_TIME_KEY: "2024-01-01T09:00:00Z",
            LOCATION_DETAILS_KEY: { test_location_key: {"description": "A sterile white laboratory."} },
            "global_events": [], "weather": "Simulated - Clear"
        },
        WORLD_TEMPLATE_DETAILS_KEY: { "description": "A test simulation environment." },
        SIMULACRA_PROFILES_KEY: {
            test_sim_id: {
                PROFILE_PERSONA_KEY: test_persona,
                PROFILE_GOAL_KEY: "Verify agent functionality.",
                PROFILE_STATUS_KEY: { # Include new status fields
                    STATUS_CONDITION_KEY: "Normal",
                    STATUS_MOOD_KEY: "Focused",
                    "social_need": "Low",
                    "recent_events": ["Initialization complete"],
                    STATUS_INTERACTION_STATUS_KEY: "idle", # Start as idle
                    STATUS_INTERACTION_PARTNER_KEY: None,
                    STATUS_INTERACTION_MEDIUM_KEY: None,
                    STATUS_LAST_INTERACTION_SNIPPET_KEY: None
                },
                PROFILE_LOCATION_KEY: test_location_key
            }
        },
        SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=test_sim_id): None,
        # Initialize intermediate keys used by the sequential agent
        OBSERVE_OUTPUT_KEY: None,
        REFLECTION_OUTPUT_KEY: None,
    }
    mock_session = Session(id="test_session", user_id="test_user", app_name="test_app", state=mock_state)

    # Configure API Key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set.")
        return
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Create the agent
    agent = create_agent(test_sim_id, test_persona, mock_session)
    if not agent:
        print("Agent creation failed.")
        return

    print(f"Agent created: {agent.name}")
    print(f"Sub-agents: {[sub.name for sub in agent.sub_agents]}")

    # Mock Runner and SessionService for testing the flow
    mock_session_service = InMemorySessionService()
    mock_session_service.create_session(
        user_id=mock_session.user_id,
        app_name=mock_session.app_name,
        session_id=mock_session.id,
        state=mock_session.state
    )
    runner = Runner(agent=agent, app_name="test_app", session_service=mock_session_service)

    print("\n--- Running Test ---")
    trigger = types.Content(parts=[types.Part(text="Start your thinking process.")])
    final_intent = None
    async for event in runner.run_async(user_id=mock_session.user_id, session_id=mock_session.id, new_message=trigger):
        print(f"Event from {event.author}:")
        if event.content and event.content.parts:
             print(f"  Content: {event.content.parts[0].text}")
        if event.actions and event.actions.state_delta:
            print(f"  State Delta: {event.actions.state_delta}")
            # Check if the final intent key is in the delta (from the last sub-agent)
            intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=test_sim_id)
            if intent_key in event.actions.state_delta:
                final_intent = event.actions.state_delta[intent_key]
        if event.error_message: print(f"  Error: {event.error_message}")

    # Check final state
    final_session = mock_session_service.get_session(app_name="test_app", user_id="test_user", session_id="test_session")
    print("\n--- Final State ---")
    print(f"Observation (Intermediate): {final_session.state.get(OBSERVE_OUTPUT_KEY)}")
    print(f"Reflection (Intermediate): {final_session.state.get(REFLECTION_OUTPUT_KEY)}")
    print(f"Intent: {final_session.state.get(SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=test_sim_id))}")
    print(f"Final Status Dict: {final_session.state.get(SIMULACRA_PROFILES_KEY, {}).get(test_sim_id, {}).get(PROFILE_STATUS_KEY)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Define globals needed for the test function if running standalone
    # These are already defined within _test_agent, but defining here ensures they exist
    # if the test function relies on them being global (which it currently doesn't seem to)
    # test_sim_id = "sim_test123"
    # test_persona = {}
    # mock_session = None
    asyncio.run(_test_agent())
