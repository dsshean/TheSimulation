# src/agents/simulacra_v3.py

import logging
import os # Import os for getenv
from typing import Any, Dict, Optional
import asyncio
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.sessions import Session
from google.genai import types

# --- State Keys (Import or define centrally) ---
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent" # Final output key

# Temporary keys for planning steps
SIMULACRA_OBSERVATION_KEY_FORMAT = "sim_{}_step1_observation"
SIMULACRA_REFLECTION_KEY_FORMAT = "sim_{}_step2_reflection"

logger = logging.getLogger(__name__)

# --- Agent Factory ---

def create_agent(
    sim_id: str,
    persona: Dict[str, Any],
    session: Session, # Pass session for potential initial state access if needed
    model_name: Optional[str] = None # Allow overriding model
) -> SequentialAgent:
    """
    Factory function to create the V3 Simulacra agent.

    Adapts the detailed thinking process from the original instruction
    into a SequentialAgent: Observe -> Reflect -> Decide Intent.
    """
    llm_model = model_name or os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest")
    persona_name = persona.get("Name", sim_id)
    persona_traits = persona.get("Personality_Traits", [])
    long_term_aspirations = persona.get("Long_Term_Aspirations", "None specified") # Extract aspirations

    # Define keys specific to this simulacrum
    persona_key = SIMULACRA_PERSONA_KEY_FORMAT.format(sim_id)
    goal_key = SIMULACRA_GOAL_KEY_FORMAT.format(sim_id)
    status_key = SIMULACRA_STATUS_KEY_FORMAT.format(sim_id)
    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
    observation_key = SIMULACRA_OBSERVATION_KEY_FORMAT.format(sim_id)
    reflection_key = SIMULACRA_REFLECTION_KEY_FORMAT.format(sim_id)

    # --- Sub-Agent Definitions ---

    # 1. Observation Agent (Incorporating details from original Step 1 & 2)
    observation_agent = LlmAgent(
        name=f"Sim_{sim_id}_Observe",
        model=llm_model,
        instruction=f"""You are {persona_name} ({sim_id}). Your task is to observe your immediate surroundings and internal state with detail, internalizing who and where you are right now.
Consider:
- Time/Date/Weather: What time/day is it? How does the weather feel/look? (from {{{WORLD_STATE_KEY}}})
- Location: Your specific location and its characteristics (e.g., public/private, objects present). (from {{{location_key}}} and {{{WORLD_STATE_KEY}}})
- Social Context: Who else is nearby? What are they doing? (from {{{WORLD_STATE_KEY}}})
- Personal Status: Your physical condition (tired, hungry?), emotional state, social mood (lonely, sociable?). (from {{{status_key}}})
- Recent Events: Briefly recall any significant recent interactions or events mentioned in your status. (from {{{status_key}}})
- Sensory Details: Any notable sounds, smells, or sights?

Output ONLY a brief, factual observation (2-4 sentences) summarizing these points. Example: "It's late afternoon, raining lightly outside. I'm in the living room alone. Feeling a bit tired after work, but also restless. The TV is off."
""",
        description=f"Observes the current state for {persona_name}.",
        output_key=observation_key
    )

    # 2. Reflection Agent (Incorporating details from original Step 3)
    reflection_agent = LlmAgent(
        name=f"Sim_{sim_id}_Reflect",
        model=llm_model,
        instruction=f"""You are {persona_name} ({sim_id}). You just made an observation about your situation. Now, reflect deeply like a real person would, connecting it to your personality, goals, feelings, and recent experiences. **Embody your persona.**

Your Core Info:
- Persona: {{{persona_key}}} (Traits: {persona_traits}, Aspirations: {long_term_aspirations})
- Current Goal: {{{goal_key}}}
- Current Status/Mood/Social Need: {{{status_key}}}
- Your Observation: {{{observation_key}}}
- World Context (Time/Weather/News): {{{WORLD_STATE_KEY}}}

**Think Step-by-Step (Internal Monologue):**
1.  **React to Observation:** How does your current status (mood, physical state, social need) influence your reaction to the observation? How do recent events/interactions color your thoughts?
2.  **Evaluate Goal:** Is your current goal `{{{goal_key}}}` still relevant or achievable given the situation (time, weather, your status, recent events)? Does it align with your long-term aspirations (`{long_term_aspirations}`)? Do you feel motivated to pursue it *now*?
3.  **Consider Alternatives/Needs:** Based on your persona traits (`{persona_traits}`), feelings, social needs, or reactions to the situation/news/weather, are there other things you *want* or *need* to do right now? (e.g., contact someone, relax, address a physical need, react to news).
4.  **Prioritize:** What feels most important *right now*? Your goal? An immediate need? A social impulse? Reacting to the environment? Briefly weigh the options that came to mind.
5.  **Possible Pivot?:** If you strongly feel like changing your short-term goal based on this reflection, clearly state the *new goal* you're considering. (e.g., "Maybe my goal should be to call Sarah instead.")

Output ONLY your realistic, first-person internal monologue (3-6 sentences) reflecting this thought process, prioritization, and potential goal shift. Example: "Ugh, rain again. Makes going out less appealing. I *should* work on [Goal], but feeling restless makes it hard to focus. That chat with Bob earlier was awkward, maybe I should clear the air? Or just relax? Calling Sarah sounds nice, actually. Yeah, maybe connecting with her is more important right now than [Goal]."
""",
        description=f"Reflects on observations for {persona_name}.",
        output_key=reflection_key
    )

    # 3. Intent Formation Agent (Incorporating details from original Step 4 & 5)
    intent_formation_agent = LlmAgent(
        name=f"Sim_{sim_id}_DecideIntent",
        model=llm_model,
        instruction=f"""You are {persona_name} ({sim_id}). Based on your observation and reflection, decide your single, primary action/intent for this turn. Be plausible and in-character.

Your Reflection/Thoughts: {{{reflection_key}}}
Your Observation: {{{observation_key}}}
World Context (Time/Weather/Location/Nearby): {{{WORLD_STATE_KEY}}}
Your Location: {{{location_key}}}

**Decision Factors:**
- Your prioritized need/desire from your reflection.
- Plausibility: Is the action feasible given the time, location, weather, social context, your status?
- Goal Alignment: Does it progress your stated goal `{{{goal_key}}}`, address a temporary priority, or represent a shift in focus (potentially towards a new goal mentioned in your reflection)?

**Action Formatting:**
Choose ONE action type ('talk', 'move', 'use', 'wait', 'think', 'update_goal'). Format it as a JSON object.
- `{{ "action_type": "talk", "target": "npc_id", "content": "Message" }}` (Only if target is nearby)
- `{{ "action_type": "move", "destination": "Location Name" }}`
- `{{ "action_type": "use", "target": "object_id", "details": "Specific action, e.g., 'call Sarah', 'check news', 'make coffee'" }}` (Use for phone/computer interactions too)
- `{{ "action_type": "wait", "duration_seconds": 60, "reason": "Brief reason" }}`
- `{{ "action_type": "think", "details": "Briefly what you are thinking about/planning" }}` (If no external action)
- `{{ "action_type": "update_goal", "new_goal": "Your new short-term goal" }}` (ONLY if your reflection strongly indicated a goal change)

**CRITICAL: Ensure the chosen action is the *single most sensible next step* based on your reflection.**

Output ONLY the JSON object representing your final intent for this turn. This output will directly update the simulation state under the key '{intent_key}'.
""",
        description=f"Forms the final intent for {persona_name}.",
        output_key=intent_key
        # No response_mime_type needed
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

    logger.info(f"Created V3 SequentialAgent '{simulacra_sequential_agent.name}' for {sim_id}")
    return simulacra_sequential_agent

# --- Example Usage (Keep the existing _test_agent function for testing) ---
async def _test_agent():
    # ... (Keep the test function as it was, it will now use the updated prompts) ...
    # Need to define these here for the test scope
    global test_sim_id, test_persona, mock_session

    print("Testing Simulacra V3 Agent Creation...")
    # Mock data
    test_sim_id = "sim_test123"
    test_persona = {
        "Name": "Tester", "Age": 30, "Occupation": "Debugger",
        "Personality_Traits": ["Curious", "Logical", "Introverted"], # Added trait
        "Long_Term_Aspirations": "Understand complex systems", # Added aspiration
        "Background": "A test simulacrum.",
        "Initial_Goal": "Verify agent functionality." # Renamed for clarity
    }
    # Mock session with some initial state
    mock_state = {
        WORLD_STATE_KEY: {"world_time": "2024-01-01T09:00:00Z", "location_details": {"Kitchen": {"description": "A standard kitchen"}}, "global_events": [], "weather": "Clear"},
        SIMULACRA_PERSONA_KEY_FORMAT.format(test_sim_id): test_persona,
        SIMULACRA_GOAL_KEY_FORMAT.format(test_sim_id): test_persona["Initial_Goal"],
        SIMULACRA_STATUS_KEY_FORMAT.format(test_sim_id): {"condition": "Normal", "mood": "Focused", "social_need": "Low", "recent_events": ["Woke up"]},
        SIMULACRA_LOCATION_KEY_FORMAT.format(test_sim_id): "Kitchen",
        # Initialize temporary keys as None or empty
        SIMULACRA_OBSERVATION_KEY_FORMAT.format(test_sim_id): None,
        SIMULACRA_REFLECTION_KEY_FORMAT.format(test_sim_id): None,
        SIMULACRA_INTENT_KEY_FORMAT.format(test_sim_id): None,
    }
    mock_session = Session(id="test_session", user_id="test_user", app_name="test_app", state=mock_state)

    # Configure API Key (replace with your actual key loading)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set.")
        return
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Create the agent
    agent = create_agent(test_sim_id, test_persona, mock_session)
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
    async for event in runner.run_async(user_id=mock_session.user_id, session_id=mock_session.id, new_message=trigger):
        print(f"Event from {event.author}:")
        if event.content: print(f"  Content: {event.content.parts[0].text}")
        if event.actions and event.actions.state_delta: print(f"  State Delta: {event.actions.state_delta}")
        if event.error_message: print(f"  Error: {event.error_message}")

    # Check final state
    final_session = mock_session_service.get_session(app_name="test_app", user_id="test_user", session_id="test_session")
    print("\n--- Final State ---")
    print(f"Observation: {final_session.state.get(SIMULACRA_OBSERVATION_KEY_FORMAT.format(test_sim_id))}")
    print(f"Reflection: {final_session.state.get(SIMULACRA_REFLECTION_KEY_FORMAT.format(test_sim_id))}")
    print(f"Intent: {final_session.state.get(SIMULACRA_INTENT_KEY_FORMAT.format(test_sim_id))}")


if __name__ == "__main__":
    # Basic setup to run the test function
    logging.basicConfig(level=logging.INFO) # Use INFO level for less noise than DEBUG
    # Define mock variables needed for the test scope if not defined globally
    test_sim_id = "sim_test123"
    test_persona = {} # Will be defined in _test_agent
    mock_session = None # Will be defined in _test_agent
    asyncio.run(_test_agent())
