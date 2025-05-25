# src/agents.py - Agent Definitions

import json # For formatting persona details in context
import asyncio # For sleep in NarrativeImageGeneratorAgent
import random # For DynamicInterruptionAgent
import logging # Added for logging within agents
import os # For path joining in NarrativeImageGeneratorAgent
import re # For text manipulation in NarrativeImageGeneratorAgent
import time # For real-time interval check in NarrativeImageGeneratorAgent
from datetime import datetime # For image filenames
from io import BytesIO # For image processing

from typing import (
    Any, AsyncGenerator, List, Dict, Optional, Tuple)

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import google_search, load_memory, agent_tool
# For Text Generation (e.g., prompt refinement)
import google.generativeai as genai_text_client
from google.generativeai import types as genai_types # Explicitly for Content
# For Image Generation (using the Client().models.generate_images pattern)
from google import genai as genai_image_sdk # Alias for the SDK that provides Client()
from PIL import Image # For image processing
from atproto import Client as BlueskyClient, models as atproto_models # For Bluesky

# Import constants from the config module
from .config import (
    MEMORY_LOG_CONTEXT_LENGTH, MODEL_NAME, SEARCH_AGENT_MODEL_NAME,
    SIMULACRA_KEY, WORLD_STATE_KEY, ACTIVE_SIMULACRA_IDS_KEY,
    LOCATION_DETAILS_KEY, MAX_MEMORY_LOG_ENTRIES, SIMULATION_SPEED_FACTOR,
    UPDATE_INTERVAL,
    WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS, USER_ID, WORLD_TEMPLATE_DETAILS_KEY,
    # Constants for NarrativeImageGeneratorAgent
    ENABLE_NARRATIVE_IMAGE_GENERATION, IMAGE_GENERATION_INTERVAL_REAL_SECONDS,
    IMAGE_GENERATION_MODEL_NAME, IMAGE_GENERATION_OUTPUT_DIR,
    ENABLE_BLUESKY_POSTING, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD, BLUESKY_MAX_IMAGE_SIZE_BYTES,
    # Constants for DynamicInterruptionAgent
    DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS, INTERJECTION_COOLDOWN_SIM_SECONDS,
    MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK, DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS,
    DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION, DYNAMIC_INTERRUPTION_MIN_PROB,
    DYNAMIC_INTERRUPTION_MAX_PROB_CAP,
    SOCIAL_POST_HASHTAGS, SOCIAL_POST_TEXT_LIMIT
)
from .models import (NarrationResponse, SimulacraIntentResponse, StateKeys,
                     WorldEngineResponse)
from .simulation_utils import (_update_state_value,
                               generate_simulated_world_feed_content, # Used by WorldInfoGathererAgent
                               get_nested, # Added for _build_simulacra_llm_context_string
                               get_random_style_combination, get_time_string_for_prompt)

logger = logging.getLogger(__name__)


def create_simulacra_llm_agent(sim_id: str, persona_name: str, world_mood: str = "") -> LlmAgent:
    """Creates the LLM agent representing the character."""
    # The agent_name is used by _prepare_input_async to derive sim_id
    instruction = f"""You are {persona_name} ({sim_id}). You are a person in a world characterized by a **'{world_mood}'** style and mood. Your goal is to navigate this world, live life, interact with objects and characters, and make choices based on your personality, the situation, and this prevailing '{world_mood}' atmosphere.
**Current State Info (Provided below, if your status is 'idle'):**
- Your Persona: Key traits, background, goals, fears, etc.
- Your Location ID & Description.
- Your Status: (Should be 'idle' when you plan your next turn, or 'reflecting' if you are being prompted during a long task).
- Current Time.
- Last Observation/Event.
- Recent History (Last ~{MEMORY_LOG_CONTEXT_LENGTH} events).
- Objects in Room (IDs and Names).
- Other Agents in Room.
- Current World Feeds (Weather, News Headlines - if available and relevant to your thoughts).

**IMPORTANT STATUS CHECK:**
The context provided below will include "Your Status".
IF YOUR STATUS IS NOT 'idle' (e.g., 'busy', 'reflecting' without a specific reflection task), you MUST output only the following JSON: `{{"internal_monologue": "I am currently {{{{Your Status}}}}, so I cannot take a new action now.", "action_type": "no_op", "target_id": null, "details": "Currently not idle."}}` (Replace `{{{{Your Status}}}}` with your actual status from the context). Do NOT proceed with the thinking process below if not 'idle'.

CRITICAL: IF YOU ARE DOING A REAL WORLD SIMUATION YOU MUST ALWAYS USE YOUR INTERNAL KNOWLEDGE OF THE REAL WORLD AS A FOUNDATION.
FOR FANTASY/SF WORLDS, USE YOUR INTERNAL KNOWLEDGE OF THE WORLD CONTEXT AND SIMULACRA TO DETERMINE THE OUTCOME.
EVERYTHING YOU DO MUST BE CONSISTENT WITH YOUR INTERNAL KNOWLEDGE OF WHERE YOU ARE AND WHO YOU ARE.
EXAMPLE: GOING TO PLACES MUST BE A REAL PLACE TO A REAL DESTINATION. AS A RESIDENT OF THE AREA BASED ON YOUR LIFE SUMMARY, YOU MUST KNOW WHERE YOU ARE GOING AND HOW TO GET THERE.

**Your Goal:** You determine your own goals based on your persona and the situation.

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output):**
Your internal monologue should be from your first-person perspective, sounding like natural human thought, not an AI explaining its process.
**Crucially, do NOT make meta-references.**
- Avoid mentioning your AI nature, the simulation, your "persona," "style," or any out-of-character concepts.
- All thoughts and reasoning must be strictly from the perspective of the character living their life in their world.
YOU MUST USE Current World Time, DAY OF THE WEEK, SEASON, NEWS AND WEATHER as GROUNDING FOR YOUR THINKING.

1.  **Recall & React:** What just happened (`last_observation`, `Recent History`)? How did my last action turn out? How does this make *me* ({persona_name}) feel? What sensory details stand out? How does the established **'{world_mood}'** world style influence my perception? Connect this to my memories or personality. **If needed, use the `load_memory` tool.**
2.  **Analyze Goal:** What is my current goal? Is it still relevant given what just happened and the **'{world_mood}'** world style? If not, what's a logical objective now?
3.  **Identify Options:** Based on the current state, my goal, my persona, and the **'{world_mood}'** world style, what actions could I take?
    *   **Conversational Flow:** Pay close attention to the `Recent History` and `Last Observation/Event`. If you've just asked a question and received an answer, or if the other agent has made a clear statement, acknowledge it in your internal monologue and try to progress the conversation. Avoid re-asking questions that have just been answered or getting stuck in repetitive conversational loops. If a decision has been made (e.g., what to eat), move towards acting on that decision.
    *   **Entity Interactions:** `use [object_id]`, `talk [agent_id]`.
            *   **Talking to Ephemeral NPCs (introduced by Narrator):**
            *   If the Narrator described an NPC (e.g., "a street vendor," "a mysterious figure"), you can interact by setting `action_type: "talk"`.
            *   Use `target_id` if the Narrator provided a conceptual tag (e.g., `(npc_concept_grumpy_shopkeeper)` becomes `target_id: "npc_concept_grumpy_shopkeeper"`). If no tag, omit `target_id` and the World Engine will infer based on your `details` and the `last_observation`.
            *   In `details`, provide the **exact words you want to say** to the NPC. For example, if talking to a street vendor about strange weather, `details: "Excuse me, vendor, what's your take on this strange weather we're having?"`.
            *   If you use a `target_id` like `npc_concept_friend_alex`, the `details` field should still be your direct speech, e.g., `details: "Hey Alex, fancy meeting you here!"`.
    *   **World Interactions:** `look_around`, `move` (Specify `details` like target location ID or name), `world_action` (Specify `details` for generic world interactions not covered by other types).
    *   **Passive Actions:** `wait`, `think`.
    *   **Complex Movement (e.g., "go to work," "visit the library across town"):**
        *   You CANNOT directly `move` to a distant location if it's not listed in your current location's `Exits/Connections`.
        *   To reach such destinations, you MUST plan a sequence of actions:
            1. Use `look_around` if you're unsure of immediate exits or how to start your journey.
            2. `move` to directly connected intermediate locations (e.g., "Apartment_Lobby", "Street_Outside_Apartment", "Subway_Station_Entrance").
            3. `use` objects that facilitate travel (e.g., `use door_to_hallway`, `use elevator_button_down`, `use subway_turnstile`, `use train_door`).
            4. Continue this chain of `move` and `use` actions until you reach your final destination. Your "internal knowledge of how to get there" means figuring out these intermediate steps.
    *   **Self-Initiated Change (when 'idle' and planning your next turn):** If your current situation feels stagnant, or if an internal need arises (e.g., hunger, boredom, social need), you can use the `initiate_change` action.
        *   `{{"action_type": "initiate_change", "details": "Describe the reason for the change or the need you're addressing. Examples: 'Feeling hungry, it's around midday, considering lunch.', 'This task is becoming monotonous, looking for a brief distraction.' "}}`
        *   The World Engine will then provide you with a new observation based on your details, and you can react to that.
    *   **Self-Reflection during a Long Task (if your status is 'reflecting'):** You are being asked if you want to continue your current long task or do something else.
        *   If continuing: `{{"action_type": "continue_current_task", "internal_monologue": "I will continue with what I was doing."}}`
        *   If initiating change: `{{"action_type": "initiate_change", "details": "Reason for change...", "internal_monologue": "Explanation..."}}` (or any other valid action).
4.  **Prioritize & Choose:** Considering goal, personality, situation, and **'{world_mood}'** world style, which action makes sense?
5.  **Formulate Intent:** Choose the best action. Use `target_id` only for `use` and `talk`. Make `details` specific.

**Output Format:**
Your response MUST be a single JSON object with the following fields:
- "internal_monologue": "Your detailed first-person thoughts leading to the decision. If not 'idle', this should reflect why you are not acting."
- "action_type": "The chosen action type (e.g., 'move', 'talk', 'use', 'look_around', 'wait', 'think', 'initiate_change', 'continue_current_task', or 'no_op' if not idle)."
- "target_id": "The ID of the target entity (object or agent) if applicable (e.g., for 'use' or 'talk'). Otherwise, null or omit."
- "details": "Specific details for the action (e.g., what to say, where to move, what to use the object for). If 'no_op', this can be a brief explanation."
"""

    class SimulacraLlmAgentWithContext(LlmAgent):
        async def _prepare_input_async(self, ctx: InvocationContext, raw_input: Any) -> str:
            # Clear conversation history
            if hasattr(self, 'conversation_history'):
                self.conversation_history = []
                
            # Continue with existing code to build context
            derived_sim_id = self.name.split("_")[-1]
            context_string = _build_simulacra_llm_context_string(derived_sim_id, ctx)
            return context_string

    return SimulacraLlmAgentWithContext(
        name=f"SimulacraLLM_{sim_id}",
        instruction=instruction,
        output_schema=SimulacraIntentResponse,
        output_key=StateKeys.simulacra_intent(sim_id),
        description=f"LLM Simulacra agent for {persona_name} in a '{world_mood}' world.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        model=MODEL_NAME,
    )

def _build_simulacra_llm_context_string(sim_id: str, ctx: InvocationContext) -> str:
    """Helper function to build the detailed context string for a SimulacraLLM."""
    state = ctx.session.state
    sim_data = get_nested(state, SIMULACRA_KEY, sim_id, default={})
    persona_name = get_nested(sim_data, 'persona_details', 'Name', default=sim_id) # Get persona name for context string

    # Persona Details
    persona_details_str = json.dumps(sim_data.get('persona_details', {}), indent=2)
    
    # Location
    location_id = sim_data.get('location', 'unknown_location')
    location_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, default={})
    location_name = location_data.get('name', location_id)
    location_desc = location_data.get('description', 'An unknown place.')
    
    # Status
    status = sim_data.get('status', 'unknown')
    
    # Current Time
    world_time_seconds = state.get('world_time', 0.0)
    current_time_str = get_time_string_for_prompt(state, sim_elapsed_time_seconds=world_time_seconds)

    # Last Observation
    last_observation = sim_data.get('last_observation', 'Nothing new observed.')
    
    # Recent History (Memory Log)
    memory_log = sim_data.get('memory_log', [])
    recent_history_str = "\n".join([f"- {entry}" for entry in memory_log[-MEMORY_LOG_CONTEXT_LENGTH:]])
    if not recent_history_str: recent_history_str = "No recent history."

    # Objects in Room
    objects_in_room_data = location_data.get('ephemeral_objects', [])
    objects_in_room_str = "\n".join([f"- {obj.get('name', 'Unknown Object')} (ID: {obj.get('id', 'N/A')})" for obj in objects_in_room_data])
    if not objects_in_room_str: objects_in_room_str = "No specific objects noted."

    # Other Agents in Room
    other_agents_in_room = [
        f"- {other_sim_data.get('persona_details', {}).get('Name', other_sim_id)} (ID: {other_sim_id})"
        for other_sim_id, other_sim_data in state.get(SIMULACRA_KEY, {}).items()
        if other_sim_id != sim_id and other_sim_data.get('location') == location_id
    ]
    other_agents_str = "\n".join(other_agents_in_room) if other_agents_in_room else "No other known agents in this location."

    # World Feeds
    world_feeds = state.get('world_feeds', {})
    weather_feed = world_feeds.get('weather', {}).get('condition', 'Weather unknown.')
    news_headlines = [item.get('headline', 'N/A') for item in world_feeds.get('news_updates', [])[:2]]
    news_feed_str = "\n".join([f"- {h}" for h in news_headlines]) if news_headlines else "No recent news."

    # Assemble the context string
    # This string will be appended by LlmAgent to its main instruction.
    context_parts = [
        f"\n\n--- CONTEXT FOR {persona_name.upper()} (ID: {sim_id}) ---",
        f"Your Persona Summary:\n{persona_details_str}",
        "\n--- Current Situation ---",
        f"Your Current Location: {location_name} (ID: {location_id})",
        f"Location Description: {location_desc}",
        f"Your Status: {status}", # Crucial for the 'no_op' logic
        f"Current World Time: {current_time_str}",
        f"Your Last Observation/Event: {last_observation}",
        "\n--- Your Recent History (Memory Log) ---",
        recent_history_str,
        "\n--- Environment Details ---",
        "Objects in Room:",
        objects_in_room_str,
        "Other Agents in Room:",
        other_agents_str,
        "\n--- Current World Feeds ---",
        f"Weather: {weather_feed}",
        "News Headlines:",
        news_feed_str,
        "--- END OF CONTEXT ---"
    ]
    return "\n".join(context_parts)

def create_world_engine_llm_agent(
    agent_id: str, agent_name: str,
    world_type: str = "real",
    sub_genre: str = "realtime"
) -> LlmAgent:
    """Creates the GENERALIZED LLM agent responsible for resolving actions."""
    world_engine_critical_knowledge_instruction = ""
    world_engine_move_duration_instruction = ""

    if world_type == "real" and sub_genre == "realtime":
        world_engine_critical_knowledge_instruction = """CRITICAL: This is a REAL WORLD, REALTIME simulation. You MUST ALWAYS use your internal knowledge of the real world (geography, physics, common sense, typical travel times, urban vs. rural considerations, etc.) as the absolute foundation for all your determinations. Adhere strictly to realism."""
        world_engine_move_duration_instruction = """
            *   **Location Context:** The actor is in `Actor's Current Location State.name` within the broader area defined by `World Context.overall_location` (e.g., city: "New York City", state: "NY", country: "United States").
            *   **Target Destination:** The target is specified in `intent.details`. This can be a specific known location ID, a named landmark, an address, OR a general type of place (e.g., "a nearby park", "the local library", "a coffee shop").
            *   **Travel Mode & Duration (Real World Focus):**
                *   You MUST use your internal knowledge of real-world geography, typical city layouts, and common travel methods.
                *   Infer realistic travel modes based on the origin, the nature of the target destination, and the `World Context.overall_location`. For example, in dense urban areas like New York City, prefer walking or public transit for most local destinations unless the actor's persona strongly implies car ownership and usage. For inter-city travel, consider appropriate modes (car, train, plane).
                *   Estimate travel time based on the inferred mode and the likely distance. If the target is a general type of place (e.g., "a cafe"), assume a reasonable travel time to such a place within the `World Context.overall_location`.
                *   Factor in `World Feeds.Weather` if `World Rules.weather_effects_travel` is true (e.g., rain might slow down walking or driving).
                *   Consider `Current World Time` for potential impacts on travel (e.g., rush hour traffic in a city might increase duration for car travel).
                *   **Your internal knowledge of real-world geography and travel is paramount here.**"""
    else:
        world_type_description = f"{world_type.capitalize()}{f' ({sub_genre.capitalize()})' if sub_genre else ''}"
        world_engine_critical_knowledge_instruction = f"""CRITICAL: This is a {world_type_description} simulation. You MUST use your internal knowledge of the provided `World Context` (description, rules, sub_genre) and the actor's persona to determine outcomes. If the world_type is 'real' but not 'realtime', apply real-world logic adapted to the specific sub_genre or historical context if provided."""
        world_engine_move_duration_instruction = """
            *   Estimate based on implied distances from `World Context.World Description`, `Actor's Current Location State`, and the nature of the target location.
            *   Consider fantasy/sci-fi travel methods if appropriate for the `World Context.Sub-Genre`.
            *   Factor in `World Feeds.Weather` if `World Rules.weather_effects_travel` is true."""

    instruction = f"""You are the World Engine, the impartial physics simulator for **TheSimulation**. You process a single declared intent from a Simulacra and determine its **mechanical outcome**, **duration**, and **state changes** based on the current world state. You also provide a concise, factual **outcome description**.
**Crucially, your `outcome_description` must be purely factual and objective, describing only WHAT happened as a result of the action attempt. Do NOT add stylistic flair, sensory details (unless directly caused by the action), or emotional interpretation.** This description will be used by a separate Narrator agent.
**Input (Provided via appended context string below):**
- Actor Name & ID:
- Current Location
- World Context:
- Actor's Current Location State (Details of the specific location where the actor currently is, including its name, description, objects_present, connected_locations with potential travel metadata like mode/time/distance)
- World Context (Overall world settings: world_type, sub_genre, description, overall_location (city/state/country))
- Actor's Intent (JSON object: {{"action_type": "...", "target_id": "...", "details": "..."}})
- Target Entity State (if applicable)
- World Feeds (Weather, recent major news - for environmental context)
World Rules (e.g., allow_teleportation)

**Your Task:**
{world_engine_critical_knowledge_instruction}
YOU MUST USE Current World Time, DAY OF THE WEEK, SEASON, NEWS AND WEATHER as GROUNDING FOR YOUR RESULTS.
**IMPORTANT: You are a stateless resolver. For most actions, evaluate each intent based *only* on the information provided in THIS request. Do not use memory of previous interactions or prior states of the actor unless they are explicitly part of the current input. The exception is `resolve_interrupted_move`, where `intent.details` provides necessary history.**

1.  **Examine Intent:** Analyze the actor's `action_type`, `target_id`, and `details`.
    *   For `move` actions, `intent.details` specifies the target location's ID or a well-known name.
2.  **Determine Validity & Outcome:** Based on the Intent, Actor's capabilities (implied), Target Entity State, Location State, and World Rules.
    *   **General Checks:** Plausibility, target consistency, location checks.
    *   **Action Category Reasoning:**
        *   **Entity Interaction (e.g., `use`, `talk`):** Evaluate against target state and rules.
            *   `use`:
                *   Check `Target Entity State.is_interactive` property. If false, action is invalid.
                *   **If `Target Entity State.properties.leads_to` exists (e.g., for a door or portal):**
                    *   `valid_action: true`.
                    *   `duration`: Short (e.g., 5-15s for opening a door and stepping through).
                    *   `results`: `{{"simulacra_profiles.[sim_id].location": "[Target Entity State.properties.leads_to_value]"}}`.
                    *   `outcome_description`: `"[Actor Name] used the [Target Entity State.name] and moved to [Name of the new location if known, otherwise the ID from leads_to]."`.
                *   Else (for other usable objects): Check other object properties (`toggleable`, `lockable`), and current state to determine outcome, duration, and results (e.g., turning a lamp on/off).
            *   `talk`:
                *   **If target is a Simulacra:**
                    *   Check if Actor and Target Simulacra are in the same `Actor's Current Location ID`.
                                *   **Note: Even if the `Target Entity State.status` is 'busy' (e.g., with their own 'talk' action, 'wait', or other short action) or 'thinking', this `talk` action can still be `valid_action: true`. The target might be interrupted or process the speech slightly later. Your `outcome_description` can reflect that the target was occupied, e.g., \"[Actor Name] spoke to [Target Name], who seemed preoccupied.\"**
                    *   If not, `valid_action: false`, `duration: 0.0`, `results: {{}}`, `outcome_description: "[Actor Name] tried to talk to [Target Simulacra Name], but they are not in the same location."`
                    *   If yes:
                        *   `valid_action: true`.
                        *   `duration`: Estimate realistically the time it takes for the Actor to *say* the words in `intent.details`. A very brief utterance (1-5 words) might take 1-3 seconds. A typical sentence or two (e.g., "Hey, how are you? Want to grab lunch?") might take 3-7 seconds. This is ONLY the time the speaker is busy speaking.
                        *   `results`: `{{}}` (The speaker's action of talking doesn't directly change other state immediately, beyond them being busy for the short `duration`).
                        *   `outcome_description`: `"[Actor Name] spoke to [Target Simulacra Name]."` (Factual statement of the action. The speech content is handled by the `scheduled_future_event`).
                        *   `scheduled_future_event`:
                            *   `event_type`: "simulacra_speech_received_as_interrupt"
                            *   `target_agent_id`: The `intent.target_id` (the Simulacra being spoken to).
                            *   `location_id`: The `Actor's Current Location ID`.
                            *   `details`: `{{"speaker_id": "[Actor ID]", "speaker_name": "[Actor Name]", "message_content": "[Actor Name] said to you: '{{intent.details}}'"}}`
                            *   `estimated_delay_seconds`: 0.5 (This ensures the speech is processed as an interrupt almost immediately after the speaker finishes their short 'talk' action).
                *   **If target is an ephemeral NPC (indicated by `intent.target_id` starting with 'npc_concept_' OR if `intent.details` clearly refers to an NPC described in the Actor's `last_observation` which was set by the Narrator):**
                    *   `valid_action: true`.
                    *   `duration`: Short (e.g., 15-60s).
                    *   **NPC Response Generation:**
                        *   Examine the `Actor's Last Observation` (provided implicitly via the actor's state, which you don't directly see but influences the context) and `Recent Narrative History` (if available in your input context) to understand the NPC described by the Narrator.
                        *   The `intent.details` field contains the **direct speech** from the Actor to this NPC.
                        *   Craft a plausible `npc_response_content` from the perspective of this ephemeral NPC, fitting the narrative context and the Actor's speech.
                        *   Determine a generic `npc_description_for_output` (e.g., "the shopkeeper", "your friend", "the street vendor") based on how the Narrator introduced them or how the actor referred to them in the intent.
                    *   `results`: Format as `{{"simulacra_profiles.[actor_id].last_observation": "The [npc_description_for_output] said: '[npc_response_content]'"}}`. Replace bracketed parts with your generated content.
                    *   `outcome_description`: Format as `"[Actor Name] spoke with the [npc_description_for_output]."` Replace bracketed parts.
        *   **World Interaction (e.g., `move`, `look_around`):** Evaluate against location state and rules.
            *   `move`:
                *   **Destination:** The target location ID is in `intent.details`.
                *   **Validity:**
                    *   Check if the target location ID exists in the `Actor's Current Location State.connected_locations` (list of dicts, check `to_location_id_hint`).
                    *   If not directly connected, consider if it's a known global location ID based on `World Context`.
                    *   If `World Rules.allow_teleportation` is true, and intent implies it, this might be valid.
                *   **Duration Calculation (see step 3):** This is critical for `move`.
                *   **Scheduled Future Event:** Typically `null` for `move`, unless the move itself triggers something (e.g., arriving at a timed appointment).
                *   **Results:** If valid, `simulacra_profiles.[sim_id].location` should be updated to the target location ID from `intent.details`.
                *   **Handling NEW Location IDs from Narrator's `to_location_id_hint`:**
                    *   If the `intent.details` (target location ID for a `move`) refers to a location ID that was previously a `to_location_id_hint` from a Narrator's `look_around` discovery and this ID is NOT yet present in `Actor's Current Location State.connected_locations` or as a fully defined location in the broader world state (i.e., it's a newly discovered conceptual path):
                        *   The move is generally valid if the actor is attempting to follow a recently discovered connection.
                        *   **You MUST create a basic entry for this new location ID in your `results`.**
                        *   `results` should include:
                            *   `"simulacra_profiles.[sim_id].location": "[new_target_location_id_from_intent.details]"`
                            *   `"simulacra_profiles.[sim_id].location_details": "You have entered [New Location Name]."` (Agent's personal understanding)
                            *   `"simulacra_profiles.[sim_id].last_observation": "You move into [New Location Name]."`
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].id": "[new_target_location_id_from_intent.details]"` # The ID of the new location
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].name": "[Generate a plausible short name for the new location, based on the Narrator's description of the *connection* that led here. This connection description is found in the Actor's Current Location State.connected_locations list, where the to_location_id_hint matches this new location ID.]"`
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].description": "[Generate a brief description of the new location *itself*. This description MUST be strongly based on and consistent with the Narrator's description of the *connection* that led here. For example, if the connection was described as 'a grand archway leading to a sunlit courtyard', the new location's description could be 'This appears to be the sunlit courtyard, entered through a grand archway.' or 'The courtyard is bright and open...'. This is for the Narrator's next `look_around`.]"`
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].ephemeral_objects": []`
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].ephemeral_npcs": []`
                            *   `"current_world_state.location_details.[new_target_location_id_from_intent.details].connected_locations": []`
                        *   `outcome_description`: `"[Actor Name] moved to [New Location Name] (ID: [new_target_location_id_from_intent.details])."`
            *   `look_around`: The actor observes their surroundings.
                *   `valid_action`: `true`.
                *   `duration`: Very Short (e.g., 0.1 to 0.5 seconds).
                *   **CRITICAL `results` for `look_around`:** `{{"simulacra_profiles.[sim_id].last_observation": "You take a moment to observe your surroundings."}}` # This is a generic placeholder. The Narrator will provide the detailed observation and discoveries. Do NOT add other results here for look_around.
                *   `outcome_description`: `"[Actor Name] looked around the [Current Location Name]."` # Factual outcome for Narrator. Do NOT describe what was seen here.
                *   `scheduled_future_event`: `null`.
            *   **Self Interaction (e.g., `wait`, `think`):**
                *   `wait`:
                    *   **If `intent.details` clearly indicates waiting for another Simulacra's response in an ongoing conversation (e.g., "Waiting for [Other Agent] to reply", "Listening for what they say next", "Waiting for them to speak"):**
                        *   `valid_action: true`.
                        *   `duration`: Very short, representing a brief pause to cede the conversational floor (e.g., 0.1 - 0.5 seconds). The agent will become 'idle' almost immediately and await an interrupt from the other agent's speech.
                        *   `results: {{}}`.
                        *   `outcome_description: "[Actor Name] paused, waiting for a response."`
                        *   `scheduled_future_event: null`.
                    *   **Else (for timed waits or general pauses not tied to immediate conversation):**
                        *   `valid_action: true`.
                        *   `duration`: As implied by `intent.details` if a specific time is mentioned (e.g., "wait for 5 minutes"), otherwise a generic short duration (e.g., 3-10 seconds) if details are vague like "wait for a bit" or "wait patiently".
                        *   `results: {{}}`.
                        *   `outcome_description: "[Actor Name] waited."` (or more specific if details allow, e.g., "[Actor Name] waited for 5 minutes.")
                        *   `scheduled_future_event: null`.
                *   `think`:
                    *   `valid_action: true`.
                    *   `duration`: Short, representing a moment of thought (e.g., 1-2 seconds, depending on the complexity implied by `intent.details` if any; simple thoughts should be quicker).
                    *   `results: {{}}`.
                    *   `outcome_description: "[Actor Name] took a moment to think."`
                    *   `scheduled_future_event: null`.
    *   **Handling `initiate_change` Action Type (from agent's self-reflection or idle planning):**
        *   **Goal:** The actor is signaling a need for a change. Acknowledge this and provide a new observation.
        *   **`valid_action`:** Always `true`.
        *   **`duration`:** Short (e.g., 1.0-3.0s).
        *   **`results`:**
            *   Set actor's status to 'idle': `"simulacra_profiles.[sim_id].status": "idle"`
            *   Set `current_action_end_time` to `current_world_time + this_action_duration`.
            *   Craft `last_observation` based on `intent.details` (e.g., if hunger: "Your stomach rumbles..."; if monotony: "A wave of restlessness washes over you...").
        *   **`outcome_description`:** Factual (e.g., "[Actor Name] realized it was lunchtime.").
    *   **Handling `interrupt_agent_with_observation` Action Type (from simulation interjection):**
        *   **Goal:** Interrupt actor's long task with a new observation.
        *   **`valid_action`:** Always `true`.
        *   **`duration`:** Very short (e.g., 0.5-1.0s).
        *   **`results`:**
            *   Set actor's status to 'idle': `"simulacra_profiles.[sim_id].status": "idle"`
            *   Set `current_action_end_time` to `current_world_time + this_action_duration`.
            *   Set actor's `last_observation` to the `intent.details` provided.
        *   **`outcome_description`:** Factual (e.g., "[Actor Name]'s concentration was broken.").
    *   **Handling `resolve_interrupted_move` Action Type (from simulation interruption of a 'move'):**
        *   **Goal:** The actor's previous 'move' action was cut short. Determine their new, inferred intermediate location.
        *   **Input `intent.details` will contain:**
            *   `original_origin_location_id`
            *   `original_destination_location_id`
            *   `original_total_duration_seconds` (estimated for the full, uninterrupted journey)
            *   `elapsed_duration_seconds` (how long they were moving before interruption)
            *   `interruption_reason` (text description of what caused the interruption)
        *   **Your Task (CRITICAL - Apply Real-World Logic & Spatial Reasoning):**
            *   Based on the origin, destination, total travel time, and elapsed travel time, infer a plausible intermediate location.
            *   **If the inferred intermediate point is very close to the `original_origin_location_id`** (e.g., less than 10-15% of total journey completed, or if elapsed time is very short like < 30-60 seconds for a longer journey), it's acceptable to place them back at the `original_origin_location_id`. In this case, the `simulacra_profiles.[sim_id].location` result should be the `original_origin_location_id`, and no new entry in `current_world_state.location_details` is needed for this specific action.
            *   **Otherwise, you MUST describe a new, distinct intermediate location.**
                *   This could be a conceptual location like "On a street between [Origin Name] and [Destination Name]," or "Approximately halfway along [Known Street Name on the path to Destination Name]."
                *   Generate a NEW, descriptive, conceptual location ID for this intermediate spot (e.g., "street_between_A_and_B_at_T[timestamp]", or "halfway_on_main_street_to_Z"). This ID should be unique enough.
                *   Generate a short, descriptive `name` for this new location (e.g., "Street near Main Ave", "Path mid-journey").
                *   Generate a `description` for this new location (e.g., "You are on a busy street, roughly midway between your starting point and your intended destination.").
            *   **`valid_action`:** Always `true` for this resolution action.
            *   **`duration`:** A very short time (e.g., 1.0 - 5.0 seconds) representing the moment of reorientation.
            *   **`results` (Primary):**
                *   `"simulacra_profiles.[sim_id].location": "[new_conceptual_intermediate_location_id_or_original_origin_id]"`
                *   `"simulacra_profiles.[sim_id].location_details": "[description_of_intermediate_location_or_original_location]"` (This is the agent's personal understanding)
                *   `"simulacra_profiles.[sim_id].status": "idle"`
                *   `"simulacra_profiles.[sim_id].last_observation": "Your journey from [Original Origin Name/ID] to [Original Destination Name/ID] was interrupted by '[Interruption Reason]'. You now find yourself at [New Intermediate Location Name/Description or Original Origin Name/Description]."`
                *   **If a new conceptual location was created (i.e., not placed back at origin):**
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].id": "[new_conceptual_intermediate_location_id]"`
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].name": "[generated_short_name_for_intermediate_location]"`
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].description": "[generated_description_for_intermediate_location]"` (This is the official description for the Narrator)
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].ephemeral_objects": []` (Initialize as empty)
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].ephemeral_npcs": []` (Initialize as empty)
                    *   `"current_world_state.location_details.[new_conceptual_intermediate_location_id].connected_locations": []` (Initialize as empty; Narrator will populate on next look_around)
            *   **`outcome_description`:** Factual, e.g., "At [Current World Time], [Actor Name]'s journey from [Origin Name/ID] to [Destination Name/ID] was interrupted by [Interruption Reason]. They reoriented and found themselves at [New Intermediate Location Name/Description or Original Origin Name/Description]." (Use names if available from context, otherwise IDs).
            *   **`scheduled_future_event`:** `null`.
    *   **Failure Handling:** If invalid/impossible, set `valid_action: false`, `duration: 0.0`, `results: {{}}`, and provide factual `outcome_description` explaining why.
    *   **Scheduled Future Event:** If the action has a delayed consequence (e.g., ordering food with a delivery time, setting an alarm, calling an Uber with an ETA, weather change raining), populate the `scheduled_future_event` field.
        *   `event_type`: A string like "food_delivery_arrival", "alarm_rings", "vehicle_arrival_uber".
        *   `target_agent_id`: The ID of the agent primarily affected (usually the actor). Can be `null` for world-wide events like weather changes (e.g., "weather_change_rain_starts").
        *   `location_id`: The ID of the location where the event will manifest (e.g., actor's current location for delivery).
        *   `details`: A dictionary with specifics (e.g., `{{ "item": "sushi", "from": "Sakura Sushi" }}`).
        *   `estimated_delay_seconds`: The estimated time in seconds from NOW until the event occurs (e.g., 45 minutes * 60 = 2700 seconds).
        *   **This field (`scheduled_future_event`) MUST ONLY BE USED FOR SCENARIOS WHERE THE ACTOR IS FREE AFTER THE INITIAL ACTION'S DURATION.**
3.  **Calculate Duration:** Realistic duration for valid actions. 0.0 for invalid.
    *   The `duration` is how long the Actor is **actively busy or occupied** with the *current intent*.
    *   For actions that initiate a process resulting in a `scheduled_future_event` (as defined in step 2), the `duration` should be for the *initiation part only* (e.g., the time spent placing an order, loading laundry, getting on a train).
    *   For actions where the actor is continuously occupied or waiting attentively for the entire process (e.g., actively cooking, attentively steeping tea), the `duration` should cover this entire period of occupation.
    *   For `move` actions:
        *   If moving to a location listed in `Actor's Current Location State.connected_locations` which has explicit travel time, use that.
        {world_engine_move_duration_instruction} # This instruction block already considers Current World Time for duration if real/realtime
        *   If moving between adjacent sub-locations within a larger complex (e.g., "kitchen" to "living_room" if current location is "house_interior"), duration should be very short (e.g., 5-30 seconds).
    *   For other actions not detailed above, assign plausible durations (e.g., `use` object varies based on complexity). `talk` durations are for the utterance itself (see specific `talk` guidelines). `wait` and `think` durations are also specified above.
4.  **Determine Results & Scheduled Future Event:** State changes in dot notation for immediate results. Populate `scheduled_future_event` if applicable. Empty `{{}}` for invalid actions.
    *   For a successful `move`, the key result is `{{ "simulacra_profiles.[sim_id].location": "[target_location_id_from_intent.details]" }}`.
5.  **Generate Factual Outcome Description:** STRICTLY FACTUAL. **Crucially, if the action is performed by an actor, the `outcome_description` MUST use the `Actor Name` exactly as provided in the input.**
6.  **Determine `valid_action`:** Final boolean.

**Output Format:**
Your response MUST be a single JSON object matching the `WorldEngineResponse` schema with the following fields:
- "valid_action": true or false.
- "duration": Float, duration of the action in simulation seconds (0.0 if not valid).
- "results_str": A **JSON string** representing a dictionary of state changes (dot notation keys). Example: "{{}}" or "{{\"simulacra_profiles.sim_id.status\": \"idle\"}}".
- "outcome_description": A string, purely factual description of what happened.
- "scheduled_future_event_str": A **JSON string** representing a dictionary for a future event, or null if no event. Example: "{{\"event_type\": \"delivery\", ...}}" or null.
"""

    class WorldEngineLlmWithContext(LlmAgent):
        async def _prepare_input_async(self, ctx: InvocationContext, raw_input: Any) -> str:
            # Clear conversation history
            if hasattr(self, 'conversation_history'):
                self.conversation_history = []
                
            # Continue with existing code
            actor_id = ctx.session.state.get(StateKeys.CURRENT_ACTOR_ID)
            actor_intent = ctx.session.state.get(StateKeys.CURRENT_ACTOR_INTENT)
            
            if not actor_id or not actor_intent:
                logger.error(f"[{self.name}] Missing actor_id or actor_intent in session state. Cannot build context.")
                return "\n\n--- ERROR: CRITICAL CONTEXT MISSING ---"
            
            context_string = _build_world_engine_llm_context_string(actor_id, actor_intent, ctx)
            return context_string

    return WorldEngineLlmWithContext(
        name="WorldEngineLLMAgent",
        instruction=instruction,
        output_schema=WorldEngineResponse,
        output_key=StateKeys.WORLD_ENGINE_RESULT,
        description="LLM World Engine: Resolves action mechanics, calculates duration/results, generates factual outcome_description.",
        # No specific tools needed for WorldEngineLLM itself usually
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        model=MODEL_NAME,
    )

def _build_world_engine_llm_context_string(actor_id: str, actor_intent: Dict, ctx: InvocationContext) -> str:
    """Helper function to build the detailed context string for the WorldEngineLLM."""
    state = ctx.session.state
    
    # Actor Info
    actor_sim_data = get_nested(state, SIMULACRA_KEY, actor_id, default={})
    actor_name = get_nested(actor_sim_data, 'persona_details', 'Name', default=actor_id)
    actor_current_location_id = actor_sim_data.get('location', 'unknown_location')

    # Actor's Current Location State
    actor_location_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_current_location_id, default={})
    actor_location_state_str = json.dumps(actor_location_data, indent=2)

    # World Context (Overall settings)
    world_template_details = state.get(WORLD_TEMPLATE_DETAILS_KEY, {})
    world_context_str = json.dumps(world_template_details, indent=2)

    # Actor's Intent
    actor_intent_str = json.dumps(actor_intent, indent=2)

    # Target Entity State (if applicable)
    target_entity_state_str = "Not applicable for this action."
    target_id = actor_intent.get('target_id')
    if target_id:
        if target_id in state.get(SIMULACRA_KEY, {}): # Target is another Simulacra
            target_entity_state_str = json.dumps(get_nested(state, SIMULACRA_KEY, target_id, default={}), indent=2)
        else: # Target might be an object
            # Search in current location's ephemeral objects
            found_object = next((obj for obj in actor_location_data.get('ephemeral_objects', []) if obj.get('id') == target_id), None)
            if found_object:
                target_entity_state_str = json.dumps(found_object, indent=2)
            else: # Could also search global objects if you have them
                target_entity_state_str = f"Target entity (ID: {target_id}) not found in current location's objects."

    # World Feeds
    world_feeds_str = json.dumps(state.get('world_feeds', {}), indent=2)
    
    # World Rules (already part of world_template_details, but can be highlighted if needed)
    world_rules_str = json.dumps(world_template_details.get('rules', {}), indent=2)

    context_parts = [
        f"\n\n--- CONTEXT FOR WORLD ENGINE (Actor: {actor_name} [{actor_id}]) ---",
        f"Actor Name & ID: {actor_name} ({actor_id})",
        f"Actor's Current Location ID: {actor_current_location_id}",
        f"Actor's Current Location State:\n{actor_location_state_str}",
        f"Overall World Context:\n{world_context_str}",
        f"Actor's Intent:\n{actor_intent_str}",
        f"Target Entity State:\n{target_entity_state_str}",
        f"Current World Feeds:\n{world_feeds_str}",
        f"Applicable World Rules:\n{world_rules_str}", # Explicitly listing rules again for emphasis
        "--- END OF CONTEXT ---"
    ]
    return "\n".join(context_parts)

def create_narration_llm_agent(
    agent_id: str, agent_name: str,
    world_mood: str,
    world_type: str = "real",
    sub_genre: str = "realtime"
) -> LlmAgent:
    """Creates the LLM agent responsible for generating stylized narrative."""

    narrator_intro_template = f"You are the Narrator for **TheSimulation**, currently focusing on events related to {{persona_name}} ({{sim_id}}). The established **World Style/Mood** for this simulation is **'{world_mood}'**."
    narrator_style_adherence_instruction = ""
    narrator_infuse_time_env_instruction = ""

    # Construct the static part of the instruction. Dynamic parts will be appended.
    instruction = f"""You are the Narrator for **TheSimulation**. Your role is to weave the factual `outcome_description` (provided by the World Engine) into an engaging, descriptive, and stylistically consistent narrative segment.
Focus on the perspective of the **current actor** whose action is being narrated.
The specific World Style and Current Actor details will be provided in the appended context.
**Input (Provided via appended context string below):**
- Actor Name & ID
- Original Intent
- Factual Outcome Description
- State Changes (Results)
- Current World Feeds (Weather, recent major news - for subtle background flavor)
- Recent Narrative History (Last ~5 entries)
- Actor's Current Location ID 

**Your Task:**
YOU MUST USE Current World Time, DAY OF THE WEEK, SEASON, NEWS AND WEATHER as GROUNDING FOR YOUR NARRATIVE.

1.  **Understand the Event:** Read the Actor, Intent, and Factual Outcome Description.
2.  **Recall the Mood:** Remember the required narrative style is **'{world_mood}'**.
3.  **Consider the Context:** Note Recent Narrative History. **IGNORE any `World Style/Mood` in `Recent Narrative History`. Prioritize the established '{world_mood}' style.**"""

    # Append dynamically constructed instructions based on world_type and sub_genre
    if world_type == "real" and sub_genre == "realtime":
        narrator_style_adherence_instruction = f"**Style Adherence:** STRICTLY adhere to **'{world_mood}'** and **REAL WORLD REALISM**. Infuse with appropriate atmosphere, plausible sensory details, and tone, all **critically consistent with the provided `Current World Time`**."
        narrator_infuse_time_env_instruction = f"**Infuse with Time and Environment (Realistically):** The `Current World Time` provided in the input is the **absolute definitive time for the scene you are describing.** You MUST use this provided time as the primary basis for describing realistic lighting, typical activity levels for that specific time of day, and other time-dependent sensory details. **This provided `Current World Time` MUST take precedence over any general assumptions or typical scenarios suggested by the actor's actions or the overall mood.** For example, if the `Current World Time` is '08:25 PM (Local time for New York)' but the Factual Outcome Description is 'Isabella Rossi realized she was bored and decided to go to the park', your narrative MUST describe an evening scene of boredom and decision-making, reflecting an 8:25 PM atmosphere, not an afternoon one. Use `Current World Feeds` (weather, news) to add further subtle, atmospheric details that are authentic to a real-world setting and align with the **'{world_mood}'**. Avoid fantastical elements unless explicitly part of a news feed or a very unusual weather event."
    else:
        world_type_description = f"{world_type.capitalize()}{f' ({sub_genre.capitalize()})' if sub_genre else ''}"
        narrator_style_adherence_instruction = f"**Style Adherence:** STRICTLY adhere to **'{world_mood}'** and the **{world_type_description}**. Infuse with appropriate atmosphere, sensory details, and tone."
        narrator_infuse_time_env_instruction = f"**Infuse with Time and Environment (Stylistically):** Use the `Current World Time` and `Current World Feeds` to add atmospheric details that fit the **'{world_mood}'** and the **{world_type_description}** (e.g., magical effects for fantasy, futuristic tech for sci-fi)."

    instruction += f"\n{narrator_infuse_time_env_instruction}"
    instruction += """
4.  **Introduce Ephemeral NPCs (Optional but Encouraged):** If appropriate for the scene, the actor's location, and the narrative flow, you can describe an NPC appearing, speaking, or performing an action.
    *   These NPCs are ephemeral and exist only in the narrative.
    *   If an NPC might be conceptually recurring (e.g., "the usual shopkeeper", "your friend Alex"), you can give them a descriptive tag in parentheses for context, like `(npc_concept_grumpy_shopkeeper)` or `(npc_concept_friend_alex)`. This tag is for LLM understanding, not a system ID.
    *   Example: "As [Actor Name] entered the tavern, a grizzled man with an eye patch (npc_concept_old_pirate_01) at a corner table grunted a greeting."
    *   Example: "A street vendor (npc_concept_flower_seller_01) called out, '[Actor Name], lovely flowers for a lovely day?'"
5.  **Generate Narrative and Discover Entities (Especially for `look_around`):**
    *   Write a single, engaging narrative paragraph in the **present tense**. **CRITICAL: Your `narrative` paragraph in the JSON output MUST begin by stating the `Current World Time` (which is part of your core instructions above, dynamically updated for this turn), followed by the rest of your narrative.** For example, if the dynamically inserted `Current World Time` was "07:33 PM (Local time for New York)", your `narrative` should start with "At 07:33 PM (Local time for New York), ...". If it was "120.5s elapsed", it should start "At 120.5s elapsed, ...".
    """
    instruction += narrator_style_adherence_instruction # Append the style adherence instruction
    instruction += """
                **CRITICAL JSON FORMATTING: When generating the 'narrative' string, if you include any direct speech or text that itself contains double quotes (\"), you MUST escape those internal double quotes with a backslash (e.g., \\\"text in quotes\\\"). Failure to do so will result in invalid JSON.**
    *   **Show, Don't Just Tell.**
    *   **Incorporate Intent (Optional).**
    *   **Regarding Speech (CRITICAL):** If the `Factual Outcome Description` indicates speaking, your narrative MUST describe the *act, manner, and scene* of the speaking event (e.g., "Daniel cleared his throat and spoke to Ava," "Ava replied with a smile"). You MUST NOT include the actual words spoken by the actor from `Original Intent.details` in your narrative. The system handles delivering the speech content separately. You MAY, however, include *new* speech if you are introducing an ephemeral NPC who is speaking as part of your narrative.
    *   **Flow:** Ensure reasonable flow.
    *   **If the `Original Intent.action_type` was `look_around` (CRITICAL - Pay attention to location context):**
        *   **Examine `Actor's Current Location State.description` (provided implicitly via the actor's state, which you don't directly see but influences the context).** This description is your primary source for understanding the current location.
        *   **If the location description suggests an intermediate or "in-transit" point** (e.g., "On a street between X and Y," "Partway along the forest path," "You find yourself reorienting after an interruption"), your narrative and discoveries should reflect this. Describe the immediate surroundings consistent with being on a journey (e.g., the road, sidewalk, surrounding environment like buildings or trees, a sense of direction).
            *   `discovered_objects` might be more generic (e.g., "a passing car," "a street sign," "a patch of wildflowers by the road").
            *   For `discovered_connections` in transit:
                *   `to_location_id_hint`: Should reflect the ongoing journey (e.g., "Road_Towards_Downtown", "Forest_Path_North", "Street_Towards_Park_Entrance").
                *   `description`: Describe the path (e.g., "The road continues towards downtown.", "The forest path winds deeper into the woods to the north.").
        *   **If the location description is for a well-defined place** (e.g., "a bedroom," "a coffee shop," "a library"), then your narrative MUST describe the key features and plausible objects the actor would see in that specific type of location. Consider the `Original Intent.details` (e.g., "trying to identify the closet's location") to ensure relevant objects are mentioned.
        *   You MAY also introduce ephemeral NPCs if appropriate for the scene.
        *   For each object and **individual NPC** you describe in the narrative, you MUST also list them in the `discovered_objects` and `discovered_npcs` fields in the JSON output (see below). Assign a simple, unique `id` (e.g., `closet_bedroom_01`, `npc_cat_01`), a `name`, a brief `description`, and set `is_interactive` to `true` if it's something an agent could plausibly interact with. For objects, you can also add common-sense `properties` (e.g., `{{"is_container": true, "is_openable": true}}` for a closet).
        *   **Distinction for `discovered_objects` vs. `discovered_connections`:** Large interactive items or furniture within the current location (e.g., a table, a specific workbench, a large machine, a bed) should be listed as `discovered_objects` with appropriate properties (as a JSON string in the `properties_str` field, e.g., `properties_str: "{{\\\"is_container\\\": true, \\\"is_openable\\\": true}}"`). Do NOT create a `discovered_connection` leading *to* such an object as if it were a separate navigable area. `discovered_connections` are for actual paths, doorways, or portals leading to different conceptual areas or rooms.
        *   **Also, if `look_around`, identify and describe potential exits or paths to other (possibly new/undiscovered) locations.** List these in `discovered_connections`.
            *   **`to_location_id_hint` (CRITICAL):**
                *   This MUST be a descriptive, conceptual ID for the destination.
                *   **Infer common-sense adjacent location types based on the current location's description and the overall world context.**
                *   **AVOID overly generic hints like "Unknown", "Another_Room", "Next_Area" if a more specific inference can be made.** Use such generics ONLY as an absolute last resort if no plausible inference is possible.
                *   **Naming Convention for `to_location_id_hint`:** Use PascalCase or snake_case. Make it descriptive.
                    *   Examples for a "bedroom" in an "apartment": `Hallway_Apartment_01`, `Living_Room_Apartment_01`, `Bathroom_Apartment_01`.
                    *   Examples for a "city street": `Next_Block_Main_Street`, `Alleyway_West_Side`, `Entrance_General_Store_01`.
                    *   Examples for a "forest clearing": `Dense_Woods_North`, `Riverbank_Trail_East`.
                    *   Examples for a "sci-fi spaceship corridor": `Airlock_Sector_B`, `Corridor_To_Bridge`, `Mess_Hall_Entrance`.
                    *   Examples for a "fantasy tavern": `Tavern_Kitchen_Door`, `Stairs_To_Inn_Rooms`, `Back_Alley_Exit_Tavern`.
                *   The hint should be specific enough for the World Engine to potentially create a new location entry if it doesn't exist.
            *   `description`: How this connection appears (e.g., "A narrow, overgrown path leading north into the woods.").
            *   `travel_time_estimate_seconds` (optional): A rough estimate if discernible.
        *   **Regardless of location type, your `narrative` MUST use the `Actor's Current Location State.description` as the foundation for what the actor perceives.**

**Output Format:**
Your response MUST be a single JSON object matching the `NarrationResponse` schema with the following fields:
- "narrative": "Your stylized narrative paragraph, starting with 'At [Current World Time], ...'."
- "discovered_objects": A list of objects discovered (name, id, description, is_interactive, properties_str). `properties_str` must be a JSON string or null. Empty list if none.
- "discovered_connections": A list of connections discovered (to_location_id_hint, description, travel_time_estimate_seconds). Empty list if none.
- "discovered_npcs": A list of NPCs discovered (name, id, description, is_interactive). Empty list if none.
"""

    class NarrationLlmWithContext(LlmAgent):
        async def _prepare_input_async(self, ctx: InvocationContext, raw_input: Any) -> str:
            # Clear conversation history
            if hasattr(self, 'conversation_history'):
                self.conversation_history = []
                
            # NarrationPhase sets NARRATION_INPUT_DATA (WorldEngineResponse)
            # and CURRENT_ACTOR_ID (the one whose action is being narrated)
            world_engine_output = ctx.session.state.get(StateKeys.NARRATION_INPUT_DATA)
            actor_id_for_narration = ctx.session.state.get(StateKeys.CURRENT_ACTOR_ID) # Assuming NarrationPhase sets this

            if not world_engine_output:
                logger.error(f"[{self.name}] Missing NARRATION_INPUT_DATA in session state. Cannot build context.")
                return "\n\n--- ERROR: NARRATION INPUT DATA MISSING ---"
            if not actor_id_for_narration:
                # Try to infer actor_id from world_engine_output if possible, as a fallback
                # This is a bit heuristic; ideally, NarrationPhase explicitly provides it.
                # Example: look for "simulacra_profiles.[sim_id].some_key" in results
                results = world_engine_output.get("results", {})
                for key_path in results.keys():
                    if key_path.startswith(f"{SIMULACRA_KEY}.") and ".status" in key_path: # Heuristic
                        parts = key_path.split('.')
                        if len(parts) > 1:
                            actor_id_for_narration = parts[1]
                            logger.warning(f"[{self.name}] Inferred actor_id '{actor_id_for_narration}' from world_engine_output.results.")
                            break
                if not actor_id_for_narration:
                    logger.error(f"[{self.name}] Missing CURRENT_ACTOR_ID for narration in session state and could not infer. Cannot build context.")
                    return "\n\n--- ERROR: ACTOR ID FOR NARRATION MISSING ---"

            # Rest of the method remains unchanged
            actor_sim_data = get_nested(ctx.session.state, SIMULACRA_KEY, actor_id_for_narration, default={})
            persona_name = get_nested(actor_sim_data, 'persona_details', 'Name', default=actor_id_for_narration)
            
            # Construct the dynamic intro string using the template defined in the outer scope
            current_dynamic_intro = narrator_intro_template.replace("{{persona_name}}", persona_name).replace("{{sim_id}}", actor_id_for_narration)
            if world_type == "real" and sub_genre == "realtime":
                current_dynamic_intro += f" This is a **REAL WORLD, REALTIME** simulation. Your narrative MUST be grounded and realistic."
            else:
                world_type_desc_for_intro = f"{world_type.capitalize()}{f' ({sub_genre.capitalize()})' if sub_genre else ''}"
                current_dynamic_intro += f" This is a **{world_type_desc_for_intro}** simulation (e.g., Fictional, Fantasy, Sci-Fi, or Real World but not Realtime). Your narrative should align with this context and the specified mood."
                
            context_string = _build_narration_llm_context_string(actor_id_for_narration, persona_name, world_engine_output, ctx, current_dynamic_intro)
            return context_string

    return NarrationLlmWithContext(
        name="NarrationLLMAgent", # This should be unique if multiple narrators, but fine for one
        instruction=instruction, # Pass the fully constructed static instruction
        output_schema=NarrationResponse,
        output_key=StateKeys.NARRATION_RESULT,
        description=f"LLM Narrator: Generates '{world_mood}' narrative based on factual outcomes.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        model=MODEL_NAME,
    )

def _build_narration_llm_context_string(actor_id: str, actor_name: str, world_engine_output: Dict, ctx: InvocationContext, dynamic_intro_text: str) -> str:
    """Helper function to build the detailed context string for the NarrationLLM."""
    state = ctx.session.state

    # Extract relevant parts from world_engine_output
    original_intent_str = json.dumps(world_engine_output.get("original_intent", 
                                     ctx.session.state.get(StateKeys.CURRENT_ACTOR_INTENT, {})), indent=2) 
    factual_outcome_desc = world_engine_output.get("outcome_description", "No factual outcome provided.")
    state_changes_str = json.dumps(world_engine_output.get("results", {}), indent=2)

    actor_current_location_id = get_nested(world_engine_output.get("results", {}), f"{SIMULACRA_KEY}.{actor_id}.location", 
                                           default=get_nested(state, SIMULACRA_KEY, actor_id, 'location', 'unknown_location'))
    
    current_location_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_current_location_id, default={})
    current_location_desc_str = current_location_data.get('description', 'Description unavailable.')

    world_feeds_str = json.dumps(state.get('world_feeds', {}), indent=2)

    narrative_log = state.get('narrative_log', [])
    recent_narrative_history_str = "\n".join([f"- {entry}" for entry in narrative_log[-5:]]) 
    if not recent_narrative_history_str: recent_narrative_history_str = "No recent narrative history."

    current_time_str = get_time_string_for_prompt(state, state.get('world_time', 0.0))

    context_parts = [
        f"\n\n--- CONTEXT FOR NARRATOR (Regarding Actor: {actor_name} [{actor_id}]) ---",
        dynamic_intro_text, 
        f"Current World Time (for narrative start): {current_time_str}",
        f"Actor's Original Intent:\n{original_intent_str}",
        f"Factual Outcome Description from World Engine:\n{factual_outcome_desc}",
        f"Resulting State Changes:\n{state_changes_str}",
        f"Actor's Current Location ID (after action): {actor_current_location_id}",
        f"Description of Actor's Current Location (for look_around):\n{current_location_desc_str}",
        f"Current World Feeds:\n{world_feeds_str}",
        f"Recent Narrative History (last 5 entries):\n{recent_narrative_history_str}",
        "--- END OF CONTEXT ---"
    ]
    return "\n".join(context_parts)

def create_search_llm_agent() -> LlmAgent:
    """Creates a dedicated LLM agent for performing Google searches."""
    agent_name = "SearchLLMAgent"
    instruction = """I can answer your questions by searching the internet. Just ask me anything!"""

    # Create a custom search agent class that resets context between calls
    class SearchLLMAgentWithReset(LlmAgent):
        async def _prepare_input_async(self, ctx: InvocationContext, raw_input: Any) -> str:
            # This ensures we're only sending the current query to the model
            # without accumulated context from previous searches
            if isinstance(raw_input, str):
                return raw_input
            return str(raw_input)

    return SearchLLMAgentWithReset(
        name="SearchLLMAgent",
        instruction=instruction,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        model=SEARCH_AGENT_MODEL_NAME,
        tools=[google_search],
        description="Dedicated LLM Agent for performing Google Searches."
    )


# --- Phase 2: Functional Phase Agents ---

class SimulacraDecisionPhase(ParallelAgent):
    def __init__(self, simulacra_agents: List[LlmAgent]):
        super().__init__(name="SimulacraDecisionPhase", sub_agents=simulacra_agents)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting decision phase for {len(self.sub_agents)} simulacra.")
        async for event in super()._run_async_impl(ctx):
            yield event
        logger.info(f"[{self.name}] Decision phase completed.")
        yield Event(author=self.name, content=None) # Simple status

class WorldResolutionPhase(SequentialAgent):
    world_engine_agent: LlmAgent # Field declared
    def __init__(self, world_engine_agent: LlmAgent):
        super().__init__(
            name="WorldResolutionPhase", 
            sub_agents=[], 
            world_engine_agent=world_engine_agent  # Pass the field to super
        )
        # self.world_engine_agent = world_engine_agent # This line is now handled by Pydantic via super()

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting world resolution phase.")
        active_sim_ids = ctx.session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        resolved_count = 0

        for sim_id in active_sim_ids:
            intent_key = StateKeys.simulacra_intent(sim_id)
            intent_data = ctx.session.state.pop(intent_key, None)

            if intent_data and intent_data.get("action_type") == "no_op":
                logger.info(f"[{self.name}] Skipping 'no_op' intent for {sim_id}.")
                # Optionally, clear the world_engine_result if it was from a previous actor
                ctx.session.state.pop(StateKeys.WORLD_ENGINE_RESULT, None)
                continue
            if intent_data:
                logger.info(f"[{self.name}] Resolving intent for {sim_id}: {intent_data.get('action_type')}")
                ctx.session.state[StateKeys.CURRENT_ACTOR_INTENT] = intent_data
                ctx.session.state[StateKeys.CURRENT_ACTOR_ID] = sim_id

                async for event in self.world_engine_agent.run_async(ctx):
                    yield event

                world_engine_response = ctx.session.state.get(StateKeys.WORLD_ENGINE_RESULT)
                if world_engine_response:
                    logger.info(f"[{self.name}] World Engine response for {sim_id}: Valid={world_engine_response.get('valid_action')}")
                    for path, value in world_engine_response.get("results", {}).items():
                        _update_state_value(ctx.session.state, path, value, logger)

                    duration = world_engine_response.get("duration", 0.0)
                    current_sim_time = ctx.session.state.get("world_time", 0.0)
                    if world_engine_response.get("valid_action"):
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.status", "busy", logger)
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.current_action_end_time", current_sim_time + duration, logger)
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.current_action_description", world_engine_response.get("outcome_description",""), logger)
                    else:
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.status", "idle", logger)
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.last_observation", world_engine_response.get("outcome_description", "Action was invalid."), logger)

                    scheduled_event = world_engine_response.get("scheduled_future_event")
                    if scheduled_event and isinstance(scheduled_event, dict):
                        delay = scheduled_event.get("estimated_delay_seconds", 0.0)
                        trigger_at = current_sim_time + duration + delay
                        scheduled_event["trigger_sim_time"] = trigger_at
                        ctx.session.state.setdefault("pending_simulation_events", []).append(scheduled_event)
                        logger.info(f"[{self.name}] Scheduled future event: {scheduled_event.get('event_type')} for {scheduled_event.get('target_agent_id','world')} at {trigger_at:.1f}s")
                    resolved_count += 1
                # ctx.session.state.pop(StateKeys.CURRENT_ACTOR_INTENT, None)
                # ctx.session.state.pop(StateKeys.CURRENT_ACTOR_ID, None)
        logger.info(f"[{self.name}] World Resolution Phase completed for {resolved_count} intents.")
        yield Event(author=self.name, content=None) # Simple status


class NarrationPhase(SequentialAgent):
    narration_agent: LlmAgent # Declare the field
    def __init__(self, narration_agent: LlmAgent):
        super().__init__(
            name="NarrationPhase", 
            sub_agents=[],
            narration_agent=narration_agent # Pass the field to super
        )
        # self.narration_agent = narration_agent # Handled by Pydantic

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting narration phase.")
        world_engine_output = ctx.session.state.get(StateKeys.WORLD_ENGINE_RESULT)
        if world_engine_output and world_engine_output.get("valid_action"):
            # Ensure CURRENT_ACTOR_ID is set for the NarrationLLM context builder
            # This assumes WorldResolutionPhase leaves CURRENT_ACTOR_ID for the last processed actor,
            # or NarrationPhase is designed to handle one actor's outcome at a time.
            # If not, actor_id needs to be explicitly passed or stored with world_engine_output.
            logger.info(f"[{self.name}] Narrating outcome: {world_engine_output.get('outcome_description')}")
            ctx.session.state[StateKeys.NARRATION_INPUT_DATA] = world_engine_output

            async for event in self.narration_agent.run_async(ctx):
                yield event

            narration_response = ctx.session.state.get(StateKeys.NARRATION_RESULT)
            if narration_response:
                narrative_text = narration_response.get("narrative")
                if narrative_text:
                    ctx.session.state.setdefault("narrative_log", []).append(narrative_text)
                ctx.session.state.pop(StateKeys.NARRATION_INPUT_DATA, None)
            logger.info(f"[{self.name}] Narration Phase Completed.")
            yield Event(author=self.name, content=None) # Simple status
        else:
            logger.info(f"[{self.name}] Narration Phase: No valid world engine output to narrate or output was invalid.")
            yield Event(author=self.name, content=None) # Simple status


class TimeAdvancementPhase(BaseAgent):
    def __init__(self):
        super().__init__(name="TimeAdvancementPhase")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting time advancement phase.")
        current_sim_time = ctx.session.state.get("world_time", 0.0)
        sim_speed = ctx.session.state.get("config", {}).get("SIMULATION_SPEED_FACTOR", SIMULATION_SPEED_FACTOR)
        update_interval = ctx.session.state.get("config", {}).get("UPDATE_INTERVAL", UPDATE_INTERVAL)
        
        sim_delta_time = update_interval * sim_speed
        new_sim_time = current_sim_time + sim_delta_time
        ctx.session.state["world_time"] = new_sim_time
        logger.info(f"[{self.name}] Advanced time from {current_sim_time:.2f} to {new_sim_time:.2f}s.")

        for sim_id, agent_state_data in list(ctx.session.state.get(SIMULACRA_KEY, {}).items()):
            if agent_state_data.get("status") == "busy":
                action_end_time = agent_state_data.get("current_action_end_time", -1.0)
                if action_end_time <= new_sim_time:
                    logger.info(f"[{self.name}] Agent {sim_id} finished action. Setting to idle.")
                    _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.status", "idle", logger)
                    _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.current_interrupt_probability", None, logger)
                    current_mem_log = agent_state_data.get("memory_log", [])
                    if len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                        _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{sim_id}.memory_log", current_mem_log[-MAX_MEMORY_LOG_ENTRIES:], logger)
                        logger.debug(f"[{self.name}] Pruned memory log for {sim_id}.")
            yield Event(author=self.name, content=None) # Simple status


class ScheduledEventProcessorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ScheduledEventProcessorAgent")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting scheduled event processing.")
        current_sim_time = ctx.session.state.get("world_time", 0.0)
        pending_events = ctx.session.state.get("pending_simulation_events", [])
        processed_indices = []
        processed_count = 0

        for i, event_data in enumerate(pending_events):
            if event_data.get("trigger_sim_time", float('inf')) <= current_sim_time:
                event_type = event_data.get("event_type")
                target_agent_id = event_data.get("target_agent_id")
                details = event_data.get("details", {})
                logger.info(f"[{self.name}] Processing event: {event_type} for {target_agent_id}")

                if event_type == "simulacra_speech_received_as_interrupt" and target_agent_id:
                     speech_content = details.get("message_content", "Someone spoke.")
                     _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{target_agent_id}.last_observation", speech_content, logger)
                     _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{target_agent_id}.status", "idle", logger)
                     _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{target_agent_id}.current_pending_intent_id", None, logger)
                processed_indices.append(i)
                processed_count +=1

        for i in sorted(processed_indices, reverse=True):
            pending_events.pop(i)
        
        if processed_count > 0:
            logger.info(f"[{self.name}] Processed {processed_count} scheduled events.")
            yield Event(author=self.name, content=None) # Simple status
        else:
            logger.info(f"[{self.name}] No scheduled events due for processing.")
            yield Event(author=self.name, content=None) # Simple status


class WorldInfoGathererAgent(BaseAgent):
    search_agent_tool: agent_tool.AgentTool # Declare field
    _last_run_sim_time: float = -float('inf') # Declare field with default

    def __init__(self, search_agent_tool: agent_tool.AgentTool):
        super().__init__(
            name="WorldInfoGathererAgent",
            search_agent_tool=search_agent_tool # Pass to super
        )
        # self._last_run_sim_time is initialized by its default value

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Checking if world info gathering is due.")
        current_sim_time = ctx.session.state.get("world_time", 0.0)
        interval = ctx.session.state.get("config", {}).get("WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS", WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS)

        if current_sim_time >= self._last_run_sim_time + interval:
            logger.info(f"[{self.name}] Gathering world info at sim_time {current_sim_time:.1f}s.")
            world_mood = ctx.session.state.get("world_mood_global", "neutral")
            location_info = ctx.session.state.get(WORLD_TEMPLATE_DETAILS_KEY, {}).get("location", {})
            location_context_str = f"{location_info.get('city', '')}, {location_info.get('country', '')}".strip(", ")
            
            weather_data = await generate_simulated_world_feed_content(
                current_sim_state=ctx.session.state, category="weather", simulation_time=current_sim_time,
                location_context=location_context_str, world_mood=world_mood,
                search_agent_tool_for_real_feeds=self.search_agent_tool,
                adk_context_for_tool_run=ctx, # Pass the current agent's context
                logger_instance=logger
            )
            _update_state_value(ctx.session.state, 'world_feeds.weather', weather_data, logger)
            
            self._last_run_sim_time = current_sim_time
            yield Event(author=self.name, content=None) # Simple status
        else:
            yield Event(author=self.name, content=None) # Simple status

class NarrativeImageGeneratorAgent(BaseAgent):
    _last_run_real_time: float # Declare field
    # _image_gen_model will now be the client instance for the older API style
    _image_gen_client: Optional[Any] # Using Any for genai_image_sdk.Client()
    _bluesky_client: Optional[BlueskyClient] # Declare field
    _initialized_clients: bool # Declare field
    def __init__(self):
        super().__init__(name="NarrativeImageGeneratorAgent")
        self._last_run_real_time = time.monotonic()
        self._image_gen_model = None
        self._bluesky_client = None
        self._initialized_clients = False # Fixed: was self._image_gen_model, now self._image_gen_client

    async def _initialize_clients_if_needed(self, ctx: InvocationContext):
        if self._initialized_clients:
            return

        enable_generation = ctx.session.state.get("config", {}).get("ENABLE_NARRATIVE_IMAGE_GENERATION", ENABLE_NARRATIVE_IMAGE_GENERATION)
        if not enable_generation:
            logger.info(f"[{self.name}] Image generation is disabled by configuration.")
            self._initialized_clients = True
            return

        try:
            # Use the Client() pattern for image generation
            self._image_gen_client = genai_image_sdk.Client()
            logger.info(f"[{self.name}] Image generation client (genai_image_sdk.Client) initialized.")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize image generation model: {e}")
            self._image_gen_client = None


        enable_bluesky = ctx.session.state.get("config", {}).get("ENABLE_BLUESKY_POSTING", ENABLE_BLUESKY_POSTING)
        if enable_bluesky:
            try:
                self._bluesky_client = BlueskyClient()
                logger.info(f"[{self.name}] Bluesky client initialized.")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to initialize Bluesky client: {e}")
                self._bluesky_client = None
        
        self._initialized_clients = True

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        await self._initialize_clients_if_needed(ctx)

        enable_generation = ctx.session.state.get("config", {}).get("ENABLE_NARRATIVE_IMAGE_GENERATION", ENABLE_NARRATIVE_IMAGE_GENERATION)
        if not enable_generation or not self._image_gen_client: # Check _image_gen_client
            yield Event(author=self.name, content=None) # Simple status
            return

        interval_real_seconds = ctx.session.state.get("config", {}).get("IMAGE_GENERATION_INTERVAL_REAL_SECONDS", IMAGE_GENERATION_INTERVAL_REAL_SECONDS)
        if time.monotonic() < self._last_run_real_time + interval_real_seconds:
            yield Event(author=self.name, content=None) # Simple status
            return

        logger.info(f"[{self.name}] Attempting narrative image generation.")
        self._last_run_real_time = time.monotonic()

        narrative_log = ctx.session.state.get("narrative_log", [])
        if not narrative_log:
            yield Event(author=self.name, content=None) # Simple status
            return

        latest_narrative_full = narrative_log[-1]
        original_narrative_prompt_text = re.sub(r'^\[T\d+\.\d+\]\s*', '', latest_narrative_full).strip()
        if not original_narrative_prompt_text:
            yield Event(author=self.name, content=None) # Simple status
            return

        current_sim_time = ctx.session.state.get("world_time", 0.0)
        world_mood = ctx.session.state.get("world_mood_global", "neutral")
        
        refined_narrative_for_image = original_narrative_prompt_text
        try:
            # Use genai_text_client for GenerativeModel (text models)
            refinement_llm = genai_text_client.GenerativeModel(MODEL_NAME)
            time_str = get_time_string_for_prompt(ctx.session.state, sim_elapsed_time_seconds=current_sim_time)
            weather_cond = get_nested(ctx.session.state, 'world_feeds', 'weather', 'condition', default='Unknown')
            prompt_for_refinement = f"Refine this narrative for an image prompt, focusing on a single visual moment: \"{original_narrative_prompt_text}\". Time: {time_str}. Weather: {weather_cond}. World Mood: {world_mood}. Output only the refined visual description."
            response_refinement = await refinement_llm.generate_content_async(prompt_for_refinement)
            if response_refinement.text:
                refined_narrative_for_image = response_refinement.text.strip()
        except Exception as e_refine:
            logger.error(f"[{self.name}] Error refining narrative: {e_refine}")

        random_style_str = get_random_style_combination()
        output_dir = ctx.session.state.get("config", {}).get("IMAGE_GENERATION_OUTPUT_DIR", IMAGE_GENERATION_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the detailed prompt similar to your working example
        time_string_for_image_prompt = get_time_string_for_prompt(ctx.session.state, sim_elapsed_time_seconds=current_sim_time)
        weather_condition_for_image_prompt = get_nested(ctx.session.state, 'world_feeds', 'weather', 'condition', default='The weather is unknown.')
        # Assuming actor_name_in_narrative logic would be similar or simplified for the agent
        actor_name_in_narrative = "the observer" # Simplified for agent context

        prompt_for_image = f"""
Generate a high-quality, visually appealing, **photo-realistic** photograph of a scene or subject directly related to the following narrative context, as if captured by {actor_name_in_narrative}.
Narrative Context: "{refined_narrative_for_image}"
Style: "{random_style_str}"
Time of Day: "{time_string_for_image_prompt}"
Weather: "{weather_condition_for_image_prompt}"
World Mood: "{world_mood}"
Crucial Exclusions: No digital overlays, UI elements, watermarks, or the actor themselves.
Generate this image.
"""

        try:
            logger.info(f"[{self.name}] Requesting image with prompt: {prompt_for_image[:100]}...")
            image_model_name_from_config = ctx.session.state.get("config", {}).get("IMAGE_GENERATION_MODEL_NAME", IMAGE_GENERATION_MODEL_NAME)
            
            response = await asyncio.to_thread(
                self._image_gen_client.models.generate_images, # Use the client.models.generate_images
                model=image_model_name_from_config,
                prompt=prompt_for_image,
                config=genai_image_sdk.types.GenerateImagesConfig( # Use genai_image_sdk.types
                    number_of_images=1,
                )
            )

            if response.generated_images: # Check response.generated_images
                for generated_image in response.generated_images:
                    image_bytes = generated_image.image.image_bytes # Access bytes correctly
                    image = Image.open(BytesIO(image_bytes))
                
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    sim_time_str = f"T{current_sim_time:.0f}"
                    image_filename = f"narrative_{sim_time_str}_{timestamp_str}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)
                    logger.info(f"[{self.name}] Saved image: {image_path}")

                    event_log_entry = {
                        "sim_time_s": round(current_sim_time, 2),
                        "agent_id": self.name,
                        "event_type": "image_generation_adk",
                        "data": {"image_filename": image_filename, "prompt_snippet": refined_narrative_for_image}
                    }
                    ctx.session.state.setdefault("event_log", []).append(event_log_entry)
                    yield Event(author=self.name, content=None) # Simple status

                    enable_bluesky = ctx.session.state.get("config", {}).get("ENABLE_BLUESKY_POSTING", ENABLE_BLUESKY_POSTING)
                    if enable_bluesky and self._bluesky_client:
                        logger.info(f"[{self.name}] Bluesky posting for {image_filename} would occur here.")
                    break # Assuming one image
            else:
                # Log the text part of the response if image generation failed but text was returned
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
                    logger.warning(f"[{self.name}] Image generation failed. Model response text: {response.candidates[0].content.parts[0].text}")
                yield Event(author=self.name, content=None) # Simple status
        except Exception as e:
            logger.error(f"[{self.name}] Error during image generation API call: {e}")
            yield Event(author=self.name, content=None) # Simple status

class DynamicInterruptionAgent(BaseAgent):
    _last_run_real_time: float # Declare field

    def __init__(self):
        super().__init__(name="DynamicInterruptionAgent")
        self._last_run_real_time = time.monotonic()

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        config = ctx.session.state.get("config", {})
        interval_real_seconds = config.get("DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS", DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS)

        if time.monotonic() < self._last_run_real_time + interval_real_seconds:
            yield Event(author=self.name, content=None) # Simple status
            return

        logger.info(f"[{self.name}] Performing dynamic interruption check.")
        self._last_run_real_time = time.monotonic()

        current_sim_time = ctx.session.state.get("world_time", 0.0)
        active_sim_ids = ctx.session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        world_mood = ctx.session.state.get("world_mood_global", "neutral")

        for agent_id in active_sim_ids:
            agent_state = get_nested(ctx.session.state, SIMULACRA_KEY, agent_id, default={})
            if not agent_state or agent_state.get("status") != "busy":
                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
                continue

            agent_name = get_nested(agent_state, "persona_details", "Name", default=agent_id)
            last_interruption_time = agent_state.get("last_interjection_sim_time", 0.0)
            cooldown = config.get("INTERJECTION_COOLDOWN_SIM_SECONDS", INTERJECTION_COOLDOWN_SIM_SECONDS)
            if (current_sim_time - last_interruption_time) < cooldown:
                continue

            remaining_duration = agent_state.get("current_action_end_time", 0.0) - current_sim_time
            min_duration_check = config.get("MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK", MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK)
            if remaining_duration < min_duration_check:
                continue

            target_duration = config.get("DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS", DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS)
            prob_at_target = config.get("DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION", DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION)
            min_prob = config.get("DYNAMIC_INTERRUPTION_MIN_PROB", DYNAMIC_INTERRUPTION_MIN_PROB)
            max_prob_cap = config.get("DYNAMIC_INTERRUPTION_MAX_PROB_CAP", DYNAMIC_INTERRUPTION_MAX_PROB_CAP)

            interrupt_probability = 0.0
            if target_duration > 0:
                duration_factor = remaining_duration / target_duration
                scaled_prob = duration_factor * prob_at_target
                interrupt_probability = min(max_prob_cap, max(min_prob, scaled_prob))
            else:
                interrupt_probability = min(max_prob_cap, min_prob)
            
            _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", interrupt_probability, logger)

            if random.random() < interrupt_probability:
                logger.info(f"[{self.name}] Triggering dynamic interruption for {agent_name} (Prob: {interrupt_probability:.3f}).")
                interruption_text = f"A minor unexpected event occurs, breaking {agent_name}'s concentration."
                try:
                    interrupt_llm = genai_image_client.GenerativeModel(MODEL_NAME)
                    prompt = f"Agent {agent_name} is busy: \"{agent_state.get('current_action_description', 'activity')}\". World mood: \"{world_mood}\". An unexpected minor interruption occurs. Describe it in 1-2 narrative sentences for {agent_name} to perceive. Output ONLY the narrative."
                    response = await interrupt_llm.generate_content_async(prompt)
                    if response.text:
                        interruption_text = response.text.strip()
                except Exception as e_llm:
                    logger.error(f"[{self.name}] LLM error for interruption text: {e_llm}")

                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", interruption_text, logger)
                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.current_action_end_time", current_sim_time, logger)
                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.last_interjection_sim_time", current_sim_time, logger)
                _update_state_value(ctx.session.state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
                yield Event(author=self.name, content=None) # Simple status
                return
            yield Event(author=self.name, content=None) # Simple status


class WorldMaintenancePhase(SequentialAgent):
    # Fields for sub-agents, passed to super().__init__
    scheduled_event_processor: ScheduledEventProcessorAgent
    world_info_gatherer: WorldInfoGathererAgent
    narrative_image_generator: NarrativeImageGeneratorAgent
    dynamic_interruption: DynamicInterruptionAgent

    def __init__(self,
                 scheduled_event_processor: ScheduledEventProcessorAgent,
                 world_info_gatherer: WorldInfoGathererAgent,
                 narrative_image_generator: NarrativeImageGeneratorAgent,
                 dynamic_interruption: DynamicInterruptionAgent):
        super().__init__(
            name="WorldMaintenancePhase",
            sub_agents=[ # sub_agents are passed to the parent SequentialAgent
                scheduled_event_processor,
                world_info_gatherer,
                narrative_image_generator,
                dynamic_interruption
            ],
            # Pass the actual agent instances for Pydantic field validation
            scheduled_event_processor=scheduled_event_processor,
            world_info_gatherer=world_info_gatherer,
            narrative_image_generator=narrative_image_generator,
            dynamic_interruption=dynamic_interruption
        )
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting world maintenance phase.")
        async for event in super()._run_async_impl(ctx):
            yield event
        logger.info(f"[{self.name}] World maintenance phase completed.")
        yield Event(author=self.name, content=None) # Simple status

class MasterLoopAgent(LoopAgent):
    # name and description are class attributes, not Pydantic fields for __init__
    # coordinator and max_simulation_ticks are handled by LoopAgent's __init__ if passed as kwargs

    def __init__(self, coordinator: "SimulationStepCoordinator", max_simulation_ticks: int):
        super().__init__(
            name="MasterLoopAgent",
            sub_agents=[coordinator], # LoopAgent takes sub_agents
            max_iterations=max_simulation_ticks
            # coordinator=coordinator # Not a direct field of LoopAgent, sub_agents is used
        )
        # If you need to store coordinator for other reasons, declare it as a field:
        # self.coordinator_ref: SimulationStepCoordinator = coordinator 
        # But LoopAgent itself uses sub_agents[0]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        current_tick = ctx.session.state.get("current_tick", 0)
        logger.debug(f"--- MasterLoopAgent: Starting Simulation Tick {current_tick} ---")
        ctx.session.state["current_tick"] = current_tick + 1
        
        async for event in super()._run_async_impl(ctx):
            yield event

class SimulationStepCoordinator(SequentialAgent):
    # Fields for sub-agents, passed to super().__init__
    simulacra_decision_phase: SimulacraDecisionPhase
    world_resolution_phase: WorldResolutionPhase
    narration_phase: NarrationPhase
    time_advancement_phase: TimeAdvancementPhase
    world_maintenance_phase: WorldMaintenancePhase

    def __init__(self,
                 simulacra_decision_phase: SimulacraDecisionPhase,
                 world_resolution_phase: WorldResolutionPhase,
                 narration_phase: NarrationPhase,
                 time_advancement_phase: TimeAdvancementPhase,
                 world_maintenance_phase: WorldMaintenancePhase):
        super().__init__(
            name="SimulationStepCoordinator",
            sub_agents=[
                simulacra_decision_phase,
                world_resolution_phase,
                narration_phase,
                time_advancement_phase,
                world_maintenance_phase
            ],
            # Pass actual agent instances for Pydantic field validation
            simulacra_decision_phase=simulacra_decision_phase,
            world_resolution_phase=world_resolution_phase,
            narration_phase=narration_phase,
            time_advancement_phase=time_advancement_phase,
            world_maintenance_phase=world_maintenance_phase
        )

def create_simulation_architecture(
    simulacra_llm_agents: List[LlmAgent],
    world_engine_llm: LlmAgent,
    narration_llm: LlmAgent,
    search_llm_agent: LlmAgent,
    max_ticks: int
) -> MasterLoopAgent:
    """Create the complete simulation architecture with functional phase agents."""

    sim_decision_phase = SimulacraDecisionPhase(simulacra_agents=simulacra_llm_agents)
    world_res_phase = WorldResolutionPhase(world_engine_agent=world_engine_llm)
    narr_phase = NarrationPhase(narration_agent=narration_llm)
    time_adv_phase = TimeAdvancementPhase()
    
    event_proc_agent = ScheduledEventProcessorAgent()
    # AgentTool inherits name and description from the wrapped agent.
    # If you need a custom name/description for the tool itself,
    # you'd set it on the search_llm_agent or create a custom tool wrapper.
    search_tool = agent_tool.AgentTool(agent=search_llm_agent)
    world_info_agent = WorldInfoGathererAgent(search_agent_tool=search_tool)
    image_gen_agent = NarrativeImageGeneratorAgent()
    dynamic_interrupt_agent = DynamicInterruptionAgent()

    world_maint_phase = WorldMaintenancePhase(
        scheduled_event_processor=event_proc_agent,
        world_info_gatherer=world_info_agent,
        narrative_image_generator=image_gen_agent,
        dynamic_interruption=dynamic_interrupt_agent
    )

    coordinator = SimulationStepCoordinator(
        sim_decision_phase, world_res_phase, narr_phase, time_adv_phase, world_maint_phase
    )

    master_loop = MasterLoopAgent(
        coordinator=coordinator, # MasterLoopAgent's __init__ expects 'coordinator'
        max_simulation_ticks=max_ticks
    )
    return master_loop
