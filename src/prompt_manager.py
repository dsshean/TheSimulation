# src/prompt_manager.py
import json
from typing import Dict, List, Optional, TYPE_CHECKING, Any
from src.models import ActionDecisionResponse, AllowedActionVerbs # Import the literal

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from src.models import (
        WorldState, ImmediateEnvironment, WorldReactionProfile,
        EmotionAnalysisResponse, ActionDecisionResponse, DayResponse,
        WorldProcessUpdateResponse # If used for schema generation
    )

# Helper function to generate schema refs safely
def get_model_schema(model_class: type) -> Dict:
    try:
        return model_class.model_json_schema(mode='validation')
    except Exception:
        return {"type": "object", "properties": {"error": {"type": "string", "description": "Schema generation failed"}}}


class PromptManager:
    """Manages all prompts used in the simulation."""

    # <<< MODIFIED: generate_day_arc_prompt >>>
    @staticmethod
    def generate_day_arc_prompt(persona: Dict, world_state: Dict) -> str:
        """Generates a prompt to create a narrative arc for the current day."""
        persona_summary = f"""
        Name: {persona.get('name', 'Unknown')}
        Age: {persona.get('age', '?')}
        Occupation: {persona.get('occupation', '?')}
        Personality: {', '.join(persona.get('personality_traits', ['Unknown']))}
        Goals: {', '.join(persona.get('goals', ['None']))}
        Current Emotional State: {persona.get('current_state', {}).get('emotional', 'Neutral')}
        """.strip()
        world_context = f"""
        Date: {world_state.get('current_date', '?')} ({world_state.get('day_of_week', '?')})
        Location Context: {world_state.get('city_name', '?')}, {world_state.get('country_name', '?')}
        General Climate: {world_state.get('social_climate', '?')} / {world_state.get('economic_condition', '?')}
        Major Events: {', '.join(world_state.get('major_events', ['None']))}
        Weather Forecast (if known): {world_state.get('forecast', 'Unknown')}
        """.strip() # Added day_of_week and forecast for more context

        return f"""
        Persona Summary:
        {persona_summary}

        World Context (Start of Day):
        {world_context}

        Task: Like a Dungeon Master setting the scene, outline the character's *expected* plan or narrative arc for the entire upcoming day (from waking until sleeping).
        This arc should describe a **complete daily cycle** with a clear beginning, middle, and end, reflecting the character's likely routine, goals, occupation, and personality.
        Keep the arc **realistic and grounded**. Avoid overly dramatic, mysterious, or open-ended scenarios. Focus on plausible daily activities and interactions. The arc provides a backdrop; specific events will unfold during simulation.

        Good Examples (Realistic, Complete Day Cycle):
        - "A typical workday: Commute to the office, focus on the [Project X] deadline with a brief lunch break, head home, make a simple dinner, and unwind before bed."
        - "A planned day off: Sleep in, meet [Friend's Name] for brunch, run some errands downtown, maybe catch a movie in the evening, then relax at home."
        - "Focus on [Personal Goal]: Spend the morning preparing, attend the [Event/Appointment] in the afternoon, followed by reflection and a quiet evening processing the outcome."
        - "A busy day balancing work and family: Juggle work responsibilities from home, handle [Family Task] mid-day, have dinner with family, and prepare for the next day."

        Bad Examples (Avoid these):
        - "An ordinary day turns extraordinary when a mysterious object is found..." (Avoid mystery/fantasy)
        - "The character decides to investigate the strange occurrences..." (Too open-ended, not a full day plan)
        - "Wake up and see what happens." (Not an arc)
        - "Go to work." (Too brief, lacks daily cycle)

        Output ONLY the narrative arc description as plain text (1-3 concise sentences).
        """

    # <<< Keep existing methods initialize_world_state_from_news_prompt, initialize_world_state_from_context_prompt, etc. as is >>>
    @staticmethod
    def initialize_world_state_from_news_prompt(news_results: str, config: Dict, current_datetime: Dict) -> str:
        """Prompt to initialize world state based on CURRENT news and config."""
        from src.models import WorldState
        location_config = config.get("location", {})
        city = location_config.get("city", "this city")

        return f"""
        Current Real-World Context & News:
        Date: {current_datetime['date']}
        Time: {current_datetime['time']}
        Day: {current_datetime['day_of_week']}

        Recent News Snippets for {city}:
        {news_results if news_results else "No specific news retrieved."}

        Base World Configuration (Location):
        {json.dumps(location_config, indent=2)}

        Task: Create a detailed, rich, and plausible WorldState JSON object based *primarily* on the provided real-world context and news.
        Infer conditions like social/economic/political climate, utility status, etc., realistically from the news and general knowledge of the location ({city}). If news is sparse, make plausible assumptions based on the location and current date/time.
        Populate ALL fields of the WorldState model, providing specific details. Use Celsius for temperature. Determine the day_phase based on the time. Use standard YYYY-MM-DD and HH:MM formats.

        Respond ONLY with the JSON object matching the WorldState schema:
        {json.dumps(get_model_schema(WorldState), indent=2)}
        """

    @staticmethod
    def initialize_world_state_from_context_prompt(date: str, time: str, location: Dict, context_summary: str) -> str:
        """Prompt to initialize world state based on PAST context from a life summary."""
        from src.models import WorldState
        city = location.get('city', 'Unknown City')
        return f"""
        Target Historical Context:
        Date: {date}
        Time: {time}
        Location: {city}, {location.get('region', '?')}, {location.get('country', '?')}

        Relevant Summary of Character's Life/Situation around this time:
        {context_summary}

        Task: Create a detailed, rich, and plausible WorldState JSON object representing the world *at the specified historical date, time, and location*.
        Base the state on general knowledge of that time period and location, influenced by the provided character context summary.
        Infer conditions like weather (typical for the season/date/location), social/economic/political climate, major events (plausible for the time), etc.
        Populate ALL fields of the WorldState model, providing specific details. Use Celsius for temperature. Determine the day_phase based on the time. Use standard YYYY-MM-DD and HH:MM formats.

        Respond ONLY with the JSON object matching the WorldState schema:
        {json.dumps(get_model_schema(WorldState), indent=2)}
        """

    @staticmethod
    def initialize_immediate_environment_prompt(world_state: Dict, location_name: str) -> str:
        """Prompt to initialize a rich immediate environment."""
        from src.models import ImmediateEnvironment
        world_context_summary = f"""
        Time: {world_state.get('current_time')} on {world_state.get('current_date')} ({world_state.get('day_phase')})
        Location: {world_state.get('city_name')}, {world_state.get('district_neighborhood', 'Unknown Area')}
        Weather: {world_state.get('weather_condition')}, Temp: {world_state.get('temperature_c')}C
        Social Climate: {world_state.get('social_climate')}
        Major Events: {', '.join(world_state.get('major_events',[]))}
        """

        return f"""
        Current World State Context:
        {world_context_summary}

        Target Location Name: {location_name}

        Task: Create a detailed, rich, and plausible ImmediateEnvironment JSON object for the character entering '{location_name}' given the world state context.
        Determine the 'location_type'. Describe the physical conditions (lighting, noise, temp feel, humidity, smells), social environment (people types, specific NPCs with names/roles if plausible, density, atmosphere), available options/interactions (objects, services, exits, POIs), and sensory details (visuals, sounds).
        Populate ALL fields of the ImmediateEnvironment model. Be specific and immersive. If some details are unknowable, use reasonable defaults or indicate uncertainty where appropriate within the description strings.

        Respond ONLY with the JSON object matching the ImmediateEnvironment schema:
        {json.dumps(get_model_schema(ImmediateEnvironment), indent=2)}
        """

    @staticmethod
    def process_update_prompt(world_state: Dict, immediate_environment: Dict,
                            simulacra_action: Dict, reaction_profile: 'WorldReactionProfile',
                            initiator_reflection: Optional[str],
                            initiator_thought_process: Optional[str],
                            step_duration_minutes: int,
                            recent_narrative_updates: List[str]
                           ) -> str:
        """
        Create a prompt for processing an update based on the simulacra's action,
        reaction profile, agent's internal state, step duration, recent narratives,
        and **enforcing world constraints**.
        """
        from src.models import ImmediateEnvironment, WorldProcessUpdateResponse # Keep imports local

        profile_guidance = reaction_profile.get_description() # Get text description

        # --- Format Internal State Context ---
        internal_context = "[Agent's internal state not provided]"
        if initiator_reflection or initiator_thought_process:
            internal_context_lines = ["--- Agent's Internal State (Leading to Action) ---"]
            if initiator_reflection:
                internal_context_lines.append(f"Reflection: {initiator_reflection}")
            if initiator_thought_process:
                internal_context_lines.append(f"Thought Process: {initiator_thought_process}")
            internal_context_lines.append("--- End Internal State ---")
            internal_context = "\n".join(internal_context_lines)

        # --- Format Recent Narrative History ---
        narrative_history_context = "[No recent narrative history provided]"
        if recent_narrative_updates:
            history_lines = []
            reversed_history = list(reversed(recent_narrative_updates[-3:])) # Get last 3 max
            for i, narrative in enumerate(reversed_history):
                 history_lines.append(f"Narrative (T-{i+1}): {narrative[:150]}...")
            narrative_history_context = "--- Recent Narrative History ---\n" + "\n".join(history_lines) + "\n--- End History ---"

        # --- NEW: Format World Constraints Context ---
        current_time = world_state.get("current_time", "12:00")
        current_hour = 12
        try: current_hour = int(current_time.split(':')[0])
        except Exception: pass # Ignore errors, use default
        day_phase = world_state.get("day_phase", "Midday").lower()
        weather = world_state.get("weather_condition", "Clear").lower()
        is_night = current_hour < 6 or current_hour >= 22 or day_phase in ["night", "late night"]
        is_business_hours = 9 <= current_hour < 17 # Stricter 9-5 business hours

        world_constraints_context = f"""
        --- CURRENT WORLD STATE & RULES (ENFORCE THESE) ---
        Current Time: {current_time} ({day_phase})
        Weather: {weather}

        **Location Access Rules:**
        - Libraries, most stores, government buildings, schools are generally CLOSED at night ({is_night}). Assume standard business hours (e.g., 9am-5pm or 10am-8pm) unless otherwise specified.
        - Bars/Clubs may be open late; restaurants vary. Homes/apartments are accessible 24/7 to residents. Parks are often open but might be unsafe/empty at night.
        - Specific Location: '{immediate_environment.get('current_location_name', 'Unknown')}' (Type: {immediate_environment.get('location_type', 'Unknown')})

        **Action Constraints:**
        - Travel takes time ({step_duration_minutes} min available). Cannot teleport.
        - Severe weather ({'Yes' if weather in ['heavy rain', 'thunderstorm', 'blizzard', 'hurricane', 'tornado'] else 'No'}) may prevent travel or outdoor actions.
        - Actions must be plausible for the location (e.g., cannot 'use library computer' if not at a library).
        --- END WORLD STATE & RULES ---
        """
        # --- END: World Constraints Context ---

        agent_input = f"""
        Current World State:
        {json.dumps(world_state, indent=2, default=str)}

        Current Immediate Environment:
        {json.dumps(immediate_environment, indent=2, default=str)}

        {narrative_history_context}

        {world_constraints_context} # <<< INSERTED CONSTRAINTS HERE

        Simulacra Action Attempted (over {step_duration_minutes} minutes):
        {json.dumps(simulacra_action, indent=2, default=str)}

        {internal_context}

        WORLD REACTION PROFILE GUIDANCE (Interpret these guidelines):
        {profile_guidance}

        **Task:** Analyze the outcome of the *attempted* simulacra action over the specified duration ({step_duration_minutes} minutes).
        **Crucially, you MUST apply the WORLD STATE & RULES.** If the action is impossible due to time, location, weather, or plausibility constraints, simulate the *failure* and its immediate consequences.

        **Consider:**
        1. **Constraint Check:** FIRST, check if the action is possible based on the WORLD STATE & RULES.
           - If **IMPOSSIBLE** (e.g., trying to enter a closed library at 1 AM):
             - The `updated_environment` MUST reflect the character's state *after failing* (e.g., still outside the library). `current_location_name` should NOT change to the target if entry failed.
             - `observations` MUST clearly state the failure and the reason (e.g., "The library doors are locked.", "A sign indicates it's closed.").
             - `consequences` should reflect the failure (e.g., "Failed to enter the library.").
             - Reflect NPC reactions *if* they observe the failed attempt.
           - If **POSSIBLE**: Proceed to simulate the successful action's effects.
        2. Plausibility over {step_duration_minutes} min.
        3. Direct environmental changes (objects moved, sounds, states).
        4. Social responses from NPCs present (update 'specific_npcs_present' list: names/roles/status). Consider initiator's intent (reflection/thought process).
        5. Natural progression of NPCs/world, consistent with history.
        6. Changes to atmosphere, opportunities, POIs.
        7. Broader world state changes ONLY if action has significant ripple effects AND is possible.

        **Output Generation:** Generate a JSON response containing:
        - 'updated_environment': The *complete* ImmediateEnvironment object state *at the end* of the {step_duration_minutes} min duration, reflecting the *actual outcome* (success or failure based on constraints). Ensure `specific_npcs_present` is a list of dicts: {{'name': 'string', 'role': 'string', 'status': 'string'}}.
        - 'world_state_changes': Dictionary of *changed* WorldState keys (only if applicable and action was possible). {{}} if none.
        - 'consequences': LIST of strings describing direct, notable consequences (reflecting success or failure).
        - 'observations': LIST of strings describing what character perceives *after* the attempt (e.g., success details or failure reason).

        Ensure the response strictly adheres to the specified JSON structure.
        """

        # The JSON output structure definition remains the same
        prompt_json_output = "\nRespond ONLY with the following JSON format (do NOT include comments): " + json.dumps(
            {
                "updated_environment": { # Placeholder: Provide the full ImmediateEnvironment object here
                    "current_location_name": "string",
                    "location_type": "string",
                    # ... other ImmediateEnvironment fields ...
                    "specific_npcs_present": [
                        {"name": "string", "role": "string", "status": "string"},
                         # ... potentially more NPCs ...
                    ],
                    # ... rest of ImmediateEnvironment fields ...
                    "attention_drawing_elements": ["string"]
                },
                "world_state_changes": {"key": "new_value", "...": "..."}, # Changes only, or {}
                "consequences": ["List of strings"],
                "observations": ["List of strings"]
            }
        , indent=2)
        return agent_input + prompt_json_output

    @staticmethod
    def reflect_on_situation_prompt(observations: str, immediate_environment: Dict, persona_state: Dict) -> str:
        """Prompt for character reflection."""
        from src.models import DayResponse # For schema
        persona_summary_lines = [
            f"- You are {persona_state.get('name', 'Unknown')}, {persona_state.get('age', '?')} years old, working as a {persona_state.get('occupation', '?')}." ,
            f"- Your personality traits include: {', '.join(persona_state.get('personality_traits', ['unknown']))}.",
            f"- Your current goals are: {', '.join(persona_state.get('goals', ['none specified']))}.",
            f"- Physically you feel: {persona_state.get('current_state', {}).get('physical', 'normal')}.",
            f"- Emotionally you feel: {persona_state.get('current_state', {}).get('emotional', 'neutral')}.",
            f"- Mentally you are: {persona_state.get('current_state', {}).get('mental', 'aware')}.",
            f"- Recent memories: {'; '.join(persona_state.get('memory', {}).get('short_term', ['nothing specific']))}."
        ]
        persona_summary = "\n".join(persona_summary_lines)

        reflection_prompt = f"""
        Current Observations:
        {observations if observations else "You observe your surroundings."}

        Detailed Immediate Environment:
        {json.dumps(immediate_environment, indent=2, default=str)}

        Your Persona Summary:
        {persona_summary}

        Task: Reflect deeply on the current situation. Consider your persona (traits, goals, memories, current state) and everything you are observing in the detailed environment. What are your internal thoughts, interpretations, feelings, and potential intentions right now? What stands out to you? What connections do you make?
        """

        prompt_json_output = "\nRespond ONLY with the following JSON format: " + json.dumps(
            get_model_schema(DayResponse) # Schema for {'reflect': '...'}
        , indent=2)
        return reflection_prompt + prompt_json_output

    @staticmethod
    def analyze_emotions_prompt(situation: str, current_emotional_state: str) -> str:
        """Prompt for emotional analysis."""
        from src.models import EmotionAnalysisResponse # For schema
        prompt_template = f"""
        Situation & Your Reflection:
        {situation}

        Your previous primary emotional state:
        {current_emotional_state}

        Task: Analyze your emotional response to this situation and your reflection. Identify the primary emotion you are feeling *now*, its intensity (Low, Medium, High), any secondary emotion, and provide a concise summary of your *new* overall emotional state in the 'emotional_update' field.
        """
        prompt_json_output = "\nRespond ONLY with the following JSON format: " + json.dumps(
            get_model_schema(EmotionAnalysisResponse)
        , indent=2)
        return prompt_template + prompt_json_output

    @staticmethod
    def decide_action_prompt(
        reflection: str,
        emotional_analysis: Dict[str, Any],
        goals: List[str],
        immediate_environment: Dict[str, Any],
        persona_state: Dict[str, Any],
        retrieved_background: str,
        step_duration_minutes: int,
        last_action_taken: Optional[str],
        world_state: Dict # <<< ADDED world_state for grounding context >>>
    ) -> str:
        """
        Prompt for deciding the next action, with grounding rules and anti-repetition.
        """
        from src.models import ActionDecisionResponse
        persona_name = persona_state.get('name', 'Unknown')
        traits_str = ", ".join(persona_state.get('personality_traits', ['Unknown']))
        allowed_actions_list = list(AllowedActionVerbs.__args__)
        allowed_actions_str = ", ".join(f"'{action}'" for action in allowed_actions_list)

        background_section = "[No relevant background retrieved or available]"
        if retrieved_background and not retrieved_background.startswith("["): background_section = retrieved_background.strip()
        duration_guidance = ""
        if step_duration_minutes <= 5: duration_guidance = "Short step: brief action."
        elif step_duration_minutes <= 30: duration_guidance = "Moderate step: focused activity."
        else: duration_guidance = f"Long step ({step_duration_minutes} min): significant activity."

        last_action_context = f"Your immediate previous action was: {last_action_taken}" if last_action_taken else "This is the first action."

        # --- NEW: Grounding in Reality Context ---
        grounding_context = f"""
        --- GROUNDING IN REALITY (IMPORTANT!) ---
        Current Time: {world_state.get('current_time', 'Unknown')} ({world_state.get('day_phase', '?')})
        Current Location: {immediate_environment.get('current_location_name', 'Unknown')} ({immediate_environment.get('location_type', '?')})
        Weather: {world_state.get('weather_condition', 'Unknown')}

        **RULES TO FOLLOW:**
        1. **Time Awareness:** Is your destination likely open now ({world_state.get('current_time', '?')})? Libraries, stores, schools usually have specific hours. Avoid planning actions for closed locations.
        2. **Location Specificity:** If moving, be specific (e.g., "Joe's Diner on Main St").
        3. **Weather:** Is travel/outdoor activity safe/practical in the current weather? ({world_state.get('weather_condition', 'Unknown')})
        4. **Plausibility:** Is the action feasible in your *current* location? (e.g., you can't 'use' a library computer if you're in a park).
        5. **Travel Time:** You have {step_duration_minutes} mins. Moving takes time.

        *The simulation will prevent impossible actions later, but you should PLAN realistically NOW.*
        --- END GROUNDING ---
        """
        # --- END: Grounding Context ---

        prompt_template = f"""
        You are {persona_name}, a {persona_state.get('age', '?')}-year-old {persona_state.get('occupation', 'Unknown')}.
        Your core personality traits are: {traits_str}.

        --- TIME & RECENT ACTION CONTEXT ---
        The current simulation step duration is {step_duration_minutes} minutes.
        {duration_guidance}
        {last_action_context}
        **RULE: You MUST choose an action DIFFERENT from your last one ('{last_action_taken or 'N/A'}').**

        {grounding_context} # <<< INSERTED GROUNDING RULES HERE

        --- YOUR CURRENT STATE & PERCEPTION ---
        Your Internal Reflection (based on situation AFTER last action):
        {reflection.strip()}

        Your Current Emotional State Analysis:
        {json.dumps(emotional_analysis, indent=2, default=str)}

        Your Current Goals:
        {json.dumps(goals, indent=2, default=str)}

        Relevant Background from Life Summary (Memory):
        {background_section}

        Current Immediate Environment (What you perceive):
        {json.dumps(immediate_environment, indent=2, default=str)}

        --- TASK ---
        Based on your internal state, perception, background, goals, the step duration, the **strict rule against repeating your last action**, and the **GROUNDING IN REALITY rules**, decide your next primary action.

        **ALLOWED ACTIONS:** You MUST choose the 'action' verb from this specific list: {allowed_actions_str}.
        *   Use 'talk' for dialogue.
        *   Use 'move' for changing location.
        *   Use 'wait' for pausing.
        *   Use 'observe' for focused looking/listening.
        *   Use 'use' for interacting with items (operating, consuming, manipulating).
        *   Use 'think' for internal planning, analysis, or recall.
        *   Use 'search' for actively looking for something.
        *   Use 'read'/'write' for text interaction.
        *   Use 'rest' for physical rest/sleep.
        *   Use 'get'/'drop' for picking up/placing items.
        *   Use 'other' ONLY if the intended action genuinely does not fit any other category. Provide clear details if using 'other'.

        **CRITICAL EVALUATION:** Consider the outcome of your last action AND the grounding rules. If analyzing clues feels stuck or unrealistic, actively choose a DIFFERENT, *plausible* kind of action (e.g., interact, move, rest, plan, use an object differently).

        1. Explain 'thought_process': Justify why this NEW, REALISTIC action is the best choice now, considering the anti-repetition rule and grounding constraints.
        2. State the 'action' clearly.
        3. Provide 'action_details' or null.

        Ensure the action is plausible and selected ONLY from the allowed list.
        """

        # Generate JSON output format using get_model_schema
        try:
            json_output_format = get_model_schema(ActionDecisionResponse)
            prompt_json_output = "\n\nRespond ONLY with the following JSON format:\n" + json.dumps(json_output_format, indent=2)
        except Exception as e:
            prompt_json_output = """
Respond ONLY with the following JSON format:
{
    "thought_process": "string",
    "action": "string",
    "action_details": {
        "target": "string | null",
        "utterance": "string | null",
        "target_location": "string | null",
        "item": "string | null",
        "manner": "string | null",
        "duration": "string | null"
    } | null
}"""
        return prompt_template + prompt_json_output

    @staticmethod
    def generate_initial_narrative_prompt(persona: Dict, world_state: Dict, immediate_environment: Dict) -> str:
        """Generates a prompt to create the initial narrative context for the simulation start."""
        persona_context = f"""
        Name: {persona.get('Name', persona.get('name','Unknown'))}
        Age: {persona.get('Age', persona.get('age','Unknown'))}
        Occupation: {persona.get('Occupation', persona.get('occupation','Unknown'))}
        Personality traits: {', '.join(persona.get('Personality_Traits', persona.get('personality_traits', ['Unknown'])))}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        Physical state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        Emotional state: {persona.get('current_state', {}).get('emotional', 'Unknown')}
        Mental state: {persona.get('current_state', {}).get('mental', 'Unknown')}
        Short-term memories: {', '.join(persona.get('memory', {}).get('short_term', ['Unknown']))}
        Long-term memories: {', '.join(persona.get('memory', {}).get('long_term', ['Unknown']))}
        """.strip()

        world_context = f"""
        Time: {world_state.get('current_time', 'Unknown')}
        Date: {world_state.get('current_date', 'Unknown')}
        City: {world_state.get('city_name', 'Unknown')}
        Weather: {world_state.get('weather_condition', 'Unknown')}
        Social climate: {world_state.get('social_climate', 'Unknown')}
        Major events: {', '.join(world_state.get('major_events', ['None']))}
        """.strip()

        location_name = immediate_environment.get('current_location_name', 'Unknown Location')

        return f"""
        Task: Create a brief narrative context (2-3 paragraphs) explaining how the character arrived at their current situation at the start of the simulation.

        Character Information:
        {persona_context}

        World Information:
        {world_context}

        Current Location Name: {location_name}

        The narrative should explain:
        1. Why the character is plausibly at this specific location ({location_name}) right now ({world_state.get('current_time')}).
        2. What they were likely doing earlier today leading up to this moment.
        3. What their immediate concerns, thoughts, or state of mind might be, connecting to their goals/personality.
        4. Briefly touch upon how their current emotional/physical state relates to recent events.

        Make the narrative realistic, engaging, and specific to this character and situation. It should provide a smooth entry point into the simulation. Avoid stating "The simulation starts...". Write from a 3rd person perspective.
        Respond ONLY with the narrative text.
        """

    # <<< Kept generate_narrative_update_prompt as is >>>
    @staticmethod
    def generate_narrative_update_prompt(persona: Dict, previous_actions: List[str],
                                         world_state: Dict, immediate_environment: Dict,
                                         consequences: List[str], observations: List[str],
                                         step_duration_minutes: int,
                                         recent_narrative_updates: List[str],
                                         initiator_reflection: Optional[str],
                                         initiator_thought_process: Optional[str],
                                         day_arc: Optional[str] = None) -> str: # <<< Added day_arc parameter
        """Generates a prompt to create a narrative update, including the day's arc."""
        # NOTE: Still not explicitly adding reflection/thought here unless needed later.
        persona_context = f"Name: {persona.get('name', '?')}, Feeling: {persona.get('current_state',{}).get('emotional', '?')}"
        world_context = f"Time: {world_state.get('current_time')}, Loc: {immediate_environment.get('current_location_name', '?')}, Weather: {world_state.get('weather_condition')}"
        last_simulacra_action = previous_actions[-1] if previous_actions else "No specific action listed."
        consequences_text = "\n".join([f"- {c}" for c in consequences]) if consequences else "None listed."
        observations_text = "\n".join([f"- {o}" for o in observations]) if observations else "None listed."
        narrative_focus = ""
        if step_duration_minutes <= 5: narrative_focus = "Focus on immediate action/result."
        elif step_duration_minutes <= 30: narrative_focus = f"Describe progression over {step_duration_minutes} mins."
        else: narrative_focus = f"Summarize events over {step_duration_minutes} mins."
        narrative_history_context = "[No recent narrative history provided]"
        last_narrative_snippet = ""
        if recent_narrative_updates:
            history_lines = []; reversed_history = list(reversed(recent_narrative_updates[-3:]))
            if reversed_history: last_narrative_snippet = reversed_history[0]
            for i, narrative in enumerate(reversed_history): history_lines.append(f"Narrative (T-{i+1}): {narrative[:150]}...")
            narrative_history_context = "Recent Narrative History:\n" + "\n".join(history_lines)
        anti_stagnation_instruction = ""
        if ("analyz" in last_simulacra_action.lower() and "analyz" in last_narrative_snippet.lower()) or \
           ("mystery" in last_simulacra_action.lower() and "mystery" in last_narrative_snippet.lower()):
             anti_stagnation_instruction = "**NARRATIVE CONTINUITY:** Previous narrative seemed similar. Ensure this update shows a *change*, *conclusion*, *interruption*, or clear *shift in focus*. Avoid repeating unresolved contemplation."

        # <<< ADDED: Day Arc Context Formatting >>>
        day_arc_context = f"\nUnderlying Theme/Arc for Today: {day_arc}" if day_arc else ""

        return f"""
        Task: Write narrative update (1-2 paragraphs) for last {step_duration_minutes} minutes.

        Character Context (end): {persona_context}
        World Context (end): {world_context}
        {day_arc_context} # <<< INSERTED DAY ARC CONTEXT HERE

        --- Key Inputs ---
        Action Taken (start): {last_simulacra_action}
        Resulting Consequences (end): {consequences_text}
        New Observations (end): {observations_text}
        {narrative_history_context}
        --- End Key Inputs ---

        Instructions: Write 3rd person narrative for past {step_duration_minutes} mins. Start with action initiation. {narrative_focus}
        **Subtly weave the 'Underlying Theme/Arc for Today' into the narrative if relevant to the action/outcome, but don't force it unnaturally.**
        **IMPORTANT:** The narrative must reflect the *actual outcome*, including any failures due to world constraints (e.g., "tried to enter the library, but found it locked").
        {anti_stagnation_instruction}
        Use history for continuity but ensure story progresses. Incorporate consequences/observations accurately. Show, don't tell.
        Respond ONLY with narrative text.
        """