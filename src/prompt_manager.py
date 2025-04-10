import json
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union
from src.models import (
    WorldState, ImmediateEnvironment, WorldReactionProfile, DayResponse,
    EmotionAnalysisResponse, ActionDecisionResponse, WorldStateResponse,
    ImmediateEnvironmentResponse
) # Import models from src

class PromptManager:
    """Manages all prompts used in the simulation."""

    @staticmethod
    def analyze_information_prompt(information: str, context: Optional[Dict] = None) -> str:
        context_str = json.dumps(context, indent=2) if context else "No context provided."
        return f"""
        Information to analyze:
        {information}

        Context (previous world state or other relevant info):
        {context_str}

        Based on this, update the world state. Focus on making realistic and plausible changes to the world.
        Return the updated world state as a JSON object.
        """

    @staticmethod
    def initialize_world_state_prompt(news_results: str, config: Dict) -> str:
        return f"""
        News results:
        {news_results}

        World configuration:
        {json.dumps(config, indent=2)}

        Based on this information, create a comprehensive world state with the following elements:
        - current_time: The current time in 24-hour format
        - current_date: The current date in YYYY-MM-DD format
        - city_name: The name of the city
        - country_name: The name of the country
        - region_name: The name of the region or state
        - weather_condition: Current weather (sunny, cloudy, rainy, etc.)
        - temperature: Current temperature with units
        - forecast: Brief weather forecast for next 24 hours
        - social_climate: General social mood and atmosphere
        - economic_condition: Current economic situation
        - major_events: List of significant events happening in the area
        - local_news: List of recent local news items
        - transportation_status: Status of public transit and traffic
        - utility_status: Status of power, water, internet services
        - public_announcements: List of official announcements
        - trending_topics: List of topics people are discussing
        - current_cultural_events: List of ongoing cultural events
        - sports_events: List of sports events
        - public_health_status: Current public health situation
        - public_safety_status: Current safety and security situation

        Create a realistic and detailed world state based on the news and configuration.
        Return the updated world state as a JSON object using the WorldStateResponse schema.
        """

    @staticmethod
    def initialize_immediate_environment_prompt(world_state: Dict, location: str) -> str:
        return f"""
        World State:
        {json.dumps(world_state, indent=2)}

        Current Location: {location}

        Based on this information, create a detailed immediate environment for this location with the following elements:
        - current_location_name: Specific name of the current location
        - location_type: Type of location (restaurant, park, office, etc.)
        - indoor_outdoor: Whether the location is indoor or outdoor
        - noise_level: How loud or quiet it is
        - lighting: Lighting conditions
        - temperature_feeling: How the temperature feels
        - air_quality: Quality of the air
        - present_people: Types of people present
        - crowd_density: How crowded the location is
        - social_atmosphere: Social mood of the location
        - ongoing_activities: Activities happening around
        - nearby_objects: Objects that can be interacted with
        - available_services: Services available at this location
        - exit_options: Ways to leave this location
        - interaction_opportunities: Opportunities for interaction
        - visible_features: Notable visible features
        - audible_sounds: Sounds that can be heard
        - noticeable_smells: Smells that can be detected
        - seating_availability: Availability of seating
        - food_drink_options: Available food and drinks
        - restroom_access: Access to restrooms
        - recent_changes: Recent changes to the environment
        - ongoing_conversations: Topics being discussed nearby
        - attention_drawing_elements: Things that draw attention

        Create a realistic and detailed immediate environment that would be consistent with the world state and location.
        Return the updated environment as a JSON object using the ImmediateEnvironmentResponse schema.
        """

    @staticmethod
    def process_update_prompt(world_state: Dict, immediate_environment: Dict,
                            simulacra_action: Dict, reaction_profile: WorldReactionProfile) -> str:
        """Create a prompt for processing an update based on the simulacra's action."""

        # Create guidance based on the reaction profile
        profile_guidance = f"""
        WORLD REACTION PROFILE:

        1. Consequence Severity: {reaction_profile.consequence_severity.upper()}
        - {"Actions have minimal consequences" if reaction_profile.consequence_severity == "mild" else
        "Actions have normal, expected consequences" if reaction_profile.consequence_severity == "moderate" else
        "Actions have amplified consequences"}

        2. Social Responsiveness: {reaction_profile.social_responsiveness.upper()}
        - {"People tend to be unfriendly or antagonistic" if reaction_profile.social_responsiveness == "hostile" else
        "People react normally based on circumstances" if reaction_profile.social_responsiveness == "neutral" else
        "People tend to be helpful and accommodating" if reaction_profile.social_responsiveness == "friendly" else
        "People largely ignore the character"}

        3. Environmental Stability: {reaction_profile.environmental_stability.upper()}
        - {"Environment changes little, predictable" if reaction_profile.environmental_stability == "stable" else
        "Environment changes at a normal, realistic pace" if reaction_profile.environmental_stability == "dynamic" else
        "Environment frequently changes in unexpected ways"}

        4. Coincidence Frequency: {reaction_profile.coincidence_frequency.upper()}
        - {"Few coincidences, highly realistic cause-and-effect" if reaction_profile.coincidence_frequency == "rare" else
        "Normal level of coincidences" if reaction_profile.coincidence_frequency == "occasional" else
        "Many coincidences (meeting just the right person, etc.)"}

        5. Challenge Level: {reaction_profile.challenge_level.upper()}
        - {"Obstacles are simpler than expected" if reaction_profile.challenge_level == "easy" else
        "Obstacles require appropriate effort" if reaction_profile.challenge_level == "normal" else
        "Obstacles require exceptional effort"}

        6. Narrative Tone: {reaction_profile.narrative_tone.upper()}
        - {"Humorous situations tend to arise" if reaction_profile.narrative_tone == "comedic" else
        "Emotionally significant events occur" if reaction_profile.narrative_tone == "dramatic" else
        "Everyday, ordinary events predominate" if reaction_profile.narrative_tone == "mundane" else
        "Tense, uncertain situations develop"}

        7. Opportunity Frequency: {reaction_profile.opportunity_frequency.upper()}
        - {"Few new opportunities present themselves" if reaction_profile.opportunity_frequency == "scarce" else
        "Realistic number of opportunities" if reaction_profile.opportunity_frequency == "normal" else
        "Many opportunities appear"}

        8. Serendipity: {reaction_profile.serendipity.upper()}
        - {"Rarely stumble upon helpful things" if reaction_profile.serendipity == "low" else
        "Occasionally find useful things by chance" if reaction_profile.serendipity == "medium" else
        "Frequently make fortunate discoveries"}

        9. World Awareness: {reaction_profile.world_awareness.upper()}
        - {"Character's actions go largely unnoticed" if reaction_profile.world_awareness == "invisible" else
        "Appropriate recognition of actions" if reaction_profile.world_awareness == "normal" else
        "Character's actions receive unusual attention"}

        10. Karmic Response: {reaction_profile.karmic_response.upper()}
        - {"Good/bad actions quickly lead to rewards/consequences" if reaction_profile.karmic_response == "strong" else
        "Some connection between moral choices and outcomes" if reaction_profile.karmic_response == "moderate" else
        "No special connection between moral choices and outcomes"}
        """

        agent_input = f"""
        Current World State:
        {json.dumps(world_state, indent=2)}

        Current Immediate Environment:
        {json.dumps(immediate_environment, indent=2)}

        Simulacra action:
        {json.dumps(simulacra_action, indent=2)}

        {profile_guidance}

        Analyze how the immediate environment should react to this action, considering:
        1. Physical laws and plausibility
        2. Environmental responses
        3. Social responses from people present
        4. Any changes to the surroundings
        5. New opportunities or limitations created by the action

        Focus primarily on updating the immediate environment, as this is what's directly affected by the simulacra's action.
        Only update the world state if the action would realistically have broader implications.

        For consequences and observations, provide a LIST of specific, detailed statements.
        Each consequence or observation should be a complete sentence describing one specific effect or thing noticed.
        """

        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
            {
                "updated_environment": "The updated immediate environment after the action (JSON object following ImmediateEnvironment schema)",
                "world_state_changes": "Any changes to the world state (JSON object with only the changed fields, if applicable)",
                "consequences": ["List of specific consequences of the action, each as a complete sentence"],
                "observations": ["List of specific things the simulacra would observe, each as a complete sentence"]
            }
        )
        return agent_input + prompt_json_output

    @staticmethod
    def reflect_on_situation_prompt(observations: str, persona_state: Dict) -> str:
        reflection_prompt = f"""
        {observations}

        Consider your personality as {persona_state['name']}, a {persona_state['age']}-year-old {persona_state['occupation']} who is {', '.join(persona_state['personality_traits'])},
        your current physical state ({persona_state['current_state']['physical']}),
        emotional state ({persona_state['current_state']['emotional']}),
        and mental state ({persona_state['current_state']['mental']}).

        Think about your goals: {', '.join(persona_state['goals'])}

        Consider your short-term memories of {', '.join(persona_state['memory']['short_term'])} and long-term experiences of {', '.join(persona_state['memory']['long_term'])}.
        """

        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
            DayResponse.model_json_schema()["properties"] # Use schema for format
        )
        return reflection_prompt + prompt_json_output

    @staticmethod
    def analyze_emotions_prompt(situation: str, current_emotional_state: str) -> str:
        prompt_template = f"""
Situation:
{situation}

Current emotional state:
{current_emotional_state}
        """

        prompt_json_day = "\nRespond in the following JSON format: " + json.dumps(
             EmotionAnalysisResponse.model_json_schema()["properties"] # Use schema for format
        )
        return prompt_template + prompt_json_day

    @staticmethod
    def decide_action_prompt(reflection: str, emotional_analysis: Dict, goals: List[str], immediate_environment: Dict) -> str:
        prompt_template = f"""
        Reflection:
        {reflection}

        Emotional Analysis:
        {json.dumps(emotional_analysis, indent=2)}

        Considering your goals:
        {json.dumps(goals, indent=2)}

        Your immediate environment:
        {json.dumps(immediate_environment, indent=2)}

        What action will you take? Consider your personality traits, current state, available options in your environment, and what would be most consistent with who you are.
        """

        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
             ActionDecisionResponse.model_json_schema()["properties"] # Use schema for format
        )
        return prompt_template + prompt_json_output