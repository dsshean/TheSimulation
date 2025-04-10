import asyncio
import datetime as dt # Use alias to avoid conflict with datetime class
import json
import logging
import os
import re
import time
from datetime import datetime # Keep this for datetime objects
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import requests
import yaml
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field, model_validator
from rich.console import Console

# Import from other modules in src
from src.models import (
    WorldReactionProfile, ImmediateEnvironment, WorldState,
    WorldStateResponse, ImmediateEnvironmentResponse
)
from src.prompt_manager import PromptManager
from src.llm_service import LLMService

logger = logging.getLogger(__name__)


class WorldEngine:
    """Manages the environment and physical laws of the simulated world."""

    def __init__(self, world_config_path: str = "world_config.yaml", load_state: bool = True,
                console: Console = None, reaction_profile: Union[str, Dict, WorldReactionProfile] = "balanced"):
        """
        Initialize the WorldEngine.

        Args:
            world_config_path: Path to the world configuration file
            load_state: Whether to load existing state
            console: Rich console for output
            reaction_profile: How the world reacts to the character's actions
                            Can be a profile name, a dict of settings, or a WorldReactionProfile
        """
        # Initialize console
        self.console = console or Console()
        self.world_config_path = world_config_path
        self.state_path = "world_state.json"
        self.history = []
        self.world_state = None
        self.immediate_environment = None
        self.llm_service = LLMService()
        self.search_tool = DuckDuckGoSearchRun() # Initialize search tool

        # Set reaction profile
        if isinstance(reaction_profile, str):
            self.reaction_profile = WorldReactionProfile.create_profile(reaction_profile)
        elif isinstance(reaction_profile, dict):
            self.reaction_profile = WorldReactionProfile(**reaction_profile)
        elif isinstance(reaction_profile, WorldReactionProfile):
            self.reaction_profile = reaction_profile
        else:
            logger.warning(f"Invalid reaction profile type. Using 'balanced' profile.")
            self.reaction_profile = WorldReactionProfile.create_profile("balanced")

        logger.info(f"World Engine initialized with reaction profile: {self.reaction_profile.model_dump()}")

        # Check if this is a continuation or new simulation
        if load_state and os.path.exists(self.state_path):
            logger.info("Loading existing world state...")
            self.load_state()

    def _search_news(self, query: str) -> str:
        """Search for news using DuckDuckGo via Langchain Tool."""
        try:
            logger.info(f"Performing news search with query: {query}")
            results = self.search_tool.run(query)
            logger.info(f"News search results (first 100 chars): {results[:100]}...")
            # Basic formatting, Langchain tool might return a more narrative result
            if not results or "No good DuckDuckGo Search Result found" in results:
                 return "No relevant news found."
            # Attempt to split into items if possible, otherwise return as block
            items = re.split(r'\n\s*-\s*|\n\d+\.\s*', results.strip()) # Split by newline-dash or newline-number
            if len(items) > 1:
                 return "\n".join([f"- {item.strip()}" for item in items if item.strip()])
            else:
                 return results # Return as is if no clear list format

        except Exception as e:
            logger.error(f"Error during news search with query '{query}': {e}")
            return "Error fetching news."


    async def _gather_comprehensive_news(self, location: Dict) -> str:
        """Gather comprehensive news about a location from multiple queries."""
        city = location.get("city", "Seattle")
        country = location.get("country", "USA")

        # Create multiple search queries for different aspects
        queries = [
            f"current news {city} {country}",
            f"weather forecast {city} {country}",
            f"events in {city} this week",
            f"economic news {city}",
            f"social issues {city}",
            f"transportation {city} status",
            f"cultural events {city}",
            f"sports news {city}"
        ]

        # Gather news for each query (can run concurrently)
        tasks = [asyncio.to_thread(self._search_news, query) for query in queries]
        results_list = await asyncio.gather(*tasks)

        news_results_dict = {}
        for i, query in enumerate(queries):
            category = query.split()[1].capitalize() if ' ' in query else query.capitalize()
            results = results_list[i]
            if results and results != "No relevant news found." and results != "Error fetching news.":
                news_results_dict.setdefault(category, []).append(results)

        # Format the results
        formatted_news = []
        for category, items in news_results_dict.items():
             formatted_news.append(f"--- {category} News ---\n" + "\n".join(items) + "\n")

        return "\n".join(formatted_news) if formatted_news else "No comprehensive news found."


    # Update the _determine_plausible_location method to use the LLM's knowledge
    async def _determine_plausible_location(self, config: Dict) -> str:
        """Use the LLM to determine a plausible location based on the world config and persona."""
        city = config.get("location", {}).get("city", "Seattle")
        country = config.get("location", {}).get("country", "USA")

        # Get persona information if available
        persona_path = "simulacra_state.json"
        persona = {}

        if os.path.exists(persona_path):
            try:
                with open(persona_path, 'r') as file:
                    data = json.load(file)
                    if "persona" in data:
                        persona = data["persona"]
                        self.console.print(f"[green]Loaded existing persona: {persona.get('name', 'Unknown')}[/green]")
            except Exception as e:
                logger.error(f"Error loading persona: {e}")

        # If no persona exists, generate one based on the location
        if not persona:
            self.console.print(f"[yellow]No existing persona found. Generating a persona based on {city}, {country}...[/yellow]")
            persona = await self._generate_persona_for_location(city, country)

            # Save the generated persona
            try:
                with open(persona_path, 'w') as file:
                    json.dump({"persona": persona, "history": []}, file, indent=2)
                self.console.print(f"[green]Generated and saved new persona: {persona.get('name', 'Unknown')}[/green]")
            except Exception as e:
                logger.error(f"Error saving generated persona: {e}")

        # Create a prompt for the LLM to determine a plausible location
        prompt = f"""
        Based on the following information, determine a realistic and specific location where the character might be at the start of the simulation:

        City: {city}
        Country: {country}

        Character information:
        Name: {persona.get('name', 'Unknown')}
        Age: {persona.get('age', 'Unknown')}
        Occupation: {persona.get('occupation', 'Unknown')}
        Personality traits: {', '.join(persona.get('personality_traits', ['Unknown']))}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        Current state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        """

        prompt += """
        Consider the character's occupation, age, and personality to determine where they might realistically be.
        Choose a specific location (like a particular coffee shop, park, office, etc.) that would exist in this city.
        The location should be realistic and specific, not generic.

        Return only the name of the location, nothing else. Just the location name as a single string.
        """

        self.console.print(f"[yellow]Determining plausible location in {city}...[/yellow]")

        try:
            # Use the LLM to determine a plausible location
            # No response model needed here, just expect raw text
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Determine a realistic and specific location name for the character based on their profile and the city. Respond with only the location name.",
                temperature=0.7,  # Slightly higher temperature for creativity
                response_model=None # Expecting raw text
            )

            # Extract the location from the response
            location = "Unknown Location"
            if isinstance(response, dict):
                 if "text" in response:
                      location = response["text"].strip()
                 elif "error" in response:
                      logger.error(f"LLM error determining location: {response['error']}")
                 else:
                      logger.warning(f"Unexpected LLM response format for location: {response}")
                      location = str(response).strip() # Fallback: convert dict to string
            elif isinstance(response, str):
                 location = response.strip()
            else:
                 location = str(response).strip()

            # Clean up the response
            location = location.strip('"\'').strip().split('\n')[0] # Take first line

            # Basic validation: avoid overly long or complex responses
            if len(location) > 100 or len(location.split()) > 10:
                logger.warning(f"LLM location response seems too complex, using fallback. Response: {location}")
                raise ValueError("LLM response for location too complex")

            self.console.print(f"[green]Determined location:[/green] {location}")
            return location if location else f"Coffee shop in {city}" # Ensure we don't return empty string
        except Exception as e:
            logger.error(f"Error determining plausible location: {e}")
            # Fallback to a generic location
            fallback_location = f"Coffee shop in {city}"
            self.console.print(f"[red]Error determining location, using fallback:[/red] {fallback_location}")
            return fallback_location

    async def _generate_persona_for_location(self, city: str, country: str) -> Dict:
        """Generate a persona that would plausibly be in the given location."""

        prompt = f"""
        Create a realistic persona for a character in {city}, {country}.

        Consider the demographics, culture, and lifestyle of this location to create a believable character.

        Return the persona as a JSON object with the following structure:
        {{
            "name": "Character's name",
            "age": age as number,
            "occupation": "Character's job",
            "personality_traits": ["trait1", "trait2", "trait3", "trait4"],
            "goals": ["goal1", "goal2"],
            "current_state": {{
                "physical": "Description of physical state",
                "emotional": "Description of emotional state",
                "mental": "Description of mental state"
            }},
            "memory": {{
                "short_term": ["recent memory 1", "recent memory 2"],
                "long_term": ["long-term memory 1", "long-term memory 2"]
            }}
        }}

        Make the character interesting but realistic for this location. Consider local industries, cultural context, and daily life.
        Respond ONLY with the JSON object, no other text.
        """

        system_instruction = f"Create a realistic persona for a character living in {city}, {country} as a structured JSON object."

        default_persona = {
            "name": f"Local Resident of {city}",
            "age": 30,
            "occupation": "Professional",
            "personality_traits": ["adaptable", "observant", "practical", "curious"],
            "goals": ["navigate daily life", "pursue personal interests"],
            "current_state": {
                "physical": "Well-rested",
                "emotional": "Neutral",
                "mental": "Alert and aware"
            },
            "memory": {
                "short_term": [f"Recent activities in {city}"],
                "long_term": [f"Life experiences in {country}"]
            }
        }

        try:
             # Expecting a JSON dictionary directly
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.7,
                response_model=None # Expecting JSON, but parse manually
            )

            # Check if the response is a dictionary and looks like our persona
            if isinstance(response, dict) and "name" in response and "age" in response:
                persona = response
                # Simple validation
                if not all(k in persona for k in ["name", "age", "occupation", "personality_traits", "goals", "current_state", "memory"]):
                     logger.warning("Generated persona missing expected keys. Using default.")
                     return default_persona
                self.console.print(f"[green]Successfully generated persona for {persona['name']}[/green]")
                return persona
            elif isinstance(response, dict) and "error" in response:
                 logger.error(f"LLM error generating persona: {response['error']}. Raw: {response.get('raw_response')}")
                 self.console.print(f"[red]Using default persona due to LLM error.[/red]")
                 return default_persona
            else:
                logger.warning(f"Unexpected LLM response format for persona: {type(response)}. Response: {response}")
                self.console.print(f"[red]Using default persona due to unexpected LLM response.[/red]")
                return default_persona

        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            self.console.print(f"[red]Using default persona due to error: {e}[/red]")
            return default_persona

    def get_current_datetime(self):
        """Get the current date and time."""
        now = dt.datetime.now() # Use aliased datetime
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A")
        }

    async def initialize_new_world(self):
        """Initialize a new world state with both macro and micro levels using real data."""
        logger.info("Initializing new world...")
        config = {}
        try:
            with open(self.world_config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"World config file not found at {self.world_config_path}. Using default settings.")
            config = {}
        except Exception as e:
            logger.error(f"Error reading world config file {self.world_config_path}: {e}")
            config = {} # Use empty config on error

        try:
            location = config.get("location", {"city": "NYC", "country": "USA"})
            logger.info(f"Initializing world in {location.get('city', 'unknown location')}")

            # Get current date and time
            current_datetime = self.get_current_datetime()

            # Gather comprehensive news - real current events
            self.console.print(f"[yellow]Gathering news for {location.get('city')}...[/yellow]")
            news_results = await self._gather_comprehensive_news(location)

            # Initialize world state (macro level) with real data
            prompt = PromptManager.initialize_world_state_prompt(news_results, config)

            # Add current date and time to the prompt explicitly for context
            prompt += f"\n\n# Current Context\nCurrent date: {current_datetime['date']}\nCurrent time: {current_datetime['time']}\nDay of week: {current_datetime['day_of_week']}"

            self.console.print("[yellow]Generating world state based on current events...[/yellow]")
            world_state_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction='Create a comprehensive world state based on the provided information, current news, and real-world data. Respond with a JSON object matching the WorldStateResponse schema.',
                response_model=WorldStateResponse # Use the response model
            )

            if "updated_world_state" in world_state_response and isinstance(world_state_response["updated_world_state"], dict):
                # Validate the received structure before assignment
                try:
                    self.world_state = WorldState(**world_state_response["updated_world_state"]).model_dump()
                     # Ensure the time and date are current, potentially overriding LLM hallucination
                    self.world_state["current_time"] = current_datetime["time"]
                    self.world_state["current_date"] = current_datetime["date"]
                    logger.info("World state generated successfully.")
                except Exception as e:
                    logger.error(f"Validation failed for generated world state: {e}. Falling back to default. Response: {world_state_response}")
                    self.world_state = self._create_default_world_state(config)

            else:
                # Fallback if LLM fails or returns unexpected format
                logger.warning(f"LLM failed to generate valid world state. Response: {world_state_response}. Falling back to default.")
                self.world_state = self._create_default_world_state(config)

            # Determine a plausible location based on persona using the LLM
            starting_location = await self._determine_plausible_location(config)

            # Initialize immediate environment (micro level)
            self.console.print(f"[yellow]Creating environment for {starting_location}...[/yellow]")
            prompt = PromptManager.initialize_immediate_environment_prompt(self.world_state, starting_location)
            environment_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction='Create a detailed immediate environment based on the world state and location. Respond with a JSON object matching the ImmediateEnvironmentResponse schema.',
                response_model=ImmediateEnvironmentResponse # Use the response model
            )

            if "updated_environment" in environment_response and isinstance(environment_response["updated_environment"], dict):
                 # Validate the received structure
                try:
                    self.immediate_environment = ImmediateEnvironment(**environment_response["updated_environment"]).model_dump()
                    logger.info("Immediate environment generated successfully.")
                except Exception as e:
                    logger.error(f"Validation failed for generated environment: {e}. Falling back to default. Response: {environment_response}")
                    self.immediate_environment = await self._create_default_immediate_environment(starting_location)
            else:
                # Fallback if LLM fails - now uses LLM for fallback too
                logger.warning(f"LLM failed to generate valid environment. Response: {environment_response}. Falling back to default generation.")
                self.immediate_environment = await self._create_default_immediate_environment(starting_location)

            # Get persona information if available
            persona = {}
            persona_path = "simulacra_state.json"
            if os.path.exists(persona_path):
                try:
                    with open(persona_path, 'r') as file:
                        data = json.load(file)
                        if "persona" in data:
                            persona = data["persona"]
                except Exception as e:
                    logger.error(f"Error loading persona: {e}")
            # If persona still empty, might need to generate it here if not done elsewhere
            if not persona:
                 logger.warning("No persona loaded or generated before narrative context generation.")
                 # Optionally generate persona here if needed for context
                 # persona = await self._generate_persona_for_location(city, country)


            # Generate narrative context only if persona is available
            narrative_context = "Narrative context could not be generated (missing persona)."
            if persona:
                 narrative_context = await self._generate_narrative_context(
                      persona,
                      self.world_state,
                      self.immediate_environment.get('current_location_name', starting_location)
                 )
            else:
                 logger.warning("Skipping narrative context generation as persona is unavailable.")


            # Save the initial state
            self.save_state()

            logger.info("New world state and immediate environment initialized")
            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "narrative_context": narrative_context,
                "observations": [
                    f"You're in {self.immediate_environment.get('current_location_name', starting_location)}.",
                    f"It's {self.world_state.get('current_time', current_datetime['time'])} on {self.world_state.get('current_date', current_datetime['date'])}.",
                    f"The weather is {self.world_state.get('weather_condition', 'normal')}.",
                    "You notice your surroundings and gather your thoughts."
                ]
            }
        except Exception as e:
            logger.error(f"Critical error initializing new world: {e}", exc_info=True)
            # Create basic defaults if everything fails
            self.world_state = self._create_default_world_state(config)
            # Ensure fallback location name is simple string
            fallback_loc_name = "Coffee shop"
            self.immediate_environment = await self._create_default_immediate_environment(fallback_loc_name)
            self.save_state()
            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "narrative_context": "You arrived at a coffee shop after a busy morning. You have several things on your mind and are trying to figure out your next steps.",
                "observations": [
                    f"You're in a {fallback_loc_name}.",
                    f"It's {self.world_state.get('current_time', 'daytime')} on {self.world_state.get('current_date', 'today')}.",
                    f"The weather is {self.world_state.get('weather_condition', 'normal')}.",
                    "You notice your surroundings and gather your thoughts."
                ]
            }

    def _create_default_world_state(self, config: Dict) -> Dict:
        """Create a default world state if LLM initialization fails."""
        now = dt.datetime.now() # Use alias
        location = config.get("location", {"city": "NYC", "country": "USA"})

        # Ensure default structure matches WorldState model
        default_state = WorldState(
             current_time=now.strftime("%H:%M"),
             current_date=now.strftime("%Y-%m-%d"),
             city_name=location.get("city", "NYC"),
             country_name=location.get("country", "USA"),
             region_name=location.get("region", "Northeast"),
             weather_condition="Partly cloudy",
             temperature="68°F (20°C)",
             forecast="Similar conditions expected for the next 24 hours",
             social_climate="Generally calm with typical urban activity",
             economic_condition="Stable with normal business activity",
             major_events=["No major events currently"],
             local_news=["Standard local news coverage"],
             transportation_status="Normal operation of public transit and typical traffic patterns",
             utility_status="All utilities functioning normally",
             public_announcements=["No significant public announcements"],
             trending_topics=["Local sports", "Weather", "Weekend activities"],
             current_cultural_events=["Regular museum exhibitions", "Some local music performances"],
             sports_events=["Regular season games for local teams"],
             public_health_status="No significant health concerns or advisories",
             public_safety_status="Normal safety conditions with typical police presence"
        )
        return default_state.model_dump()


    async def _create_default_immediate_environment(self, location_name: str) -> Dict:
        """Create a default immediate environment using the LLM if the standard initialization fails."""
        # Ultimate fallback - hardcoded environment matching the model structure
        fallback_env = ImmediateEnvironment(
             current_location_name=location_name,
             location_type="Public space",
             indoor_outdoor="Indoor",
             noise_level="Moderate with conversation",
             lighting="Adequate lighting",
             temperature_feeling="Comfortable",
             air_quality="Fresh",
             present_people=["Various people"],
             crowd_density="Moderately busy",
             social_atmosphere="Neutral",
             ongoing_activities=["People going about their business"],
             nearby_objects=["Furniture", "Fixtures"],
             available_services=["Basic amenities"],
             exit_options=["Main entrance/exit"],
             interaction_opportunities=["People nearby"],
             visible_features=["Standard features for this type of location"],
             audible_sounds=["Ambient noise", "Conversations"],
             noticeable_smells=["Neutral smells"],
             seating_availability="Some seating available",
             food_drink_options=["Basic options if applicable"],
             restroom_access="Standard access",
             recent_changes=["Nothing notable has changed recently"],
             ongoing_conversations=["General conversations"],
             attention_drawing_elements=["Nothing particularly notable"]
        )
        logger.warning(f"Using hardcoded fallback environment for {location_name}.")
        return fallback_env.model_dump()


    async def process_update(self, simulacra_action: Dict[str, Any], simulacra_persona: Dict = None) -> Dict[str, Any]:
        """Process an action from the simulacra and update both world state and immediate environment."""

        # Save the action to history
        self.history.append({
            "timestamp": self.world_state.get("current_time", dt.datetime.now().strftime("%H:%M")),
            "date": self.world_state.get("current_date", dt.datetime.now().strftime("%Y-%m-%d")),
            "action": simulacra_action
        })

        self.console.print(f"\n[bold yellow]Processing action:[/bold yellow] {simulacra_action.get('action', 'Unknown action')}")

        # Include reaction profile in the prompt
        prompt = PromptManager.process_update_prompt(
            self.world_state,
            self.immediate_environment,
            simulacra_action,
            self.reaction_profile
        )

        system_instruction = 'Update the environment based on an action taken by a simulated character, following the specified world reaction profile. Respond ONLY with the JSON structure specified in the prompt.'

        try:
            self.console.print("[bold]Generating world response...[/bold]")
            # No specific response model, parse manually but expect the structure from the prompt
            update_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=None
            )

            # Basic validation of the response structure
            if not isinstance(update_response, dict) or not all(k in update_response for k in ["updated_environment", "world_state_changes", "consequences", "observations"]):
                 logger.error(f"LLM response for process_update is not a valid dictionary or missing keys: {update_response}")
                 raise ValueError("Invalid format received from LLM for world update.")


            # Update immediate environment
            if "updated_environment" in update_response and isinstance(update_response["updated_environment"], dict):
                 try:
                      # Validate against the model before updating
                      validated_env = ImmediateEnvironment(**update_response["updated_environment"])
                      self.immediate_environment = validated_env.model_dump()
                      self.console.print("[green]Environment updated successfully[/green]")
                 except Exception as e:
                      logger.error(f"Validation failed for updated environment: {e}. Keeping previous environment. Response: {update_response['updated_environment']}")
                      self.console.print("[red]Environment update failed validation, state unchanged.[/red]")
            else:
                 logger.warning("No valid 'updated_environment' found in LLM response.")


            # Update world state if there are broader changes
            if "world_state_changes" in update_response and isinstance(update_response["world_state_changes"], dict) and update_response["world_state_changes"]:
                # Apply only the changes to the world state
                world_changes = update_response["world_state_changes"]
                updated = False
                for key, value in world_changes.items():
                    if key in self.world_state:
                        # Basic type check (optional but recommended)
                        # if isinstance(value, type(self.world_state[key])):
                        self.world_state[key] = value
                        updated = True
                        # else:
                        # logger.warning(f"Type mismatch for world state key '{key}'. Expected {type(self.world_state[key])}, got {type(value)}. Skipping update.")
                    else:
                         logger.warning(f"LLM tried to update non-existent world state key '{key}'. Skipping.")
                if updated:
                    self.console.print("[blue]World state updated with broader changes[/blue]")
                else:
                    self.console.print("[yellow]World state changes proposed by LLM were invalid or empty.[/yellow]")
            elif "world_state_changes" in update_response and update_response["world_state_changes"]:
                # Log if it's present but not a dict or empty
                 logger.warning(f"Invalid 'world_state_changes' format in LLM response: {update_response['world_state_changes']}")


            # Ensure consequences and observations are properly formatted as lists of strings
            consequences_raw = update_response.get("consequences", [])
            observations_raw = update_response.get("observations", [])

            # Validate and sanitize consequences
            if isinstance(consequences_raw, list) and all(isinstance(item, str) for item in consequences_raw):
                consequences = consequences_raw
            elif isinstance(consequences_raw, str): # Handle single string case
                consequences = [consequences_raw] if consequences_raw else []
                logger.warning("LLM returned consequences as string, converted to list.")
            else:
                logger.warning(f"Invalid format for consequences: {consequences_raw}. Using empty list.")
                consequences = []

            # Validate and sanitize observations
            if isinstance(observations_raw, list) and all(isinstance(item, str) for item in observations_raw):
                observations = observations_raw
            elif isinstance(observations_raw, str): # Handle single string case
                 observations = [observations_raw] if observations_raw else []
                 logger.warning("LLM returned observations as string, converted to list.")
            else:
                logger.warning(f"Invalid format for observations: {observations_raw}. Using empty list.")
                observations = []

            # Generate an updated narrative context if persona is provided
            narrative_update = "Narrative update requires persona state."
            if simulacra_persona:
                try:
                     # Extract only the 'action' string from the history for simplicity
                    action_history = [
                        item['action']['action']
                        for item in self.history[-5:] # Last 5 items
                        if isinstance(item, dict) and 'action' in item and isinstance(item['action'], dict) and 'action' in item['action']
                     ]
                    narrative_update = await self._generate_updated_narrative(
                        simulacra_persona,
                        action_history, # Pass list of action strings
                        self.world_state,
                        self.immediate_environment,
                        consequences,
                        observations
                    )
                except Exception as narr_err:
                     logger.error(f"Error generating narrative update: {narr_err}")
                     narrative_update = "Error generating narrative update."


            # Save the updated state
            self.save_state()
            self.console.print("[green]World state saved successfully[/green]")

            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "consequences": consequences,
                "observations": observations,
                "narrative_update": narrative_update
            }

        except Exception as e:
            logger.error(f"Error processing update: {e}", exc_info=True)
            self.console.print(f"[bold red]Error processing update:[/bold red] {str(e)}")
            # Return current state without changes in case of error
            return {
                "error": str(e),
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "consequences": ["Error occurred during update processing."],
                "observations": ["The environment remains unchanged due to an error."],
                "narrative_update": "The story is paused due to an unexpected error."
            }

    def save_state(self):
        """Save current world state and immediate environment to a file."""
        try:
            # Ensure states are not None before saving
            if self.world_state is None or self.immediate_environment is None:
                 logger.error("Attempted to save state, but world_state or immediate_environment is None.")
                 return

            with open(self.state_path, 'w') as file:
                json.dump({
                    "world_state": self.world_state,
                    "immediate_environment": self.immediate_environment,
                    "history": self.history,
                    "reaction_profile": self.reaction_profile.model_dump()
                }, file, indent=2)
            logger.info(f"World state and immediate environment saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving world state: {e}")

    def load_state(self):
        """Load world state and immediate environment from a file."""
        try:
            with open(self.state_path, 'r') as file:
                state_data = json.load(file)

                # Validate loaded data against models
                world_state_loaded = False
                if "world_state" in state_data and isinstance(state_data["world_state"], dict):
                    try:
                         # Validate and assign directly
                        self.world_state = WorldState(**state_data["world_state"]).model_dump()
                        world_state_loaded = True
                    except Exception as e:
                         logger.error(f"Validation error loading world_state: {e}. State may be corrupted.")
                         self.world_state = None # Reset state if validation fails

                env_loaded = False
                if "immediate_environment" in state_data and isinstance(state_data["immediate_environment"], dict):
                    try:
                         # Validate and assign directly
                        self.immediate_environment = ImmediateEnvironment(**state_data["immediate_environment"]).model_dump()
                        env_loaded = True
                    except Exception as e:
                         logger.error(f"Validation error loading immediate_environment: {e}. State may be corrupted.")
                         self.immediate_environment = None # Reset state

                # Load history if present and is a list
                if "history" in state_data and isinstance(state_data["history"], list):
                    self.history = state_data["history"]
                else:
                     logger.warning("History not found or invalid in state file. Initializing empty history.")
                     self.history = [] # Initialize history if not found or invalid

                # Load reaction profile
                if "reaction_profile" in state_data and isinstance(state_data["reaction_profile"], dict):
                    try:
                        self.reaction_profile = WorldReactionProfile(**state_data["reaction_profile"])
                        logger.info(f"Loaded reaction profile: {self.reaction_profile.model_dump()}")
                    except Exception as e:
                        logger.error(f"Error loading reaction profile: {e}. Using default profile.")
                        self.reaction_profile = WorldReactionProfile.create_profile("balanced")
                else:
                    logger.info("No reaction profile found in state file. Using default profile.")
                    self.reaction_profile = WorldReactionProfile.create_profile("balanced")

                if world_state_loaded and env_loaded:
                    logger.info(f"World state and immediate environment loaded from {self.state_path}")
                else:
                    logger.warning(f"Failed to fully load state from {self.state_path}. World state or environment might be missing or invalid.")


        except FileNotFoundError:
            logger.warning(f"No world state file found at {self.state_path}. Starting with a new world.")
            self.world_state = None
            self.immediate_environment = None
            self.history = []
            self.reaction_profile = WorldReactionProfile.create_profile("balanced")
        except json.JSONDecodeError:
            logger.error(f"Corrupted world state file at {self.state_path}. Starting with a new world.")
            self.world_state = None
            self.immediate_environment = None
            self.history = []
            self.reaction_profile = WorldReactionProfile.create_profile("balanced")
        except Exception as e:
            logger.error(f"Error loading world state from {self.state_path}: {e}. Starting with a new world.")
            self.world_state = None
            self.immediate_environment = None
            self.history = []
            self.reaction_profile = WorldReactionProfile.create_profile("balanced")


    def get_world_summary(self) -> str:
        """Generate a concise summary of the current world state, highlighting key events."""
        if not self.world_state:
            return "[bold red]World state not initialized.[/bold red]"

        # Extract key information
        time_date = f"[bold cyan]{self.world_state.get('current_time', 'unknown time')} on {self.world_state.get('current_date', 'unknown date')}[/bold cyan]"
        location = f"[bold green]{self.world_state.get('city_name', 'unknown city')}, {self.world_state.get('region_name', 'unknown region')}[/bold green]"
        weather = f"[bold yellow]{self.world_state.get('weather_condition', 'unknown weather')}, {self.world_state.get('temperature', '')}[/bold yellow]"

        # Check for major events
        major_events = self.world_state.get('major_events', [])
        if not major_events or major_events == ["No major events currently"] or not isinstance(major_events, list):
            event_str = "[italic]No significant events.[/italic]"
        else:
            event_str = f"[bold magenta]Events:[/bold magenta] {', '.join(map(str, major_events[:2]))}" # Ensure items are strings

        # Check for any unusual conditions
        conditions = []
        if "unstable" in str(self.world_state.get('economic_condition', '')).lower() or "recession" in str(self.world_state.get('economic_condition', '')).lower():
            conditions.append("[bold red]Economic instability[/bold red]")
        if "tension" in str(self.world_state.get('social_climate', '')).lower() or "unrest" in str(self.world_state.get('social_climate', '')).lower():
            conditions.append("[bold red]Social tensions[/bold red]")
        if "storm" in str(self.world_state.get('weather_condition', '')).lower() or "severe" in str(self.world_state.get('weather_condition', '')).lower():
            conditions.append("[bold red]Severe weather[/bold red]")
        if "outage" in str(self.world_state.get('utility_status', '')).lower() or "disruption" in str(self.world_state.get('utility_status', '')).lower():
            conditions.append("[bold red]Utility disruptions[/bold red]")
        if "delay" in str(self.world_state.get('transportation_status', '')).lower() or "closure" in str(self.world_state.get('transportation_status', '')).lower():
            conditions.append("[bold red]Transportation issues[/bold red]")
        if "warning" in str(self.world_state.get('public_health_status', '')).lower() or "outbreak" in str(self.world_state.get('public_health_status', '')).lower():
            conditions.append("[bold red]Health concerns[/bold red]")

        if conditions:
            condition_str = f"[bold orange3]Notable conditions:[/bold orange3] {', '.join(conditions)}"
        else:
            condition_str = "[green]Normal conditions overall.[/green]"

        # Combine into summary
        summary = f"[bold blue]World Summary:[/bold blue] {time_date} in {location}. {weather}. {event_str} {condition_str}"
        return summary

    def get_environment_summary(self) -> str:
        """Generate a concise summary of the immediate environment, highlighting key elements."""
        if not self.immediate_environment:
            return "[bold red]Immediate environment not initialized.[/bold red]"

        # Extract key information
        location = f"[bold cyan]{self.immediate_environment.get('current_location_name', 'unknown location')} ({self.immediate_environment.get('location_type', 'unknown type')})[/bold cyan]"
        setting = f"[bold green]{self.immediate_environment.get('indoor_outdoor', 'unknown setting')}[/bold green]"
        atmosphere = f"[bold yellow]{self.immediate_environment.get('social_atmosphere', 'unknown atmosphere')}[/bold yellow]"

        # People and activities
        people = self.immediate_environment.get('present_people', [])
        if isinstance(people, list) and people:
             people_str = f"[bold magenta]People:[/bold magenta] {', '.join(map(str, people[:2]))}" # Ensure strings
        else:
            people_str = "[italic]No people around.[/italic]"

        activities = self.immediate_environment.get('ongoing_activities', [])
        if isinstance(activities, list) and activities:
            activities_str = f"[bold blue]Activities:[/bold blue] {', '.join(map(str, activities[:2]))}" # Ensure strings
        else:
            activities_str = "[italic]No notable activities.[/italic]"

        # Attention-grabbing elements
        attention = self.immediate_environment.get('attention_drawing_elements', [])
        if isinstance(attention, list) and attention:
            attention_str = f"[bold orange3]Notable:[/bold orange3] {', '.join(map(str, attention[:2]))}" # Ensure strings
        else:
            attention_str = "[italic]Nothing particularly notable.[/italic]"

        # Recent changes
        changes = self.immediate_environment.get('recent_changes', [])
        if isinstance(changes, list) and changes and changes != ["Nothing notable has changed recently"]:
            changes_str = f"[bold red]Recent changes:[/bold red] {', '.join(map(str, changes[:2]))}" # Ensure strings
        else:
            changes_str = ""

        # Combine into summary
        summary = f"[bold purple]Environment Summary:[/bold purple] {location}, {setting}. {atmosphere} atmosphere. {people_str} {activities_str} {attention_str} {changes_str}".strip()
        return summary

    async def _generate_narrative_context(self, persona: Dict, world_state: Dict, location: str) -> str:
        """Generate a narrative context explaining how the character arrived at their current situation."""

        # Ensure persona is not empty before proceeding
        if not persona:
            logger.warning("Cannot generate narrative context: persona is empty.")
            return "Narrative context requires a valid persona."

        # Create a prompt for the LLM to generate a narrative context
        prompt = f"""
        Based on the following information, create a brief narrative context (2-3 paragraphs) explaining how the character arrived at their current situation:

        Character information:
        Name: {persona.get('name', 'Unknown')}
        Age: {persona.get('age', 'Unknown')}
        Occupation: {persona.get('occupation', 'Unknown')}
        Personality traits: {', '.join(persona.get('personality_traits', ['Unknown']))}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        Current physical state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        Current emotional state: {persona.get('current_state', {}).get('emotional', 'Unknown')}
        Current mental state: {persona.get('current_state', {}).get('mental', 'Unknown')}
        Short-term memories: {', '.join(persona.get('memory', {}).get('short_term', ['Unknown']))}
        Long-term memories: {', '.join(persona.get('memory', {}).get('long_term', ['Unknown']))}

        World information:
        Current time: {world_state.get('current_time', 'Unknown')}
        Current date: {world_state.get('current_date', 'Unknown')}
        City: {world_state.get('city_name', 'Unknown')}
        Weather: {world_state.get('weather_condition', 'Unknown')}
        Social climate: {world_state.get('social_climate', 'Unknown')}
        Major events: {', '.join(world_state.get('major_events', ['None']))}

        Current location: {location}

        The narrative should explain:
        1. Why the character is at this specific location
        2. What they were doing earlier today
        3. What their immediate concerns or thoughts are
        4. How their current emotional and physical state came to be

        The narrative should be realistic, specific to this character, and connect to their goals and personality.
        Respond only with the narrative text, no preamble or JSON structure.
        """

        self.console.print(f"[yellow]Generating narrative context...[/yellow]")

        try:
            # Use the LLM to generate a narrative context
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Create a realistic narrative context for the character's current situation. Respond ONLY with the narrative text.",
                temperature=0.7,  # Slightly higher temperature for creativity
                response_model=None # Expect raw text
            )

            # Extract the narrative from the response
            if isinstance(response, dict) and "text" in response:
                narrative = response["text"].strip()
            elif isinstance(response, dict) and "error" in response:
                 logger.error(f"LLM error generating narrative: {response['error']}")
                 narrative = f"Error generating narrative context: {response['error']}"
            elif isinstance(response, str):
                narrative = response.strip()
            else:
                narrative = str(response).strip()

            self.console.print(f"[green]Narrative context generated successfully[/green]")
            return narrative if narrative else "Narrative generation failed."
        except Exception as e:
            logger.error(f"Error generating narrative context: {e}", exc_info=True)
            # Fallback to a generic narrative
            fallback_narrative = f"{persona.get('name', 'The character')} arrived at {location} after a busy morning. They have several things on their mind, particularly {', '.join(persona.get('goals', ['their goals']))}. They're feeling {persona.get('current_state', {}).get('emotional', 'mixed emotions')} as they navigate their day."
            self.console.print(f"[red]Error generating narrative context, using fallback[/red]")
            return fallback_narrative

    async def _generate_updated_narrative(self, persona: Dict, previous_actions: List[str], # Expect list of strings
                                        world_state: Dict, immediate_environment: Dict,
                                        consequences: List[str], observations: List[str]) -> str:
        """Generate an updated narrative context based on recent actions and events."""

         # Ensure persona is not empty
        if not persona:
            logger.warning("Cannot generate updated narrative: persona is empty.")
            return "Narrative update requires a valid persona."

        actions_text = "\n".join([f"- {action}" for action in previous_actions]) if previous_actions else "No recent actions recorded."

        # Create a prompt for the LLM to generate an updated narrative
        prompt = f"""
        Based on the following information, create a brief narrative update (1-2 paragraphs) that continues the character's story:

        Character information:
        Name: {persona.get('name', 'Unknown')}
        Current physical state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        Current emotional state: {persona.get('current_state', {}).get('emotional', 'Unknown')}
        Current mental state: {persona.get('current_state', {}).get('mental', 'Unknown')}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}

        Current location: {immediate_environment.get('current_location_name', 'Unknown location')}
        Current time: {world_state.get('current_time', 'Unknown time')}

        Recent actions taken by the character:
        {actions_text}

        Recent consequences of these actions:
        {', '.join(consequences) if consequences else 'No notable consequences.'}

        Recent observations by the character:
        {', '.join(observations) if observations else 'Nothing notable observed.'}

        The narrative update should:
        1. Reflect the character's progress toward their goals
        2. Acknowledge any changes in their emotional or physical state
        3. Incorporate the consequences of their recent actions
        4. Set up potential next steps or challenges
        5. Maintain continuity with their overall story

        The narrative should be realistic, specific to this character, and feel like a continuation of their ongoing story.
        Respond only with the narrative update text, no preamble or JSON structure.
        """

        self.console.print(f"[yellow]Generating narrative update...[/yellow]")

        try:
            # Use the LLM to generate a narrative update
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Create a realistic narrative update that continues the character's story. Respond ONLY with the narrative text.",
                temperature=0.7,  # Slightly higher temperature for creativity
                response_model=None # Expect raw text
            )

            # Extract the narrative from the response
            if isinstance(response, dict) and "text" in response:
                narrative = response["text"].strip()
            elif isinstance(response, dict) and "error" in response:
                 logger.error(f"LLM error generating narrative update: {response['error']}")
                 narrative = f"Error generating narrative update: {response['error']}"
            elif isinstance(response, str):
                narrative = response.strip()
            else:
                narrative = str(response).strip()

            self.console.print(f"[green]Narrative update generated successfully[/green]")
            return narrative if narrative else "Narrative update generation failed."
        except Exception as e:
            logger.error(f"Error generating narrative update: {e}", exc_info=True)
            # Fallback to a generic narrative update
            fallback_narrative = f"{persona.get('name', 'The character')} continues their day at {immediate_environment.get('current_location_name', 'their location')}. They're still focused on {', '.join(persona.get('goals', ['their goals']))} as they navigate the next steps."
            self.console.print(f"[red]Error generating narrative update, using fallback[/red]")
            return fallback_narrative
