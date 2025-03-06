import json
import logging
import re
import yaml
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from rich.console import Console
from rich.prompt import IntPrompt
from langchain_community.tools import DuckDuckGoSearchRun
import argparse
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("simulation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with the LLM."""
    
    async def generate_content(self, prompt: str, system_instruction: str = "", temperature: float = 0.7) -> Any:
        """Generate content using the LLM."""
        try:
            # This is a placeholder for the actual LLM call
            # In a real implementation, this would call an API or local model
            
            # For demonstration purposes, we'll return a mock response
            # In a real implementation, replace this with actual LLM API calls
            
            if "weather" in prompt.lower():
                return json.dumps({
                    "weather_condition": "Partly Cloudy",
                    "temperature": "72°F (22°C)",
                    "precipitation": "None",
                    "data_source": "Mock Weather Data"
                })
            elif "news" in prompt.lower():
                return json.dumps([
                    {
                        "headline": "Local Festival Announced for Next Weekend",
                        "summary": "The annual city festival will take place next weekend with music, food, and activities.",
                        "data_source": "Mock News Data"
                    },
                    {
                        "headline": "New Restaurant Opens Downtown",
                        "summary": "A new Italian restaurant has opened in the downtown area, featuring authentic cuisine.",
                        "data_source": "Mock News Data"
                    }
                ])
            elif "local events" in prompt.lower():
                return json.dumps([
                    {
                        "name": "Farmers Market",
                        "description": "Weekly farmers market with local produce and crafts",
                        "date": "Every Saturday",
                        "location": "City Square",
                        "data_source": "Mock Events Data"
                    },
                    {
                        "name": "Art Exhibition",
                        "description": "Local artists showcase their work",
                        "date": "This weekend",
                        "location": "Community Gallery",
                        "data_source": "Mock Events Data"
                    }
                ])
            elif "create a persona" in prompt.lower():
                return json.dumps({
                    "name": "Alex Johnson",
                    "age": 32,
                    "occupation": "Graphic Designer",
                    "personality_traits": ["creative", "curious", "outgoing"],
                    "background": "Born and raised in the city, studied art in college, now works at a design agency.",
                    "goals": ["Advance in career", "Find new inspiration", "Expand social circle"],
                    "current_emotion": "curious",
                    "current_goal": "Explore new parts of the city"
                })
            elif "immediate environment" in prompt.lower():
                return json.dumps({
                    "current_location_name": "Central Park",
                    "location_type": "Park",
                    "indoor_outdoor": "Outdoor",
                    "description": "A vast urban park with walking paths, lakes, and open spaces.",
                    "features": ["Trees", "Benches", "Walking paths", "Lake"],
                    "present_people": ["Joggers", "Dog walkers", "Tourists", "Families"],
                    "crowd_density": "Moderate",
                    "social_atmosphere": "Relaxed and recreational",
                    "ongoing_activities": ["Jogging", "Picnicking", "Photography", "Dog walking"],
                    "data_source": "Mock Environment Data"
                })
            elif "search" in prompt.lower() and "results" in prompt.lower():
                return json.dumps({
                    "consequences": ["You learned about the opening hours of the museum", "You discovered there's a special exhibition today"],
                    "observations": ["The search results were detailed and helpful", "Several reviews mentioned the impressive architecture"]
                })
            elif "decide" in prompt.lower() and "action" in prompt.lower():
                return "I decide to walk over to the lake and sit on a bench to enjoy the view for a few minutes."
            elif "updated narrative" in prompt.lower():
                return "I stroll through the park, enjoying the dappled sunlight filtering through the trees. The weather is pleasant, and I can hear birds chirping nearby. I decide to find a quiet spot to sit and observe the people around me."
            elif "location details" in prompt.lower():
                return json.dumps({
                    "full_name": "Central Park",
                    "location_type": "Park",
                    "description": "A vast urban park with walking paths, lakes, and open spaces.",
                    "features": ["Trees", "Benches", "Walking paths", "Lake", "Zoo", "Playgrounds"],
                    "opening_hours": "6:00 AM - 1:00 AM daily",
                    "address": "Central Park, New York, NY",
                    "popular_for": "Walking, jogging, picnics, boating",
                    "price_level": "Free",
                    "busy_times": "Weekends and holidays",
                    "rating": "4.8/5",
                    "data_source": "Mock Location Data"
                })
            elif "nearby locations" in prompt.lower():
                return json.dumps([
                    {
                        "name": "Metropolitan Museum of Art",
                        "type": "Museum",
                        "distance": "Adjacent to Central Park"
                    },
                    {
                        "name": "Bethesda Terrace",
                        "type": "Landmark",
                        "distance": "Within Central Park"
                    },
                    {
                        "name": "Loeb Boathouse",
                        "type": "Restaurant",
                        "distance": "Within Central Park"
                    }
                ])
            elif "opening hours" in prompt.lower():
                return json.dumps({
                    "is_open": True,
                    "hours": "6:00 AM - 1:00 AM daily",
                    "data_source": "Mock Hours Data"
                })
            elif "travel time" in prompt.lower():
                return "15"
            elif "transition" in prompt.lower():
                return "I walk from Central Park to the Metropolitan Museum of Art, enjoying the pleasant weather. As I approach the grand entrance of the museum, I notice the impressive architecture and the steady stream of visitors entering and exiting. The change from the natural setting of the park to this cultural institution is striking."
            else:
                # Generic response for other prompts
                return json.dumps({
                    "consequences": ["Your action was successful", "People around you noticed what you did"],
                    "observations": ["The environment responded naturally to your action", "You feel satisfied with the outcome"]
                })
                
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return "Error generating content. Please try again."

class Simulacra:
    """A simulated character with a persona."""
    
    def __init__(self):
        """Initialize the simulacra."""
        self.persona = {}
        self.llm_service = LLMService()
    
    async def decide_action(self, perception) -> str:
        """Decide on an action based on the current perception."""
        # Generate prompt for deciding action
        prompt = PromptManager.decide_action_prompt(perception)
        
        # Get response from LLM
        action_response = await self.llm_service.generate_content(
            prompt=prompt,
            system_instruction="You are a decision-making system for a simulated character. Respond with a specific, realistic action in first person present tense.",
            temperature=0.7
        )
        
        # Process the response
        if isinstance(action_response, list) and len(action_response) > 0:
            action = action_response[0]
        elif isinstance(action_response, dict) and "text" in action_response:
            action = action_response["text"]
        elif isinstance(action_response, str):
            action = action_response
        else:
            logger.warning(f"Unexpected action response format: {type(action_response)}")
            action = "I decide to wait and observe my surroundings for a moment."
        
        # Clean up the action text
        action = action.strip().strip('"')
        
        # Update the simulacra's state based on the action
        self._update_state_based_on_action(action)
        
        return action
    
    def _update_state_based_on_action(self, action: str) -> None:
        """Update the simulacra's state based on the action they've decided to take."""
        # Extract emotion from action if possible
        emotion_keywords = {
            "happy": ["smile", "laugh", "joy", "happy", "excited", "pleased", "delighted"],
            "sad": ["sigh", "cry", "sad", "upset", "disappointed", "unhappy", "tearful"],
            "angry": ["frown", "angry", "annoyed", "irritated", "frustrated", "mad"],
            "anxious": ["nervous", "anxious", "worried", "concerned", "uneasy", "apprehensive"],
            "curious": ["curious", "interested", "intrigued", "wonder", "explore", "investigate"],
            "tired": ["tired", "exhausted", "yawn", "weary", "fatigued", "sleepy"],
            "relaxed": ["relax", "calm", "peaceful", "comfortable", "content", "at ease"]
        }
        
        action_lower = action.lower()
        detected_emotion = None
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in action_lower for keyword in keywords):
                detected_emotion = emotion
                break
        
        if detected_emotion:
            self.persona["current_emotion"] = detected_emotion
        
        # Update recent memory with the action
        if "recent_actions" not in self.persona:
            self.persona["recent_actions"] = []
        
        # Add the action to recent actions, keeping only the last 5
        self.persona["recent_actions"].append(action)
        if len(self.persona["recent_actions"]) > 5:
            self.persona["recent_actions"] = self.persona["recent_actions"][-5:]
        
        # Update recent memory
        self.persona["recent_memory"] = f"I recently {action.lower().replace('I ', '')}"

class PromptManager:
    """Manages prompts for the simulation."""
    
    @staticmethod
    def process_update_prompt(world_state: Dict, immediate_environment: Dict, simulacra: Dict, action: str, reaction_profile: str = "balanced") -> str:
        """Generate a prompt for processing an update to the world state based on an action."""
        # Extract relevant information from world_state
        city = world_state.get("city", "Unknown")
        country = world_state.get("country", "Unknown")
        current_time = world_state.get("current_time", "Unknown")
        current_date = world_state.get("current_date", "Unknown")
        day_of_week = world_state.get("day_of_week", "Unknown")
        
        # Extract weather information
        weather = world_state.get("weather", {})
        weather_condition = weather.get("weather_condition", "Unknown")
        temperature = weather.get("temperature", "Unknown")
        
        # Extract location information
        current_location = immediate_environment.get("current_location_name", "Unknown")
        location_type = immediate_environment.get("location_type", "Unknown")
        indoor_outdoor = immediate_environment.get("indoor_outdoor", "Unknown")
        location_description = immediate_environment.get("description", "No description available")
        
        # Extract social context
        present_people = immediate_environment.get("present_people", ["Unknown"])
        crowd_density = immediate_environment.get("crowd_density", "Unknown")
        social_atmosphere = immediate_environment.get("social_atmosphere", "Unknown")
        ongoing_activities = immediate_environment.get("ongoing_activities", ["Unknown"])
        
        # Extract persona information
        persona_name = simulacra.get("name", "Unknown")
        persona_age = simulacra.get("age", "Unknown")
        persona_occupation = simulacra.get("occupation", "Unknown")
        persona_traits = simulacra.get("personality_traits", ["Unknown"])
        persona_goal = simulacra.get("current_goal", "Unknown")
        persona_emotion = simulacra.get("current_emotion", "neutral")
        
        # Get news and events if available
        news = world_state.get("news", [])
        local_events = world_state.get("local_events", [])
        
        # Format news and events for the prompt
        news_text = ""
        if news:
            news_text = "Recent News:\n"
            for i, item in enumerate(news[:2], 1):  # Include up to 2 news items
                news_text += f"- {item.get('headline', 'Unknown')}: {item.get('summary', 'No details')}\n"
        
        events_text = ""
        if local_events:
            events_text = "Local Events:\n"
            for i, event in enumerate(local_events[:2], 1):  # Include up to 2 events
                events_text += f"- {event.get('name', 'Unknown')}: {event.get('description', 'No details')} at {event.get('location', 'Unknown')}\n"
        
        # Determine reaction guidance based on profile
        reaction_guidance = ""
        if reaction_profile == "balanced":
            reaction_guidance = "Provide balanced, realistic consequences that match the scale of the action."
        elif reaction_profile == "realistic":
            reaction_guidance = "Provide highly realistic, detailed consequences that would occur in the real world."
        elif reaction_profile == "dramatic":
            reaction_guidance = "Provide somewhat dramatic consequences that make the simulation more interesting."
        elif reaction_profile == "optimistic":
            reaction_guidance = "Provide generally positive consequences, focusing on what goes well."
        elif reaction_profile == "pessimistic":
            reaction_guidance = "Provide generally challenging consequences, focusing on complications that arise."
        
        # Construct the prompt
        prompt = f"""
        You are simulating realistic world reactions to a character's action in {city}, {country}.
        
        REAL-WORLD CONTEXT:
        - Location: {city}, {country}
        - Date: {current_date} ({day_of_week})
        - Time: {current_time}
        - Weather: {weather_condition}, {temperature}
        {news_text}
        {events_text}
        
        CURRENT LOCATION: {current_location} ({location_type}, {indoor_outdoor})
        {location_description}
        
        Notable features:
        - {', '.join(immediate_environment.get('features', ['None']))}
        
        Present people: {', '.join(present_people)}
        Crowd density: {crowd_density}
        Social atmosphere: {social_atmosphere}
        Ongoing activities: {', '.join(ongoing_activities)}
        
        CHARACTER:
        - Name: {persona_name}
        - Age: {persona_age}
        - Occupation: {persona_occupation}
        - Personality: {', '.join(persona_traits)}
        - Current goal: {persona_goal}
        - Current emotion: {persona_emotion}
        
        ACTION:
        {action}
        
        Based on this action and context, determine:
        1. The realistic consequences of this action (what happens as a result)
        2. What the character observes during and after taking this action
        
        GUIDANCE:
        {reaction_guidance}
        - Consider the physical environment, social context, and time of day
        - Consider how other people present would realistically react
        - Consider any limitations or opportunities presented by the location
        - Keep consequences proportional to the action
        
        Format your response as a JSON object with the following structure:
        {{
            "consequences": ["List of 1-3 consequences of the action"],
            "observations": ["List of 1-3 observations the character makes"],
            "environment_updates": {{
                "optional fields to update in the environment": "new values"
            }}
        }}
        """
        
        return prompt
    
    @staticmethod
    def decide_action_prompt(perception) -> str:
        """Generate a prompt for deciding an action based on perception."""
        # Extract relevant information from perception
        time_context = perception.get("time_context", {})
        location_context = perception.get("location_context", {})
        environmental_context = perception.get("environmental_context", {})
        social_context = perception.get("social_context", {})
        self_context = perception.get("self_context", {})
        narrative_context = perception.get("narrative_context", "")
        
        # Format the prompt
        prompt = f"""
        You are deciding the next action for a character in a realistic simulation based on their perception of the world.
        
        Current Situation:
        - Time: {time_context.get('current_time', 'Unknown')} on {time_context.get('current_date', 'Unknown')} ({time_context.get('day_of_week', 'Unknown')})
        - Location: {location_context.get('current_location', 'Unknown')} in {location_context.get('city', 'Unknown')}, {location_context.get('country', 'Unknown')}
        - Location Type: {location_context.get('location_type', 'Unknown')}
        """
        
        # Add weather if available
        if "weather" in environmental_context:
            weather = environmental_context.get("weather", {})
            prompt += f"""
        - Weather: {weather.get('condition', 'Unknown')}, {weather.get('temperature', 'Unknown')}
        """
        
        # Add location description if available
        if "description" in location_context:
            prompt += f"""
        - Description: {location_context.get('description', 'No description available')}
        """
        
        # Add social context if available
        if social_context:
            prompt += f"""
        - People Present: {', '.join(social_context.get('present_people', ['Unknown']))}
        - Crowd Density: {social_context.get('crowd_density', 'Unknown')}
        - Social Atmosphere: {social_context.get('social_atmosphere', 'Unknown')}
        - Ongoing Activities: {', '.join(social_context.get('ongoing_activities', ['Unknown']))}
        """
        
        # Add nearby locations if available
        if "nearby_locations" in location_context:
            nearby = location_context.get("nearby_locations", [])
            if nearby:
                prompt += "\n    - Nearby Places:\n"
                for i, place in enumerate(nearby[:3], 1):
                    prompt += f"      {i}. {place.get('name', 'Unknown')} ({place.get('type', 'place')})\n"
        
        # Add character's current state
        prompt += f"""
        Character's Current State:
        - Name: {self_context.get('name', 'Unknown')}
        - Age: {self_context.get('age', 'Unknown')}
        - Occupation: {self_context.get('occupation', 'Unknown')}
        - Current Emotion: {self_context.get('current_emotion', 'Unknown')}
        - Current Goal: {self_context.get('current_goal', 'Unknown')}
        - Recent Memory: {self_context.get('recent_memory', 'None')}
        """
        
        # Add narrative context if available
        if narrative_context:
            prompt += f"""
        Narrative Context:
        {narrative_context}
        """
        
        # Add action guidance
        prompt += """
        Based on the character's current situation and state, decide on a realistic and specific action for them to take next.
        
        The action should:
        1. Be realistic and appropriate for the current situation
        2. Consider the character's goals, emotions, and personality
        3. Be specific and detailed (e.g., "Order a cappuccino and sit by the window" rather than just "Get coffee")
        4. Consider the time of day, location, and social context
        
        Available action types:
        - Physical actions (e.g., "Walk to the counter", "Sit down at a table")
        - Social interactions (e.g., "Ask the barista about their coffee recommendations")
        - Movement to new locations (e.g., "Leave the cafe and walk to the nearby park")
        - Information seeking (e.g., "Search online for local events happening today")
        - Internal actions (e.g., "Reflect on the recent conversation")
        
        Return ONLY the action as a single paragraph, written in first person present tense (e.g., "I walk to the counter and order a cappuccino").
        """
        
        return prompt

class WorldEngine:
    """Engine for simulating the world and its interactions with the simulacra."""
    
    def __init__(self, config: Dict, reaction_profile: str = "balanced"):
        """Initialize the world engine."""
        self.config = config
        self.reaction_profile = reaction_profile
        self.world_state = {}
        self.immediate_environment = {}
        self.simulacra = Simulacra()
        self.narrative_context = ""
        self.llm_service = LLMService()
        self.console = Console()
    
    async def initialize_new_world(self, persona_file: str) -> None:
        """Initialize a new world state with a persona."""
        try:
            # Get city and country from config
            city = self.config.get("city", "New York")
            country = self.config.get("country", "United States")
            
            # Load or create persona
            try:
                with open(persona_file, "r") as f:
                    self.simulacra.persona = json.load(f)
                    self.console.print(f"[green]Loaded persona: {self.simulacra.persona.get('name', 'Unknown')}[/green]")
            except FileNotFoundError:
                # Create a new persona
                self.simulacra.persona = await create_persona(city, country)
            
            # Determine a plausible location name
            location_name = self.config.get("starting_location", "")
            if not location_name:
                location_name = await self._determine_plausible_location(city, country, self.simulacra.persona)
            
            # Initialize world state with real-time data
            self.world_state = {
                "city": city,
                "country": country,
                "current_location": location_name
            }
            
            # Update with real-time data
            await self._update_real_time_data()
            
            # Create immediate environment
            self.immediate_environment = await self._create_default_immediate_environment(location_name)
            
            # Generate initial narrative context
            self.narrative_context = await self._generate_initial_narrative()
            
            # Save the initial state
            await self.save_state()
            
            self.console.print(f"[green]Successfully initialized new world in {city}, {country} at {location_name}[/green]")
            
        except Exception as e:
            logger.error(f"Error initializing new world: {e}")
            self.console.print(f"[red]Error initializing new world: {e}[/red]")
            
            # Create a minimal fallback world state
            self.world_state = {
                "city": self.config.get("city", "New York"),
                "country": self.config.get("country", "United States"),
                "current_time": datetime.now().strftime("%H:%M"),
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "day_of_week": datetime.now().strftime("%A"),
                "weather": {
                    "weather_condition": "Clear",
                    "temperature": "72°F (22°C)",
                    "data_source": "Fallback - No data retrieved"
                }
            }
            
            # Create a minimal immediate environment
            location_name = self.config.get("starting_location", "Coffee Shop")
            self.immediate_environment = {
                "current_location_name": location_name,
                "location_type": "Coffee Shop",
                "indoor_outdoor": "Indoor",
                "description": "A cozy coffee shop with a warm atmosphere.",
                "features": ["Tables", "Chairs", "Counter", "Coffee machines"],
                "present_people": ["Baristas", "Customers"],
                "crowd_density": "Moderate",
                "social_atmosphere": "Casual and relaxed",
                "ongoing_activities": ["People drinking coffee", "Conversations", "People working on laptops"],
                "data_source": "Fallback - No data retrieved"
            }
            
            # Generate a simple narrative context
            self.narrative_context = f"I find myself at a {location_name} in {self.world_state['city']}, {self.world_state['country']}. It's {self.world_state['current_time']} on {self.world_state['day_of_week']}."
    
    async def _determine_plausible_location(self, city: str, country: str, persona: Dict) -> str:
        """Determine a plausible location for the simulacra based on their persona."""
        try:
            # Extract relevant information from persona
            occupation = persona.get("occupation", "")
            current_goal = persona.get("current_goal", "")
            
            # Get current time
            current_time = datetime.now().strftime("%H:%M")
            current_hour = int(current_time.split(":")[0])
            
            # Determine day of week
            day_of_week = datetime.now().strftime("%A")
            is_weekend = day_of_week in ["Saturday", "Sunday"]
            
            # Create a prompt for determining a plausible location
            prompt = f"""
            Determine a plausible current location for a person with the following characteristics:
            
            - Occupation: {occupation}
            - Current goal: {current_goal}
            - Current time: {current_time}
            - Day of week: {day_of_week}
            - City: {city}, {country}
            
            Consider:
            1. What would be a realistic place for this person to be at this time and day?
            2. Consider their occupation and goals
            3. Consider the time of day (morning, afternoon, evening)
            4. Consider whether it's a weekday or weekend
            
            Return ONLY the name of a specific, realistic location (e.g., "Central Park", "Starbucks on 5th Avenue", "Public Library").
            """
            
            # Get response from LLM
            location_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Determine a realistic location for a person based on their characteristics and the current time and day.",
                temperature=0.7
            )
            
            # Process the response
            if isinstance(location_response, list) and len(location_response) > 0:
                location = location_response[0]
            elif isinstance(location_response, dict) and "text" in location_response:
                location = location_response["text"]
            elif isinstance(location_response, str):
                location = location_response
            else:
                raise ValueError(f"Unexpected response format: {type(location_response)}")
            
            # Clean up the location name
            location = location.strip().strip('"\'.,;:')
            
            return location
            
        except Exception as e:
            logger.error(f"Error determining plausible location: {e}")
            
            # Fallback to time-based location
            current_hour = datetime.now().hour
            day_of_week = datetime.now().strftime("%A")
            is_weekend = day_of_week in ["Saturday", "Sunday"]
            
            if is_weekend:
                if 8 <= current_hour < 12:
                    return "Local Cafe"
                elif 12 <= current_hour < 18:
                    return "City Park"
                else:
                    return "Home"
            else:  # Weekday
                if 8 <= current_hour < 9:
                    return "Coffee Shop"
                elif 9 <= current_hour < 17:
                    return "Office"
                elif 17 <= current_hour < 20:
                    return "Restaurant"
                else:
                    return "Home"
    
    async def _update_real_time_data(self) -> None:
        """Update the world state with real-time data."""
        try:
            # Get city and country
            city = self.world_state.get("city", "New York")
            country = self.world_state.get("country", "United States")
            
            # Update current date and time
            current_datetime = datetime.now()
            self.world_state["current_time"] = current_datetime.strftime("%H:%M")
            self.world_state["current_date"] = current_datetime.strftime("%Y-%m-%d")
            self.world_state["day_of_week"] = current_datetime.strftime("%A")
            
            # Check if we need to update weather data
            update_weather = False
            if "weather" not in self.world_state:
                update_weather = True
            elif "last_weather_update" in self.world_state:
                last_update = datetime.strptime(self.world_state["last_weather_update"], "%Y-%m-%d")
                if (current_datetime.date() - last_update.date()).days >= 1:
                    update_weather = True
            else:
                update_weather = True
            
            if update_weather and self.config.get("use_real_data", True):
                weather_data = await self._get_real_weather_data(city, country)
                self.world_state["weather"] = weather_data
                self.world_state["last_weather_update"] = current_datetime.strftime("%Y-%m-%d")
                self.console.print(f"[green]Updated weather data for {city}, {country}[/green]")
            
            # Check if we need to update news data
            update_news = False
            if "news" not in self.world_state:
                update_news = True
            elif "last_news_update" in self.world_state:
                last_update = datetime.strptime(self.world_state["last_news_update"], "%Y-%m-%d")
                if (current_datetime.date() - last_update.date()).days >= 3:  # Update news every 3 days
                    update_news = True
            else:
                update_news = True
            
            if update_news and self.config.get("use_real_data", True):
                news_data = await self._get_real_news_data(city, country)
                self.world_state["news"] = news_data
                self.world_state["last_news_update"] = current_datetime.strftime("%Y-%m-%d")
                self.console.print(f"[green]Updated news data for {city}, {country}[/green]")
            
            # Check if we need to update local events
            update_events = False
            if "local_events" not in self.world_state:
                update_events = True
            elif "last_events_update" in self.world_state:
                last_update = datetime.strptime(self.world_state["last_events_update"], "%Y-%m-%d")
                # Update events weekly or if it's Monday
                if (current_datetime.date() - last_update.date()).days >= 7 or current_datetime.strftime("%A") == "Monday":
                    update_events = True
            else:
                update_events = True
            
            if update_events and self.config.get("use_real_data", True):
                events_data = await self._get_local_events(city, country)
                self.world_state["local_events"] = events_data
                self.world_state["last_events_update"] = current_datetime.strftime("%Y-%m-%d")
                self.console.print(f"[green]Updated local events for {city}, {country}[/green]")
            
        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            self.console.print(f"[red]Error updating real-time data: {e}[/red]")
    
    async def _get_real_weather_data(self, city: str, country: str) -> Dict:
        """Get real weather data for the specified city and country."""
        try:
            # Construct a search query for weather
            query = f"current weather in {city}, {country}"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract weather information from search results
            prompt = f"""
            Extract the current weather information for {city}, {country} from these search results:
            
            {search_results}
            
            Extract the following information:
            1. Current weather condition (e.g., sunny, cloudy, rainy)
            2. Current temperature (in both Fahrenheit and Celsius if available)
            3. Precipitation (if any)
            
            Format your response as a JSON object with these fields. If information is not available, use "Unknown" as the value.
            """
            
            weather_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract weather information from search results as a JSON object.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(weather_response, list) and len(weather_response) > 0:
                weather_text = weather_response[0]
            elif isinstance(weather_response, dict) and "text" in weather_response:
                weather_text = weather_response["text"]
            elif isinstance(weather_response, str):
                weather_text = weather_response
            else:
                raise ValueError(f"Unexpected response format: {type(weather_response)}")
            
            # Parse the JSON
            weather_data = json.loads(weather_text)
            
            # Add data source
            weather_data["data_source"] = "DuckDuckGo Search Results"
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error getting real weather data: {e}")
            
            # Return fallback weather data
            return {
                "weather_condition": "Clear",
                "temperature": "72°F (22°C)",
                "precipitation": "None",
                "data_source": "Fallback - Error retrieving weather"
            }
    
    async def _get_real_news_data(self, city: str, country: str) -> List[Dict]:
        """Get real news data for the specified city and country."""
        try:
            # Construct a search query for news
            query = f"latest news in {city}, {country}"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract news information from search results
            prompt = f"""
            Extract the latest news for {city}, {country} from these search results:
            
            {search_results}
            
            Extract 3-5 recent news items, each with:
            1. Headline
            2. Brief summary (1-2 sentences)
            
            Format your response as a JSON array of objects, each with "headline" and "summary" fields.
            Only include real news that is explicitly mentioned in the search results.
            """
            
            news_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract news information from search results as a JSON array.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(news_response, list) and len(news_response) > 0:
                news_text = news_response[0]
            elif isinstance(news_response, dict) and "text" in news_response:
                news_text = news_response["text"]
            elif isinstance(news_response, str):
                news_text = news_response
            else:
                raise ValueError(f"Unexpected response format: {type(news_response)}")
            
            # Parse the JSON
            news_data = json.loads(news_text)
            
            # Add data source to each news item
            for item in news_data:
                item["data_source"] = "DuckDuckGo Search Results"
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error getting real news data: {e}")
            
            # Return fallback news data
            return [
                {
                    "headline": "Local Government Announces Infrastructure Project",
                    "summary": "The city has announced plans for road improvements and public transportation upgrades.",
                    "data_source": "Fallback - Error retrieving news"
                },
                {
                    "headline": "Community Festival Planned for Next Month",
                    "summary": "Local organizers are preparing for the annual community festival with music, food, and activities.",
                    "data_source": "Fallback - Error retrieving news"
                }
            ]
    
    async def _get_local_events(self, city: str, country: str) -> List[Dict]:
        """Get local events for the specified city and country."""
        try:
            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Construct a search query for events
            query = f"upcoming events in {city}, {country} this week"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract event information from search results
            prompt = f"""
            Extract upcoming events in {city}, {country} from these search results:
            
            {search_results}
            
            Extract 3-5 upcoming events, each with:
            1. Event name
            2. Brief description
            3. Date/time (if available)
            4. Location (if available)
            
            Format your response as a JSON array of objects, each with "name", "description", "date", and "location" fields.
            Only include real events that are explicitly mentioned in the search results.
            """
            
            events_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract event information from search results as a JSON array.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(events_response, list) and len(events_response) > 0:
                events_text = events_response[0]
            elif isinstance(events_response, dict) and "text" in events_response:
                events_text = events_response["text"]
            elif isinstance(events_response, str):
                events_text = events_response
            else:
                raise ValueError(f"Unexpected response format: {type(events_response)}")
            
            # Parse the JSON
            events_data = json.loads(events_text)
            
            # Add data source to each event
            for event in events_data:
                event["data_source"] = "DuckDuckGo Search Results"
            
            return events_data
            
        except Exception as e:
            logger.error(f"Error getting local events: {e}")
            
            # Return fallback events data
            return [
                {
                    "name": "Farmers Market",
                    "description": "Weekly farmers market with local produce and crafts",
                    "date": "Every Saturday",
                    "location": "City Square",
                    "data_source": "Fallback - Error retrieving events"
                },
                {
                    "name": "Art Exhibition",
                    "description": "Local artists showcase their work",
                    "date": "This weekend",
                    "location": "Community Gallery",
                    "data_source": "Fallback - Error retrieving events"
                }
            ]
    
    async def _create_default_immediate_environment(self, location_name: str) -> Dict:
        """Create a detailed immediate environment using real-world data about the location."""
        try:
            # Get city and country from world state
            city = self.world_state.get("city", "New York")
            country = self.world_state.get("country", "United States")
            
            # Get detailed information about the location
            location_details = await self._get_location_details(location_name, city, country)
            
            # Get nearby locations
            nearby_locations = await self._get_nearby_locations(location_name)
            
            # Determine if it's indoor or outdoor based on location type
            location_type = location_details.get("location_type", "").lower()
            indoor_types = ["restaurant", "cafe", "museum", "library", "store", "shop", "mall", "cinema", "theater", "hotel"]
            outdoor_types = ["park", "garden", "plaza", "square", "beach", "trail", "market"]
            
            is_indoor = any(t in location_type for t in indoor_types)
            is_outdoor = any(t in location_type for t in outdoor_types)
            
            if is_indoor and not is_outdoor:
                indoor_outdoor = "Indoor"
            elif is_outdoor and not is_indoor:
                indoor_outdoor = "Outdoor"
            else:
                indoor_outdoor = "Mixed indoor and outdoor"
            
            # Create environment based on real data
            environment = {
                "current_location_name": location_details.get("full_name", location_name),
                "location_type": location_details.get("location_type", "Unknown"),
                "indoor_outdoor": indoor_outdoor,
                "description": location_details.get("description", "No description available"),
                "features": location_details.get("features", []),
                "opening_hours": location_details.get("opening_hours", "Not available"),
                "popular_for": location_details.get("popular_for", "Not specified"),
                "price_level": location_details.get("price_level", "Not available"),
                "busy_times": location_details.get("busy_times", "Not available"),
                "rating": location_details.get("rating", "Not available"),
                "address": location_details.get("address", "Not available"),
                "data_source": "DuckDuckGo Search Results"
            }
            
            # Add nearby locations if available
            if nearby_locations:
                environment["nearby_locations"] = nearby_locations
            
            # Add weather-related information if outdoor
            if is_outdoor:
                weather = self.world_state.get("weather", {})
                environment.update({
                    "weather_condition": weather.get("weather_condition", "Not available"),
                    "temperature": weather.get("temperature", "Not available"),
                    "precipitation": weather.get("precipitation", "None")
                })
            
            # Add standard fields based on location type
            if is_indoor:
                environment.update({
                    "noise_level": "Typical for a " + location_details.get("location_type", "place"),
                    "lighting": "Standard indoor lighting",
                    "temperature_feeling": "Climate controlled",
                    "air_quality": "Indoor air"
                })
            else:
                environment.update({
                    "noise_level": "Typical outdoor sounds",
                    "lighting": "Natural light",
                    "temperature_feeling": "Outdoor temperature",
                    "air_quality": "Outdoor air"
                })
            
            # Add people and activities based on location type and time
            current_time = self.world_state.get("current_time", "12:00")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            # Determine crowd density based on time and day
            try:
                hour = int(current_time.split(":")[0])
                is_weekend = day_of_week in ["Saturday", "Sunday"]
                
                # Busy times for different location types
                if "restaurant" in location_type or "cafe" in location_type:
                    if (11 <= hour <= 14) or (17 <= hour <= 21):  # Lunch or dinner time
                        crowd_density = "Busy" if is_weekend else "Moderately busy"
                    elif 7 <= hour <= 10:  # Breakfast time
                        crowd_density = "Moderately busy" if is_weekend else "Somewhat quiet"
                    elif 14 < hour < 17:  # Afternoon lull
                        crowd_density = "Somewhat quiet"
                    else:  # Late night
                        crowd_density = "Quiet" if hour >= 22 else "Somewhat quiet"
                elif "park" in location_type or "garden" in location_type:
                    if 10 <= hour <= 18 and is_weekend:  # Weekend daytime
                        crowd_density = "Busy"
                    elif 10 <= hour <= 18:  # Weekday daytime
                        crowd_density = "Moderately busy"
                    else:  # Early morning or evening
                        crowd_density = "Quiet" if hour >= 20 or hour < 7 else "Somewhat quiet"
                elif "mall" in location_type or "shop" in location_type or "store" in location_type:
                    if 12 <= hour <= 18 and is_weekend:  # Weekend shopping hours
                        crowd_density = "Busy"
                    elif 12 <= hour <= 18:  # Weekday shopping hours
                        crowd_density = "Moderately busy"
                    else:  # Early or late hours
                        crowd_density = "Quiet" if hour >= 20 or hour < 9 else "Somewhat quiet"
                else:
                    # Default crowd density
                    if 9 <= hour <= 17:  # Business hours
                        crowd_density = "Moderately busy"
                    else:
                        crowd_density = "Somewhat quiet"
            except:
                # Default if time parsing fails
                crowd_density = "Moderately busy"
            
            # Add crowd information
            environment["crowd_density"] = crowd_density
            
            # Add people and activities based on location type
            if "restaurant" in location_type or "cafe" in location_type:
                environment["present_people"] = ["Customers", "Wait staff", "Baristas/Cooks"]
                environment["social_atmosphere"] = "Casual dining atmosphere"
                environment["ongoing_activities"] = ["People eating", "Conversations", "Staff serving customers"]
            elif "park" in location_type or "garden" in location_type:
                environment["present_people"] = ["Visitors", "Joggers", "People walking dogs", "Families"]
                environment["social_atmosphere"] = "Relaxed outdoor leisure"
                environment["ongoing_activities"] = ["Walking", "Jogging", "Picnicking", "People enjoying nature"]
            elif "museum" in location_type or "gallery" in location_type:
                environment["present_people"] = ["Visitors", "Tour guides", "Staff"]
                environment["social_atmosphere"] = "Quiet appreciation"
                environment["ongoing_activities"] = ["People viewing exhibits", "Guided tours", "Photography"]
            elif "mall" in location_type or "shop" in location_type or "store" in location_type:
                environment["present_people"] = ["Shoppers", "Store employees", "Security personnel"]
                environment["social_atmosphere"] = "Retail environment"
                environment["ongoing_activities"] = ["Shopping", "Browsing", "Staff assisting customers"]
            else:
                environment["present_people"] = ["Various people appropriate for this location"]
                environment["social_atmosphere"] = "Typical for this type of location"
                environment["ongoing_activities"] = ["Activities appropriate for this location"]
            
            self.console.print(f"[green]Successfully created detailed environment for {location_name}[/green]")
            return environment
            
        except Exception as e:
            logger.error(f"Error creating detailed environment: {e}")
            self.console.print(f"[red]Error creating detailed environment: {e}. Using basic environment.[/red]")
            
            # Fall back to a simpler method
            return {
                "current_location_name": location_name,
                "location_type": "Unknown location type",
                "indoor_outdoor": "Unknown",
                "description": "No information available about this location",
                "features": ["Unknown features"],
                "present_people": ["Various people"],
                "crowd_density": "Moderate",
                "social_atmosphere": "Neutral",
                "ongoing_activities": ["Various activities"],
                "data_source": "Fallback - No data retrieved",
                "note": "Location information could not be retrieved. Using placeholder values."
            }
    
    async def _get_location_details(self, location_name: str, city: str, country: str) -> Dict:
        """Get detailed information about a location using web search."""
        try:
            # Construct a search query
            query = f"{location_name} in {city}, {country} details information"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract location details from search results
            prompt = f"""
            Extract detailed information about "{location_name}" in {city}, {country} from these search results:
            
            {search_results}
            
            Extract the following information if available:
            1. Full name of the location
            2. Location type (e.g., restaurant, park, museum, cafe, etc.)
            3. Description (a brief description of the place)
            4. Features (notable features of the place)
            5. Opening hours
            6. Address
            7. Popular for (what the place is known for)
            8. Price level (if applicable)
            9. Busy times (if mentioned)
            10. Rating (if mentioned)
            
            Format your response as a JSON object with these fields. If information is not available, use "Not available" or an empty list for features.
            
            IMPORTANT: Only include information that is explicitly mentioned in the search results. Do not make up or infer details that aren't present.
            """
            
            details_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract location details from search results as a JSON object. Only include information explicitly mentioned in the results.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(details_response, list) and len(details_response) > 0:
                details_text = details_response[0]
            elif isinstance(details_response, dict) and "text" in details_response:
                details_text = details_response["text"]
            elif isinstance(details_response, str):
                details_text = details_response
            else:
                raise ValueError(f"Unexpected response format: {type(details_response)}")
            
            # Parse the JSON
            details = json.loads(details_text)
            
            # Add data source
            details["data_source"] = "DuckDuckGo Search Results"
            
            # Ensure all expected fields are present
            expected_fields = [
                "full_name", "location_type", "description", "features", 
                "opening_hours", "address", "popular_for", "price_level", 
                "busy_times", "rating"
            ]
            
            for field in expected_fields:
                if field not in details:
                    if field == "features":
                        details[field] = []
                    else:
                        details[field] = "Not available"
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting location details: {e}")
            
            # Create fallback location details
            return {
                "full_name": location_name,
                "location_type": self._guess_location_type(location_name),
                "description": f"A place called {location_name} in {city}, {country}.",
                "features": [],
                "opening_hours": "Not available",
                "address": f"Somewhere in {city}, {country}",
                "popular_for": "Not available",
                "price_level": "Not available",
                "busy_times": "Not available",
                "rating": "Not available",
                "data_source": "Fallback - Error retrieving details"
            }
    
    def _guess_location_type(self, location_name: str) -> str:
        """Make an educated guess about the location type based on its name."""
        location_name_lower = location_name.lower()
        
        # Common location types and their keywords
        location_types = {
            "restaurant": ["restaurant", "bistro", "eatery", "grill", "diner", "steakhouse", "pizzeria"],
            "cafe": ["cafe", "coffee", "bakery", "patisserie", "tea"],
            "park": ["park", "garden", "gardens", "square", "commons", "green"],
            "museum": ["museum", "gallery", "exhibition", "memorial"],
            "library": ["library", "archives"],
            "store": ["store", "shop", "market", "boutique", "mall", "outlet", "supermarket"],
            "bar": ["bar", "pub", "tavern", "brewery", "lounge"],
            "hotel": ["hotel", "inn", "motel", "hostel", "resort", "suites"],
            "theater": ["theater", "theatre", "cinema", "movies"],
            "gym": ["gym", "fitness", "wellness", "spa"],
            "school": ["school", "university", "college", "academy", "institute"],
            "hospital": ["hospital", "clinic", "medical", "health"],
            "office": ["office", "headquarters", "building", "tower", "center", "centre"],
            "station": ["station", "terminal", "airport", "port", "dock"]
        }
        
        # Check for matches
        for location_type, keywords in location_types.items():
            if any(keyword in location_name_lower for keyword in keywords):
                return location_type
        
        # If no match found, make a guess based on common naming patterns
        if any(char.isdigit() for char in location_name):
            return "address"
        elif "street" in location_name_lower or "avenue" in location_name_lower or "road" in location_name_lower:
            return "street"
        elif "square" in location_name_lower or "plaza" in location_name_lower:
            return "plaza"
        
        # Default
        return "place"
    
    async def _get_nearby_locations(self, location_name: str = None) -> List[Dict]:
        """Get nearby locations based on the current location."""
        try:
            # Get city and country from world state
            city = self.world_state.get("city", "New York")
            country = self.world_state.get("country", "United States")
            
            # Use provided location name or get from immediate environment
            if location_name is None:
                location_name = self.immediate_environment.get("current_location_name", "Unknown")
            
            # Construct a search query
            query = f"places near {location_name} in {city}, {country}"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract nearby locations from search results
            prompt = f"""
            Extract a list of real places near "{location_name}" in {city}, {country} from these search results:
            
            {search_results}
            
            Extract 3-7 nearby places, each with:
            1. Name of the place
            2. Type of place (e.g., restaurant, park, museum, etc.)
            3. Distance from {location_name} (if mentioned)
            
            Format your response as a JSON array of objects, each with "name", "type", and "distance" fields.
            
            IMPORTANT: Only include places that are explicitly mentioned in the search results. Do not make up or infer places that aren't present.
            """
            
            nearby_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract nearby locations from search results as a JSON array. Only include places explicitly mentioned in the results.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(nearby_response, list) and len(nearby_response) > 0:
                nearby_text = nearby_response[0]
            elif isinstance(nearby_response, dict) and "text" in nearby_response:
                nearby_text = nearby_response["text"]
            elif isinstance(nearby_response, str):
                nearby_text = nearby_response
            else:
                raise ValueError(f"Unexpected response format: {type(nearby_response)}")
            
            # Parse the JSON
            nearby_places = json.loads(nearby_text)
            
            # Add data source to each place
            for place in nearby_places:
                place["data_source"] = "DuckDuckGo Search Results"
            
            return nearby_places
            
        except Exception as e:
            logger.error(f"Error getting nearby locations: {e}")
            
            # Return fallback nearby places
            return [
                {
                    "name": "Local Cafe",
                    "type": "Cafe",
                    "distance": "Nearby",
                    "data_source": "Fallback - Error retrieving nearby places"
                },
                {
                    "name": "Convenience Store",
                    "type": "Store",
                    "distance": "Within walking distance",
                    "data_source": "Fallback - Error retrieving nearby places"
                },
                {
                    "name": "Public Park",
                    "type": "Park",
                    "distance": "A few blocks away",
                    "data_source": "Fallback - Error retrieving nearby places"
                }
            ]
    
    async def _check_location_hours(self) -> Dict:
        """Check if the current location is open based on real-time data."""
        try:
            # Get current time and day of week
            current_time = self.world_state.get("current_time", "12:00")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            # Get location details
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            location_type = self.immediate_environment.get("location_type", "Unknown")
            
            # If opening hours are already available in the immediate environment, use those
            if "opening_hours" in self.immediate_environment and self.immediate_environment["opening_hours"] != "Not available":
                opening_hours = self.immediate_environment["opening_hours"]
            else:
                # Get city and country
                city = self.world_state.get("city", "New York")
                country = self.world_state.get("country", "United States")
                
                # Perform a web search to find opening hours
                query = f"{location_name} {location_type} opening hours {city} {country}"
                
                search = DuckDuckGoSearchRun()
                search_results = search.run(query)
                
                # Use the LLM to extract opening hours from search results
                prompt = f"""
                Extract the opening hours for "{location_name}" ({location_type}) in {city}, {country} from these search results:
                
                {search_results}
                
                Extract ONLY the opening hours information. If different hours are listed for different days, include that information.
                
                Return ONLY the opening hours text, nothing else. If no opening hours are found, return "Not available".
                """
                
                hours_response = await self.llm_service.generate_content(
                    prompt=prompt,
                    system_instruction="Extract opening hours from search results. Return only the hours text.",
                    temperature=0.3
                )
                
                # Process the response
                if isinstance(hours_response, list) and len(hours_response) > 0:
                    opening_hours = hours_response[0]
                elif isinstance(hours_response, dict) and "text" in hours_response:
                    opening_hours = hours_response["text"]
                elif isinstance(hours_response, str):
                    opening_hours = hours_response
                else:
                    opening_hours = "Not available"
                
                # Clean up the hours text
                opening_hours = opening_hours.strip().strip('"\'.,;:')
                
                # Update the immediate environment with the opening hours
                self.immediate_environment["opening_hours"] = opening_hours
            
            # Check if the location is currently open
            is_open = True  # Default to open
            
            # If we have valid opening hours, check if the location is open
            if opening_hours != "Not available" and "24/7" not in opening_hours.lower() and "always open" not in opening_hours.lower():
                # Parse the current time
                try:
                    current_hour = int(current_time.split(":")[0])
                    current_minute = int(current_time.split(":")[1])
                    
                    # Simple heuristic for common opening hours patterns
                    if "closed" in opening_hours.lower() and day_of_week.lower() in opening_hours.lower():
                        is_open = False
                    elif any(f"{day_of_week.lower()}: closed" in opening_hours.lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                        is_open = False
                    elif "9" in opening_hours and "5" in opening_hours and (current_hour < 9 or current_hour >= 17):
                        is_open = False
                    elif "10" in opening_hours and "6" in opening_hours and (current_hour < 10 or current_hour >= 18):
                        is_open = False
                    elif "8" in opening_hours and "8" in opening_hours and (current_hour < 8 or current_hour >= 20):
                        is_open = False
                    elif "closed" in opening_hours.lower() and "open" not in opening_hours.lower():
                        is_open = False
                except:
                    # If parsing fails, assume it's open
                    pass
            
            return {
                "is_open": is_open,
                "hours": opening_hours,
                "data_source": "DuckDuckGo Search Results"
            }
            
        except Exception as e:
            logger.error(f"Error checking location hours: {e}")
            
            # Return fallback status
            return {
                "is_open": True,  # Default to open
                "hours": "Not available",
                "data_source": "Fallback - Error checking hours"
            }
    
    async def _generate_initial_narrative(self) -> str:
        """Generate the initial narrative context for the simulation."""
        try:
            # Extract relevant information
            persona_name = self.simulacra.persona.get("name", "Unknown")
            persona_occupation = self.simulacra.persona.get("occupation", "Unknown")
            persona_goal = self.simulacra.persona.get("current_goal", "Unknown")
            persona_emotion = self.simulacra.persona.get("current_emotion", "neutral")
            
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            location_type = self.immediate_environment.get("location_type", "Unknown")
            location_description = self.immediate_environment.get("description", "No description available")
            
            city = self.world_state.get("city", "Unknown")
            country = self.world_state.get("country", "Unknown")
            current_time = self.world_state.get("current_time", "Unknown")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            weather = self.world_state.get("weather", {})
            weather_condition = weather.get("weather_condition", "Unknown")
            temperature = weather.get("temperature", "Unknown")
            
            # Create a prompt for generating the narrative
            prompt = f"""
            Generate an initial narrative context for a simulation with the following details:
            
            Character:
            - Name: {persona_name}
            - Occupation: {persona_occupation}
            - Current Goal: {persona_goal}
            - Current Emotion: {persona_emotion}
            
            Location:
            - Name: {location_name}
            - Type: {location_type}
            - Description: {location_description}
            - City: {city}, {country}
            
            Time and Weather:
            - Current Time: {current_time}
            - Day of Week: {day_of_week}
            - Weather: {weather_condition}, {temperature}
            
            Write a first-person narrative paragraph (3-5 sentences) that sets the scene for the character at this location.
            Focus on sensory details, the character's current state of mind, and their immediate surroundings.
            """
            
            # Generate the narrative
            narrative_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Write an engaging, first-person narrative paragraph that sets the scene for a simulation.",
                temperature=0.7
            )
            
            # Process the response
            if isinstance(narrative_response, list) and len(narrative_response) > 0:
                narrative = narrative_response[0]
            elif isinstance(narrative_response, dict) and "text" in narrative_response:
                narrative = narrative_response["text"]
            elif isinstance(narrative_response, str):
                narrative = narrative_response
            else:
                raise ValueError(f"Unexpected response format: {type(narrative_response)}")
            
            # Clean up the narrative
            narrative = narrative.strip().strip('"')
            
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating initial narrative: {e}")
            
            # Create a simple fallback narrative
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            city = self.world_state.get("city", "Unknown")
            country = self.world_state.get("country", "Unknown")
            current_time = self.world_state.get("current_time", "Unknown")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            return f"I find myself at {location_name} in {city}, {country}. It's {current_time} on {day_of_week}. I take a moment to gather my thoughts and observe my surroundings."
    
    async def _generate_updated_narrative(self, action: str, consequences: List[str]) -> str:
        """Generate an updated narrative context based on the character's action and consequences."""
        try:
            # Extract relevant information
            persona_name = self.simulacra.persona.get("name", "Unknown")
            
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            
            city = self.world_state.get("city", "Unknown")
            current_time = self.world_state.get("current_time", "Unknown")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            weather = self.world_state.get("weather", {})
            weather_condition = weather.get("weather_condition", "Unknown")
            
            # Format the consequences for the prompt
            consequences_text = "\n".join([f"- {consequence}" for consequence in consequences[:2]])  # Limit to 2 consequences for brevity
            
            # Create a prompt for generating the updated narrative
            prompt = f"""
            Update the narrative context based on the character's action and its consequences:
            
            Character: {persona_name}
            Location: {location_name} in {city}
            Time: {current_time} on {day_of_week}
            Weather: {weather_condition}
            
            Previous narrative context:
            {self.narrative_context}
            
            Action taken:
            {action}
            
            Consequences:
            {consequences_text}
            
            Write an updated first-person narrative paragraph (3-5 sentences) that continues the story.
            Maintain continuity with the previous narrative while incorporating the action and its consequences.
            Focus on the character's experience, sensory details, and current situation.
            """
            
            # Generate the narrative
            narrative_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Write an engaging, first-person narrative paragraph that updates the simulation context.",
                temperature=0.7
            )
            
            # Process the response
            if isinstance(narrative_response, list) and len(narrative_response) > 0:
                narrative = narrative_response[0]
            elif isinstance(narrative_response, dict) and "text" in narrative_response:
                narrative = narrative_response["text"]
            elif isinstance(narrative_response, str):
                narrative = narrative_response
            else:
                raise ValueError(f"Unexpected response format: {type(narrative_response)}")
            
            # Clean up the narrative
            narrative = narrative.strip().strip('"')
            
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating updated narrative: {e}")
            
            # Create a simple fallback narrative update
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            
            # Simplify the consequences for the fallback
            consequences_summary = ". ".join(consequences[:1]) if consequences else "Nothing significant happened."
            
            return f"{self.narrative_context}\n\nI {action.lower().replace('I ', '')}. {consequences_summary} I continue to observe my surroundings at {location_name}."
    
    async def process_perception(self) -> Dict:
        """Process perception for the simulacra."""
        try:
            # Get current time and date
            current_time = self.world_state.get("current_time", "Unknown")
            current_date = self.world_state.get("current_date", "Unknown")
            day_of_week = self.world_state.get("day_of_week", "Unknown")
            
            # Get location information
            city = self.world_state.get("city", "Unknown")
            country = self.world_state.get("country", "Unknown")
            current_location = self.immediate_environment.get("current_location_name", "Unknown")
            location_type = self.immediate_environment.get("location_type", "Unknown")
            location_description = self.immediate_environment.get("description", "No description available")
            
            # Get weather information
            weather = self.world_state.get("weather", {})
            weather_condition = weather.get("weather_condition", "Unknown")
            temperature = weather.get("temperature", "Unknown")
            
            # Get social context
            present_people = self.immediate_environment.get("present_people", ["Unknown"])
            crowd_density = self.immediate_environment.get("crowd_density", "Unknown")
            social_atmosphere = self.immediate_environment.get("social_atmosphere", "Unknown")
            ongoing_activities = self.immediate_environment.get("ongoing_activities", ["Unknown"])
            
            # Get nearby locations
            nearby_locations = self.immediate_environment.get("nearby_locations", [])
            
            # Get news and events
            news = self.world_state.get("news", [])
            local_events = self.world_state.get("local_events", [])
            
            # Construct the perception object
            perception = {
                "time_context": {
                    "current_time": current_time,
                    "current_date": current_date,
                    "day_of_week": day_of_week
                },
                "location_context": {
                    "city": city,
                    "country": country,
                    "current_location": current_location,
                    "location_type": location_type,
                    "description": location_description,
                    "nearby_locations": nearby_locations
                },
                "environmental_context": {
                    "weather": {
                        "condition": weather_condition,
                        "temperature": temperature
                    },
                    "indoor_outdoor": self.immediate_environment.get("indoor_outdoor", "Unknown")
                },
                "social_context": {
                    "present_people": present_people,
                    "crowd_density": crowd_density,
                    "social_atmosphere": social_atmosphere,
                    "ongoing_activities": ongoing_activities
                },
                "self_context": self.simulacra.persona,
                "narrative_context": self.narrative_context,
                "world_context": {
                    "news": news[:3] if news else [],  # Include up to 3 news items
                    "local_events": local_events[:3] if local_events else []  # Include up to 3 local events
                }
            }
            
            # Check if the last action was a search action
            last_action = self.simulacra.persona.get("recent_actions", [""])[-1] if self.simulacra.persona.get("recent_actions") else ""
            
            if "search" in last_action.lower() or "look up" in last_action.lower() or "google" in last_action.lower():
                # Extract the search query from the action
                search_query = self._extract_search_query(last_action)
                
                if search_query:
                    # Perform the search
                    search_result = await self.perform_search(search_query)
                    
                    # Add the search result to the perception
                    perception["search_result"] = search_result
                    
                    # Update the narrative context to include the search
                    self.narrative_context += f"\n\nI searched for information about '{search_query}' and found: {search_result['summary']}"
            
            return perception
            
        except Exception as e:
            logger.error(f"Error processing perception: {e}")
            self.console.print(f"[red]Error processing perception: {e}[/red]")
            
            # Return a minimal perception object in case of error
            return {
                "time_context": {"current_time": "Unknown", "current_date": "Unknown", "day_of_week": "Unknown"},
                "location_context": {"city": "Unknown", "country": "Unknown", "current_location": "Unknown"},
                "environmental_context": {"weather": {"condition": "Unknown", "temperature": "Unknown"}},
                "social_context": {"present_people": ["Unknown"], "social_atmosphere": "Unknown"},
                "self_context": self.simulacra.persona,
                "narrative_context": "There was an error processing your perception of the world."
            }
    
    def _extract_search_query(self, action: str) -> str:
        """Extract a search query from an action text."""
        search_indicators = [
            "search for", "search about", "look up", "google", "find information about",
            "research", "check online for", "search the web for", "look for information on"
        ]
        
        action_lower = action.lower()
        
        for indicator in search_indicators:
            if indicator in action_lower:
                # Find the position of the indicator
                start_pos = action_lower.find(indicator) + len(indicator)
                
                # Extract everything after the indicator
                query_text = action[start_pos:].strip()
                
                # Clean up the query
                query_text = query_text.strip('".\',:;')
                
                # Handle cases where the query is part of a larger sentence
                end_markers = ['.', '!', '?', 'and', 'then', 'while', 'before', 'after']
                for marker in end_markers:
                    if f" {marker} " in query_text:
                        query_text = query_text.split(f" {marker} ")[0]
                
                return query_text
        
        # If no specific search indicator is found but the action mentions search
        if "search" in action_lower:
            # Try to extract what comes after "search"
            parts = action_lower.split("search")
            if len(parts) > 1:
                query_text = parts[1].strip('".\',:; ')
                return query_text
        
        return ""
    
    async def perform_search(self, query: str) -> Dict:
        """Perform a web search based on a query from the simulacra."""
        try:
            self.console.print(f"[yellow]Performing web search: {query}[/yellow]")
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to summarize the search results
            prompt = f"""
            Summarize the following search results for the query: "{query}"
            
            {search_results}
            
            IMPORTANT:
            1. ONLY include information that is explicitly mentioned in the search results
            2. Organize the information in a clear, concise manner
            3. If the search results don't contain relevant information, state that clearly
            4. Include 3-5 key points from the search results
            
            Format your response as a search result summary that would be helpful to someone who asked this question.
            """
            
            summary_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Summarize search results in a helpful, factual manner. Only include information from the search results.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(summary_response, list) and len(summary_response) > 0:
                summary = summary_response[0]
            elif isinstance(summary_response, dict) and "text" in summary_response:
                summary = summary_response["text"]
            elif isinstance(summary_response, str):
                summary = summary_response
            else:
                raise ValueError(f"Unexpected response format: {type(summary_response)}")
            
            # Clean up the summary
            summary = summary.strip()
            
            self.console.print(f"[green]Search completed[/green]")
            
            return {
                "query": query,
                "summary": summary,
                "data_source": "DuckDuckGo Search Results"
            }
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            self.console.print(f"[red]Error performing search: {e}[/red]")
            
            return {
                "query": query,
                "summary": f"The search for '{query}' failed due to a technical error.",
                "data_source": "Error - Search Failed"
            }
    
    async def process_update(self, action: str) -> Dict:
        """Process an update to the world state based on an action."""
        self.console.print(f"[cyan]Processing action: {action}[/cyan]")
        
        # Update real-time data
        await self._update_real_time_data()
        
        # Check if the current location is open
        location_status = await self._check_location_hours()
        if location_status.get("is_open") is False:
            self.console.print(f"[yellow]Note: {self.immediate_environment.get('current_location_name')} is currently closed. Opening hours: {location_status.get('hours', 'Unknown')}[/yellow]")
        
        # Get nearby locations
        nearby_places = await self._get_nearby_locations()
        if nearby_places:
            self.immediate_environment["nearby_locations"] = nearby_places
        
        # Check if this is a search action
        is_search_action = "search" in action.lower() or "look up" in action.lower() or "google" in action.lower()
        
        if is_search_action:
            search_query = self._extract_search_query(action)
            if search_query:
                search_result = await self.perform_search(search_query)
                
                # Generate a response that incorporates the search result
                prompt = f"""
                The character has performed a search for: "{search_query}"
                
                Search results summary:
                {search_result['summary']}
                
                Based on this search, describe:
                1. What the character learns from this search
                2. How this information might affect their current situation
                3. Any observations about the search results
                
                Format your response as a JSON object with the following structure:
                {{
                    "consequences": ["List of 1-3 consequences of the search"],
                    "observations": ["List of 1-3 observations about the search results"]
                }}
                """
                
                system_instruction = "You are simulating realistic world reactions to a character's search action. Provide consequences and observations in the requested JSON format."
                
                response = await self.llm_service.generate_content(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=0.5
                )
                
                # Process the response
                try:
                    if isinstance(response, list) and len(response) > 0:
                        response_text = response[0]
                    elif isinstance(response, dict) and "text" in response:
                        response_text = response["text"]
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        raise ValueError(f"Unexpected response format: {type(response)}")
                    
                    # Extract JSON from the response
                    response_json = json.loads(response_text)
                    
                    consequences = response_json.get("consequences", ["You found some information from your search."])
                    observations = response_json.get("observations", ["The search results provided some insights."])
                    
                    # Update the narrative context
                    updated_narrative = await self._generate_updated_narrative(action, consequences)
                    self.narrative_context = updated_narrative
                    
                    # Return the results
                    return {
                        "consequences": consequences,
                        "observations": observations,
                        "search_result": search_result,
                        "nearby_places": nearby_places[:3] if nearby_places else []
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing search response: {e}")
                    self.console.print(f"[red]Error processing search response: {e}[/red]")
                    
                    # Fallback response
                    consequences = ["You found some information from your search."]
                    observations = ["The search results provided some insights."]
                    
                    # Update the narrative context
                    updated_narrative = await self._generate_updated_narrative(action, consequences)
                    self.narrative_context = updated_narrative
                    
                    return {
                        "consequences": consequences,
                        "observations": observations,
                        "search_result": search_result,
                        "nearby_places": nearby_places[:3] if nearby_places else []
                    }
        
        # For non-search actions, proceed with normal update
        prompt = PromptManager.process_update_prompt(
            world_state=self.world_state,
            immediate_environment=self.immediate_environment,
            simulacra=self.simulacra.persona,
            action=action,
            reaction_profile=self.reaction_profile
        )
        
        system_instruction = "You are simulating realistic world reactions to a character's action. Provide consequences and observations in the requested JSON format."
        
        response = await self.llm_service.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.5
        )
        
        # Process the response
        try:
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0]
            elif isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            elif isinstance(response, str):
                response_text = response
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            # Extract JSON from the response
            response_json = json.loads(response_text)
            
            consequences = response_json.get("consequences", ["Your action had no noticeable consequences."])
            observations = response_json.get("observations", ["You didn't observe anything significant."])
            
            # Update the immediate environment if provided
            if "environment_updates" in response_json:
                environment_updates = response_json["environment_updates"]
                for key, value in environment_updates.items():
                    if key in self.immediate_environment:
                        self.immediate_environment[key] = value
            
            # Update the narrative context
            updated_narrative = await self._generate_updated_narrative(action, consequences)
            self.narrative_context = updated_narrative
            
            # Return the results
            return {
                "consequences": consequences,
                "observations": observations,
                "nearby_places": nearby_places[:3] if nearby_places else []
            }
            
        except Exception as e:
            logger.error(f"Error processing update response: {e}")
            self.console.print(f"[red]Error processing update response: {e}[/red]")
            
            # Fallback response
            consequences = ["Your action had no noticeable consequences."]
            observations = ["You didn't observe anything significant."]
            
            # Update the narrative context with a simple fallback
            self.narrative_context += f"\n\nI {action.lower().replace('I ', '')}. Nothing significant happened."
            
            return {
                "consequences": consequences,
                "observations": observations,
                "nearby_places": nearby_places[:3] if nearby_places else []
            }
    
    async def change_location(self, new_location_name: str) -> Dict:
        """Change the character's location to a new place."""
        try:
            self.console.print(f"[cyan]Changing location to: {new_location_name}[/cyan]")
            
            # Get current city and country
            city = self.world_state.get("city", "New York")
            country = self.world_state.get("country", "United States")
            
            # Get current location for reference
            current_location = self.immediate_environment.get("current_location_name", "Unknown")
            
            # Calculate travel time based on distance (simplified)
            travel_time = await self._calculate_travel_time(current_location, new_location_name, city, country)
            
            # Advance time based on travel
            if travel_time > 0:
                self.advance_time(travel_time)
                self.console.print(f"[yellow]Traveled for {travel_time} minutes to reach {new_location_name}[/yellow]")
            
            # Create a new immediate environment for the new location
            new_environment = await self._create_default_immediate_environment(new_location_name)
            
            # Update the immediate environment
            self.immediate_environment = new_environment
            
            # Update the world state with the new location
            self.world_state["current_location"] = new_location_name
            
            # Generate a transition narrative
            transition_prompt = f"""
            The character has moved from {current_location} to {new_location_name} in {city}, {country}.
            
            Current time: {self.world_state.get("current_time", "Unknown")}
            Weather: {self.world_state.get("weather", {}).get("weather_condition", "Unknown")}, {self.world_state.get("weather", {}).get("temperature", "Unknown")}
            
            New location description: {new_environment.get("description", "No description available")}
            
            Write a brief, engaging paragraph describing the transition and arrival at the new location.
            Focus on sensory details and the character's first impressions of the new environment.
            """
            
            transition_response = await self.llm_service.generate_content(
                prompt=transition_prompt,
                system_instruction="Write a brief, engaging narrative transition for a character moving to a new location.",
                temperature=0.6
            )
            
            # Process the response
            if isinstance(transition_response, list) and len(transition_response) > 0:
                transition_narrative = transition_response[0]
            elif isinstance(transition_response, dict) and "text" in transition_response:
                transition_narrative = transition_response["text"]
            elif isinstance(transition_response, str):
                transition_narrative = transition_response
            else:
                transition_narrative = f"I arrived at {new_location_name} after leaving {current_location}."
            
            # Update the narrative context
            self.narrative_context = transition_narrative
            
            return {
                "new_location": new_location_name,
                "travel_time": travel_time,
                "transition_narrative": transition_narrative,
                "environment": new_environment
            }
            
        except Exception as e:
            logger.error(f"Error changing location: {e}")
            self.console.print(f"[red]Error changing location: {e}[/red]")
            
            # Fallback - create a simple environment for the new location
            fallback_environment = {
                "current_location_name": new_location_name,
                "location_type": "Unknown",
                "indoor_outdoor": "Unknown",
                "description": f"A location called {new_location_name}",
                "features": ["Basic features"],
                "present_people": ["Various people"],
                "crowd_density": "Moderate",
                "social_atmosphere": "Neutral",
                "ongoing_activities": ["Various activities"],
                "data_source": "Fallback - Error during location change"
            }
            
            # Update the immediate environment with the fallback
            self.immediate_environment = fallback_environment
            
            # Update the world state with the new location
            self.world_state["current_location"] = new_location_name
            
            # Simple fallback narrative
            fallback_narrative = f"I arrived at {new_location_name} after leaving my previous location."
            self.narrative_context = fallback_narrative
            
            return {
                "new_location": new_location_name,
                "travel_time": 15,  # Default travel time
                "transition_narrative": fallback_narrative,
                "environment": fallback_environment
            }
    
    async def _calculate_travel_time(self, origin: str, destination: str, city: str, country: str) -> int:
        """Calculate approximate travel time between two locations in minutes."""
        try:
            # If the locations are the same, no travel time
            if origin.lower() == destination.lower():
                return 0
            
            # Construct a search query for travel time
            query = f"travel time from {origin} to {destination} in {city}, {country}"
            
            search = DuckDuckGoSearchRun()
            search_results = search.run(query)
            
            # Use the LLM to extract travel time from search results
            prompt = f"""
            Extract the approximate travel time between {origin} and {destination} in {city}, {country} from these search results:
            
            {search_results}
            
            If you can find a specific travel time (e.g., "15 minutes by car", "30 minutes walking"), extract that value in minutes.
            If multiple travel methods are mentioned (walking, driving, public transit), prioritize the most common or reasonable method.
            If no specific time is mentioned, estimate based on:
            - Walking: 5 minutes for very close locations, 15 minutes for nearby locations, 30+ minutes for distant locations
            - Driving: 5-10 minutes for locations in the same area, 15-30 minutes for cross-town trips
            
            Return ONLY the number of minutes as an integer. If you cannot determine a time, return 15 as a default.
            """
            
            time_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Extract or estimate travel time in minutes from search results. Return only an integer.",
                temperature=0.3
            )
            
            # Process the response
            if isinstance(time_response, list) and len(time_response) > 0:
                time_text = time_response[0]
            elif isinstance(time_response, dict) and "text" in time_response:
                time_text = time_response["text"]
            elif isinstance(time_response, str):
                time_text = time_response
            else:
                return 15  # Default if response format is unexpected
            
            # Extract the number from the response
            time_text = time_text.strip()
            try:
                travel_time = int(re.search(r'\d+', time_text).group())
                return min(travel_time, 120)  # Cap at 120 minutes to prevent unreasonable values
            except:
                return 15  # Default if parsing fails
                
        except Exception as e:
            logger.error(f"Error calculating travel time: {e}")
            return 15  # Default travel time in minutes
    
    def advance_time(self, minutes: int) -> None:
        """Advance the simulation time by the specified number of minutes."""
        try:
            # Get current time
            current_time = self.world_state.get("current_time", "12:00")
            
            # Parse the time
            try:
                # Try 24-hour format first
                hour, minute = map(int, current_time.split(":"))
            except:
                try:
                    # Try 12-hour format
                    time_parts = current_time.split()
                    hour, minute = map(int, time_parts[0].split(":"))
                    if time_parts[1].lower() == "pm" and hour < 12:
                        hour += 12
                    elif time_parts[1].lower() == "am" and hour == 12:
                        hour = 0
                except:
                    # Default to noon if parsing fails
                    hour, minute = 12, 0
            
            # Advance time
            minute += minutes
            hour += minute // 60
            minute %= 60
            
            # Handle day change
            new_day = False
            if hour >= 24:
                hour %= 24
                new_day = True
            
            # Format the new time
            new_time = f"{hour:02d}:{minute:02d}"
            self.world_state["current_time"] = new_time
            
            # If a new day has started, update the date and day of week
            if new_day:
                try:
                    current_date = datetime.strptime(self.world_state.get("current_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
                    new_date = current_date + timedelta(days=1)
                    self.world_state["current_date"] = new_date.strftime("%Y-%m-%d")
                    self.world_state["day_of_week"] = new_date.strftime("%A")
                    
                    # Get new weather and news for the new day
                    self.console.print(f"[yellow]A new day has begun: {new_date.strftime('%A, %Y-%m-%d')}[/yellow]")
                    
                    # Update weather and news asynchronously
                    asyncio.create_task(self._update_real_time_data())
                except:
                    # If date parsing fails, just note the day change
                    self.console.print("[yellow]A new day has begun.[/yellow]")
            
            # Check if the location status might have changed
            location_name = self.immediate_environment.get("current_location_name", "Unknown")
            opening_hours = self.immediate_environment.get("opening_hours", "Not available")
            
            if opening_hours != "Not available" and "24/7" not in opening_hours.lower() and "always open" not in opening_hours.lower():
                self.console.print(f"[dim]Time is now {new_time}. You may want to check if {location_name} is still open.[/dim]")
            else:
                self.console.print(f"[dim]Time is now {new_time}.[/dim]")
                
        except Exception as e:
            logger.error(f"Error advancing time: {e}")
            self.console.print(f"[red]Error advancing time: {e}[/red]")
            
            # Fallback - just set a reasonable time
            self.world_state["current_time"] = "12:00"
    
    async def save_state(self) -> bool:
        """Save the current state to a file."""
        try:
            state = {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "persona": self.simulacra.persona,
                "narrative_context": self.narrative_context,
                "reaction_profile": self.reaction_profile
            }
            
            with open("simulation_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            self.console.print(f"[red]Error saving state: {e}[/red]")
            return False
    
    async def load_state(self) -> bool:
        """Load the state from a file."""
        try:
            with open("simulation_state.json", "r") as f:
                state = json.load(f)
            
            self.world_state = state.get("world_state", {})
            self.immediate_environment = state.get("immediate_environment", {})
            self.simulacra.persona = state.get("persona", {})
            self.narrative_context = state.get("narrative_context", "")
            self.reaction_profile = state.get("reaction_profile", "balanced")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            self.console.print(f"[red]Error loading state: {e}[/red]")
            return False

async def create_persona(city: str, country: str) -> Dict:
    """Create a new persona based on the specified city and country."""
    try:
        console = Console()
        console.print("[yellow]No persona file found. Creating a new persona...[/yellow]")
        
        # Initialize LLM service
        llm_service = LLMService()
        
        # Generate a prompt for creating a persona
        prompt = f"""
        Create a realistic persona for someone living in {city}, {country}.
        
        The persona should include:
        1. Name (appropriate for the location)
        2. Age (between 18 and 65)
        3. Occupation (realistic for the location)
        4. Personality traits (3-5 traits)
        5. Background (brief life history)
        6. Goals (1-3 current goals)
        7. Current emotional state
        
        Make the persona realistic, nuanced, and appropriate for the location.
        """
        
        system_instruction = """
        You are creating a realistic persona for a simulation. 
        Return the persona as a JSON object with the following structure:
        {
            "name": "Full Name",
            "age": age_as_integer,
            "occupation": "Occupation",
            "personality_traits": ["trait1", "trait2", "trait3"],
            "background": "Brief background story",
            "goals": ["goal1", "goal2"],
            "current_emotion": "emotion",
            "current_goal": "Most immediate goal"
        }
        """
        
        # Generate the persona
        persona_response = await llm_service.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.7
        )
        
        # Process the response
        if isinstance(persona_response, list) and len(persona_response) > 0:
            persona_text = persona_response[0]
        elif isinstance(persona_response, dict) and "text" in persona_response:
            persona_text = persona_response["text"]
        elif isinstance(persona_response, str):
            persona_text = persona_response
        else:
            raise ValueError(f"Unexpected response format: {type(persona_response)}")
        
        # Parse the JSON
        persona = json.loads(persona_text)
        
        # Add additional fields needed for simulation
        persona["recent_memory"] = f"I woke up this morning in {city}, {country}."
        persona["recent_actions"] = []
        
        # Save the persona to a file
        with open("persona.json", "w") as f:
            json.dump(persona, f, indent=2)
        
        console.print(f"[green]Created new persona: {persona['name']}, a {persona['age']}-year-old {persona['occupation']}[/green]")
        
        return persona
        
    except Exception as e:
        logger.error(f"Error creating persona: {e}")
        console.print(f"[red]Error creating persona: {e}[/red]")
        
        # Create a default persona
        default_persona = {
            "name": f"Alex Smith",
            "age": 30,
            "occupation": "Office Worker",
            "personality_traits": ["adaptable", "curious", "friendly"],
            "background": f"Born and raised in {city}, {country}. Has lived there all their life.",
            "goals": ["Find a better job", "Make new friends", "Explore the city"],
            "current_emotion": "neutral",
            "current_goal": "Get through the day",
            "recent_memory": f"I woke up this morning in {city}, {country}.",
            "recent_actions": []
        }
        
        # Save the default persona
        with open("persona.json", "w") as f:
            json.dump(default_persona, f, indent=2)
        
        console.print("[yellow]Created default persona due to error.[/yellow]")
        
        return default_persona

async def run_simulation(
    persona_file: str,
    world_config: Dict,
    reaction_profile: str = "balanced",
    max_steps: int = 10,
    new_simulation: bool = False,
    time_increment: int = 15
) -> None:
    """Run the simulation with the specified parameters."""
    # Initialize console for output
    console = Console()
    
    # Determine profile name for display
    profile_display = {
        "balanced": "Balanced",
        "optimistic": "Optimistic",
        "pessimistic": "Pessimistic",
        "creative": "Creative",
        "analytical": "Analytical"
    }.get(reaction_profile, "Custom")
    
    # Initialize the world engine
    world_engine = WorldEngine(
        persona_file=persona_file,
        world_config=world_config,
        reaction_profile=reaction_profile
    )
    
    # Initialize or load the world state
    if new_simulation:
        console.print("[yellow]Initializing new world with real-time data...[/yellow]")
        await world_engine.initialize_world()
    else:
        console.print("[yellow]Attempting to load existing state...[/yellow]")
        state_loaded = await world_engine.load_state()
        if not state_loaded:
            console.print("[yellow]No existing state found or error loading. Initializing new world...[/yellow]")
            await world_engine.initialize_world()
    
    # Print world state information
    console.print("\n[bold cyan]===== WORLD STATE =====[/bold cyan]")
    console.print(f"[cyan]Location:[/cyan] {world_engine.world_state.get('city', 'Unknown')}, {world_engine.world_state.get('country', 'Unknown')}")
    console.print(f"[cyan]Date/Time:[/cyan] {world_engine.world_state.get('current_date', 'Unknown')} ({world_engine.world_state.get('day_of_week', 'Unknown')}), {world_engine.world_state.get('current_time', 'Unknown')}")
    
    # Print current weather
    weather = world_engine.world_state.get('weather', {})
    if weather:
        console.print(f"[cyan]Current Weather:[/cyan] {weather.get('weather_condition', 'Unknown')}, {weather.get('temperature', 'Unknown')}")
        if 'data_source' in weather:
            console.print(f"[dim](Weather data source: {weather.get('data_source', 'Unknown')})[/dim]")
    
    # Print current news
    news = world_engine.world_state.get('news', [])
    if news:
        console.print("\n[bold cyan]CURRENT NEWS[/bold cyan]")
        for i, item in enumerate(news[:3], 1):  # Show top 3 news items
            console.print(f"[cyan]{i}.[/cyan] {item}")
        if 'news_data_source' in world_engine.world_state:
            console.print(f"[dim](News data source: {world_engine.world_state.get('news_data_source', 'Unknown')})[/dim]")
    
    # Print local events
    events = world_engine.world_state.get('local_events', [])
    if events:
        console.print("\n[bold cyan]LOCAL EVENTS[/bold cyan]")
        for i, event in enumerate(events[:3], 1):  # Show top 3 events
            console.print(f"[cyan]{i}.[/cyan] {event}")
        if 'events_data_source' in world_engine.world_state:
            console.print(f"[dim](Events data source: {world_engine.world_state.get('events_data_source', 'Unknown')})[/dim]")
    
    # Print reaction profile
    console.print(f"\n[bold magenta]Reaction Profile:[/bold magenta] {profile_display}")
    
    # Print character information
    persona = world_engine.simulacra.persona
    console.print("\n[bold green]===== CHARACTER =====[/bold green]")
    console.print(f"[green]Name:[/green] {persona.get('name', 'Unknown')}")
    console.print(f"[green]Age:[/green] {persona.get('age', 'Unknown')}")
    console.print(f"[green]Occupation:[/green] {persona.get('occupation', 'Unknown')}")
    console.print(f"[green]Current Goal:[/green] {persona.get('current_goal', 'Unknown')}")
    console.print(f"[green]Current Emotion:[/green] {persona.get('current_emotion', 'Unknown')}")
    
    # Print current location information
    console.print("\n[bold blue]===== CURRENT LOCATION =====[/bold blue]")
    location = world_engine.immediate_environment
    console.print(f"[blue]Name:[/blue] {location.get('current_location_name', 'Unknown')}")
    console.print(f"[blue]Type:[/blue] {location.get('location_type', 'Unknown')} ({location.get('indoor_outdoor', 'Unknown')})")
    console.print(f"[blue]Description:[/blue] {location.get('description', 'No description available')}")
    
    if 'address' in location:
        console.print(f"[blue]Address:[/blue] {location.get('address', 'Unknown')}")
    
    if 'opening_hours' in location:
        console.print(f"[blue]Opening Hours:[/blue] {location.get('opening_hours', 'Unknown')}")
    
    # Print initial narrative context
    console.print("\n[bold yellow]===== NARRATIVE =====[/bold yellow]")
    console.print(f"{world_engine.narrative_context}")
    
    # Main simulation loop
    step = 1
    while step <= max_steps:
        console.print(f"\n[bold]===== SIMULATION STEP {step}/{max_steps} =====[/bold]")
        
        # Perception phase
        console.print("\n[bold cyan]PERCEPTION[/bold cyan]")
        perception = await world_engine.process_perception()
        
        # Action phase
        console.print("\n[bold green]ACTION[/bold green]")
        action = await world_engine.simulacra.decide_action(perception)
        
        # World update phase
        console.print("\n[bold blue]WORLD UPDATE[/bold blue]")
        update_result = await world_engine.process_update(action)
        
        # Advance time
        console.print(f"\n[bold yellow]Time passes... ({time_increment} minutes)[/bold yellow]")
        world_engine.advance_time(time_increment)
        
        # Save state
        await world_engine.save_state()
        
        # Prompt user to continue
        if step < max_steps:
            choice = Prompt.ask(
                "\n[bold]Continue simulation?[/bold]",
                choices=["y", "n", "t", "q"],
                default="y"
            )
            
            if choice.lower() == "n":
                break
            elif choice.lower() == "t":
                new_increment = IntPrompt.ask(
                    "Enter new time increment (minutes)",
                    default=time_increment
                )
                time_increment = max(1, min(new_increment, 1440))  # Between 1 minute and 24 hours
            elif choice.lower() == "q":
                console.print("[yellow]Quitting simulation...[/yellow]")
                break
        
        step += 1
    
    console.print("\n[bold]===== SIMULATION COMPLETE =====[/bold]")
    console.print(f"Completed {step-1} steps of simulation.")

def create_default_world_config():
    """Create a default world configuration file if it doesn't exist."""
    if not os.path.exists("world_config.yaml"):
        default_config = {
            "city": "New York",
            "country": "United States",
            "starting_location": "Central Park",
            "use_real_data": True
        }
        
        with open("world_config.yaml", "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print("Created default world configuration file: world_config.yaml")
        return default_config
    
    with open("world_config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a character simulation with real-world data integration")
    parser.add_argument("--persona", type=str, default="persona.json", help="Path to persona file")
    parser.add_argument("--config", type=str, default="world_config.yaml", help="Path to world configuration file")
    parser.add_argument("--steps", type=int, default=10, help="Maximum number of simulation steps")
    parser.add_argument("--new", action="store_true", help="Start a new simulation")
    parser.add_argument("--city", type=str, help="Override city in configuration")
    parser.add_argument("--country", type=str, help="Override country in configuration")
    parser.add_argument("--no-real-data", action="store_true", help="Disable real-world data fetching")
    parser.add_argument("--profile", type=str, default="balanced", 
                        choices=["balanced", "optimistic", "pessimistic", "creative", "analytical"],
                        help="Reaction profile for the simulation")
    parser.add_argument("--time-increment", type=int, default=15, help="Time increment in minutes between steps")
    
    args = parser.parse_args()
    
    # Create default world config if it doesn't exist
    world_config = create_default_world_config()
    
    # Override with command line arguments if provided
    if args.city:
        world_config["city"] = args.city
    
    if args.country:
        world_config["country"] = args.country
    
    if args.no_real_data:
        world_config["use_real_data"] = False
    
    # Check if persona file exists, create if it doesn't
    if not os.path.exists(args.persona):
        asyncio.run(create_persona(world_config['location']["city"], world_config['location']["country"]))
    
    # Run the simulation
    asyncio.run(run_simulation(
        persona_file=args.persona,
        world_config=world_config,
        reaction_profile=args.profile,
        max_steps=args.steps,
        new_simulation=args.new,
        time_increment=args.time_increment
    ))