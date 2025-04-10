import json
import logging
import os
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

# Import from other modules in src
from src.models import DayResponse, EmotionAnalysisResponse, ActionDecisionResponse
from src.prompt_manager import PromptManager
from src.llm_service import LLMService


logger = logging.getLogger(__name__)

class Simulacra:
    """Represents a simulated human with personality, goals, and behaviors."""

    def __init__(self, persona_path: Optional[str] = None, console: Console = None):
        # Initialize console
        self.console = console or Console()
        self.state_path = "simulacra_state.json" # Use this for loading/saving
        self.persona = None
        self.history = []

        # Try loading existing state first
        if persona_path and os.path.exists(persona_path):
             self.state_path = persona_path # Use provided path if it exists
             self.load_state() # Load both persona and history

        # If loading failed or no path provided, use default persona
        if self.persona is None:
            logger.info("No valid persona loaded. Initializing with default persona.")
            self.persona = self._get_default_persona()
            self.history = [] # Ensure history is empty for default persona

        self.llm_service = LLMService()


    def _get_default_persona(self) -> Dict:
         """Returns the default persona dictionary."""
         return {
            "name": "Alex Chen",
            "age": 34,
            "occupation": "Software Engineer",
            "personality_traits": ["analytical", "introverted", "creative", "detail-oriented"],
            "goals": ["complete work project before deadline", "find better work-life balance"],
            "current_state": {
                "physical": "Slightly tired, had coffee 1 hour ago",
                "emotional": "Mildly stressed about project deadline",
                "mental": "Focused but distracted occasionally"
            },
            "memory": {
                "short_term": ["Meeting with team this morning", "Email from boss about deadline"],
                "long_term": ["Computer Science degree", "5 years at current company"]
            }
        }


    async def _reflect_on_situation(self, observations: str, immediate_environment: Dict, persona_state: Optional[Dict] = None) -> str:
        """Reflect on the current situation based on observations using Gemini API."""
        if persona_state is None:
            persona_state = self.persona

        # Ensure persona_state is valid before proceeding
        if not persona_state or not isinstance(persona_state, dict):
             logger.error("Invalid persona state provided for reflection.")
             return "Internal error: Invalid persona state."


        # Include immediate environment in the reflection
        enhanced_observations = f"""
        Observations:
        {observations}

        Your immediate environment:
        {json.dumps(immediate_environment, indent=2)}
        """

        prompt = PromptManager.reflect_on_situation_prompt(enhanced_observations, persona_state)
        system_instruction = 'Reflect on the current situation based on these observations and your environment. Respond ONLY with the JSON structure specified.'

        try:
            reflection_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=DayResponse # Expecting DayResponse structure
            )
            # Check for errors from LLMService
            if isinstance(reflection_response, dict) and "error" in reflection_response:
                 logger.error(f"LLM Error during reflection: {reflection_response['error']}. Raw: {reflection_response.get('raw_response')}")
                 return "I'm having trouble processing my thoughts right now due to an internal error."
            # Validate structure before accessing 'reflect'
            if isinstance(reflection_response, dict) and 'reflect' in reflection_response:
                return reflection_response.get('reflect', "I'm trying to understand what's happening around me.")
            else:
                 logger.warning(f"Unexpected response format during reflection: {reflection_response}")
                 return "I'm processing the situation..."

        except Exception as e:
            logger.error(f"Error in reflection: {e}", exc_info=True)
            return "I'm trying to understand what's happening around me."

    async def _analyze_emotions(self, situation: str, current_emotional_state: str) -> Dict:
        """Analyze emotional response to a situation using Gemini API."""
        prompt = PromptManager.analyze_emotions_prompt(situation, current_emotional_state)
        system_instruction = "Analyze the emotional tone of the following situation and the character's current emotional state. Respond ONLY with the JSON structure specified."

        default_emotion = {
                "primary_emotion": "confused",
                "intensity": "Medium",
                "secondary_emotion": "uncertain",
                "emotional_update": "I'm feeling a bit confused by what's happening."
            }

        try:
            emotion_analysis_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=EmotionAnalysisResponse # Expecting EmotionAnalysisResponse structure
            )

            # Check for errors from LLMService
            if isinstance(emotion_analysis_response, dict) and "error" in emotion_analysis_response:
                 logger.error(f"LLM Error during emotion analysis: {emotion_analysis_response['error']}. Raw: {emotion_analysis_response.get('raw_response')}")
                 return default_emotion
            # Validate structure before returning
            if isinstance(emotion_analysis_response, dict) and all(k in emotion_analysis_response for k in default_emotion.keys()):
                 return emotion_analysis_response
            else:
                 logger.warning(f"Unexpected response format during emotion analysis: {emotion_analysis_response}")
                 return default_emotion

        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}", exc_info=True)
            return default_emotion


    async def _decide_action(self, reflection: str, emotional_analysis: Dict, goals: List[str], immediate_environment: Dict) -> Dict:
        """Decide on an action based on reflection, emotional analysis, and immediate environment."""
        prompt = PromptManager.decide_action_prompt(reflection, emotional_analysis, goals, immediate_environment)
        system_instruction = 'Decide on an action to take based on your reflection, emotional analysis, and environment. Respond ONLY with the JSON structure specified.'

        default_action = {
                "thought_process": "I need to take a moment to think about my next steps.",
                "action": "Pause and consider options",
                "action_details": {"manner": "thoughtful", "duration": "brief"}
            }

        try:
            action_decision_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=ActionDecisionResponse # Expecting ActionDecisionResponse structure
            )
            # Check for errors from LLMService
            if isinstance(action_decision_response, dict) and "error" in action_decision_response:
                 logger.error(f"LLM Error during action decision: {action_decision_response['error']}. Raw: {action_decision_response.get('raw_response')}")
                 return default_action
            # Validate structure before returning
            if isinstance(action_decision_response, dict) and all(k in action_decision_response for k in default_action.keys()):
                # Ensure action_details is a dict, default to empty if not present or not dict
                if 'action_details' not in action_decision_response or not isinstance(action_decision_response['action_details'], dict):
                    action_decision_response['action_details'] = {}
                return action_decision_response
            else:
                 logger.warning(f"Unexpected response format during action decision: {action_decision_response}")
                 return default_action

        except Exception as e:
            logger.error(f"Error deciding action: {e}", exc_info=True)
            return default_action

    def save_state(self):
        """Save current simulacra state (persona and history) to a file."""
        try:
            # Ensure persona is not None before saving
            if self.persona is None:
                 logger.error("Attempted to save simulacra state, but persona is None.")
                 return

            with open(self.state_path, 'w') as file:
                json.dump({
                    "persona": self.persona,
                    "history": self.history
                }, file, indent=2)
            logger.info(f"Successfully saved simulacra state to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving simulacra state to {self.state_path}: {e}")


    def load_state(self):
        """Load simulacra state (persona and history) from a file."""
        try:
            with open(self.state_path, 'r') as file:
                data = json.load(file)
                # Basic validation
                if "persona" in data and isinstance(data["persona"], dict):
                     self.persona = data["persona"]
                     logger.info(f"Loaded persona from {self.state_path}")
                else:
                     logger.warning(f"Persona not found or invalid in {self.state_path}. Will use default.")
                     self.persona = self._get_default_persona() # Fallback

                if "history" in data and isinstance(data["history"], list):
                    self.history = data["history"]
                    logger.info(f"Loaded history ({len(self.history)} items) from {self.state_path}")
                else:
                     logger.warning(f"History not found or invalid in {self.state_path}. Initializing empty history.")
                     self.history = [] # Fallback

        except FileNotFoundError:
            logger.warning(f"Simulacra state file not found at {self.state_path}. Will initialize with defaults.")
            self.persona = self._get_default_persona()
            self.history = []
        except json.JSONDecodeError:
            logger.error(f"Corrupted simulacra state file at {self.state_path}. Will initialize with defaults.")
            self.persona = self._get_default_persona()
            self.history = []
        except Exception as e:
            logger.error(f"Error loading simulacra state from {self.state_path}: {e}. Will initialize with defaults.")
            self.persona = self._get_default_persona()
            self.history = []


    async def process_perception(self, world_update: Dict[str, Any]) -> Dict[str, Any]:
        """Process perceptions from the world and decide on actions."""

        # Ensure persona is loaded before processing
        if not self.persona:
             logger.error("Simulacra cannot process perception: Persona not loaded.")
             # Return a default 'error' action or state
             return {
                  "thought_process": "Internal error: Persona not available.",
                  "emotional_update": "Confused due to internal error.",
                  "action": "Wait",
                  "action_details": {},
                  "updated_state": {}
             }


        # Extract the relevant parts from the world update
        world_state = world_update.get("world_state", {})
        immediate_environment = world_update.get("immediate_environment", {})
        observations = world_update.get("observations", [])

        # Print simulacra's current state before processing
        self.console.print(Panel(self.get_simulacra_summary(),
                            title="[bold cyan]SIMULACRA CURRENT STATE[/bold cyan]",
                            border_style="cyan"))

        # Save the perception to history
        self.history.append({
            "timestamp": world_state.get("current_time", "unknown"),
            "date": world_state.get("current_date", "unknown"),
            "perception": observations
        })

        # Ensure observations is a string for the prompts
        observations_str = json.dumps(observations) if isinstance(observations, list) else str(observations)
        self.console.print(f"\n[bold yellow]Processing observations:[/bold yellow]", observations_str)

        # --- Perception Processing Pipeline ---
        self.console.print("\n[bold green]Reflecting on situation...[/bold green]")
        reflection_text = await self._reflect_on_situation(observations_str, immediate_environment)
        self.console.print(f"[italic green]Reflection:[/italic green] {reflection_text}\n")

        self.console.print("[bold blue]Analyzing emotions...[/bold blue]")
        # Ensure current emotional state is passed correctly
        current_emotional = self.persona.get("current_state", {}).get("emotional", "Neutral")
        emotional_analysis = await self._analyze_emotions(reflection_text, current_emotional)
        self.console.print(f"[italic blue]Emotional analysis:[/italic blue] {json.dumps(emotional_analysis, indent=2)}\n")

        self.console.print("[bold magenta]Deciding on action...[/bold magenta]")
        # Ensure goals are passed correctly
        current_goals = self.persona.get("goals", [])
        action_decision = await self._decide_action(
            reflection_text,
            emotional_analysis,
            current_goals,
            immediate_environment
        )
        self.console.print(f"[italic magenta]Action decision:[/italic magenta] {json.dumps(action_decision, indent=2)}\n")
        # --- End Pipeline ---


        # Update the persona's emotional state based on the analysis
        if "emotional_update" in emotional_analysis and isinstance(emotional_analysis["emotional_update"], str):
             if "current_state" not in self.persona: self.persona["current_state"] = {} # Ensure structure exists
             previous_emotional = self.persona["current_state"].get("emotional", "Unknown")
             self.persona["current_state"]["emotional"] = emotional_analysis["emotional_update"]
             self.console.print(f"[bold orange3]Emotional state updated:[/bold orange3] {previous_emotional} → {self.persona['current_state']['emotional']}")
        else:
             logger.warning("Emotional update missing or invalid in analysis.")


        # Prepare the response structure
        simulacra_response = {
            "thought_process": action_decision.get("thought_process", "Processing observations..."),
            "emotional_update": emotional_analysis.get("emotional_update", "Maintaining emotional state."),
            "action": action_decision.get("action", "Observe further."),
            "action_details": action_decision.get("action_details", {}),
             # Ensure current_state is returned, default to empty if not available
            "updated_state": self.persona.get("current_state", {})
        }

        # Print final response
        response_panel_content = f"[bold]Thought process:[/bold] {simulacra_response['thought_process']}\n\n"
        response_panel_content += f"[bold]Emotional update:[/bold] {simulacra_response['emotional_update']}\n\n"
        response_panel_content += f"[bold]Action:[/bold] {simulacra_response['action']}"

        # Safely access action_details
        action_details = simulacra_response.get("action_details")
        if action_details and isinstance(action_details, dict):
             # Add details only if they exist and are not empty
             details_str = json.dumps(action_details, indent=2)
             if details_str != '{}':
                  response_panel_content += f"\n\n[bold]Action details:[/bold] {details_str}"


        self.console.print(Panel(response_panel_content,
                            title="[bold red]SIMULACRA RESPONSE[/bold red]",
                            border_style="red"))

        # Save state after processing perception
        self.save_state()

        return simulacra_response

    def get_simulacra_summary(self) -> str:
        """Generate a concise summary of the simulacra's current state."""
        if not self.persona:
            return "[bold red]Simulacra persona not loaded.[/bold red]"

        name = self.persona.get("name", "Unknown")
        age = self.persona.get("age", "Unknown")
        occupation = self.persona.get("occupation", "Unknown")

        # Get current state information safely
        current_state = self.persona.get("current_state", {})
        physical = current_state.get("physical", "Unknown physical state")
        emotional = current_state.get("emotional", "Unknown emotional state")
        mental = current_state.get("mental", "Unknown mental state")

        # Get goals safely
        goals = self.persona.get("goals", [])
        goals_str = ", ".join(goals) if goals else "No specific goals"

        # Get recent memories safely
        memory = self.persona.get("memory", {})
        short_term = memory.get("short_term", [])
        recent_memories = ", ".join(short_term[:2]) if short_term else "No recent memories"

        # Create summary with rich formatting
        summary = f"[bold blue]Simulacra:[/bold blue] {name}, {age}, {occupation}\n"
        summary += f"[bold green]Physical state:[/bold green] {physical}\n"
        summary += f"[bold yellow]Emotional state:[/bold yellow] {emotional}\n"
        summary += f"[bold magenta]Mental state:[/bold magenta] {mental}\n"
        summary += f"[bold cyan]Current goals:[/bold cyan] {goals_str}\n"
        summary += f"[bold orange3]Recent memories:[/bold orange3] {recent_memories}"

        return summary
