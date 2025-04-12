# src/npc.py
import json
import logging
from typing import Any, Dict, List, Optional

from rich.console import Console

# <<< ADD IMPORTS NEEDED FOR DECISION MAKING >>>
from src.prompt_manager import PromptManager
from src.llm_service import LLMService # Assuming NPC needs its own or accesses WorldEngine's
from src.models import ActionDecisionResponse, AllowedActionVerbs # To validate response
from src.utils.llm_utils import generate_and_validate_llm_response
# <<< END ADDED IMPORTS >>>

logger = logging.getLogger(__name__)

class NPC:
    """Represents a Non-Player Character in the simulation."""

    def __init__(self, name: str, initial_context: Dict, console: Console):
        """Initialize the NPC."""
        self.name = name
        self.persona: Dict[str, Any] = {"name": name} # Basic persona, will be updated
        self.history: List[Dict] = []
        self.console = console
        self.llm_service = LLMService() # Give NPC its own LLM service instance for now
        self.current_environment: Dict = initial_context.get("immediate_environment", {})
        self.current_world_state: Dict = initial_context.get("world_state", {})
        # <<< ADDED: State needed for decision prompt >>>
        self.last_observation: Optional[str] = "Initially present." # Store last key observation
        self.last_action: Optional[str] = None # Track NPC's last action
        self.current_emotional_state: str = "Neutral" # Simple emotion tracking for now
        # ---

        logger.info(f"NPC '{self.name}' initialized.")

    def update_persona(self, persona_data: Dict):
        """Updates the NPC's persona."""
        self.persona.update(persona_data)
        # Ensure basic structure
        self.persona.setdefault("personality_traits", ["Neutral"])
        self.persona.setdefault("goals", ["Be present"])
        self.persona.setdefault("current_state", {"emotional": "Neutral", "physical": "Normal"})
        self.current_emotional_state = self.persona["current_state"].get("emotional", "Neutral")
        logger.info(f"NPC '{self.name}' persona updated.")

    async def process_perception(self, perception_data: Dict):
        """Processes observations and updates internal state."""
        # Simplified: Just log history and store last observation string
        self.current_environment = perception_data.get("immediate_environment", self.current_environment)
        self.current_world_state = perception_data.get("world_state", self.current_world_state)
        observations = perception_data.get("observations", [])
        consequences = perception_data.get("consequences", [])

        timestamp = f"{self.current_world_state.get('current_date','?')}_{self.current_world_state.get('current_time','?')}"
        perception_log = {"timestamp": timestamp, "type": "npc_perception", "data": {"observations": observations, "consequences": consequences}}
        self.history.append(perception_log)

        # Store a simple string summary of the last observation for reflection context
        if observations:
            last_obs_item = observations[-1]
            if isinstance(last_obs_item, dict) and last_obs_item.get("type") == "dialogue":
                self.last_observation = f"Heard {last_obs_item.get('from', '?')} say: '{last_obs_item.get('utterance', '...')}'"
            elif isinstance(last_obs_item, str):
                self.last_observation = last_obs_item
            else:
                self.last_observation = "Observed something unusual."
        else:
            self.last_observation = "Observed the environment."

        logger.debug(f"NPC '{self.name}' processed perception. Last observation: {self.last_observation}")
        # In a more complex NPC, this could involve LLM calls for emotion analysis, etc.

    # <<< MODIFIED: decide_action >>>
    async def decide_action(self, step_duration_minutes: int = 1) -> Optional[Dict[str, Any]]:
        """
        Decides the NPC's next action based on its current state and recent perception.
        Uses the standard PromptManager.decide_action_prompt.
        """
        logger.info(f"NPC '{self.name}' deciding action...")

        # Basic reflection/emotion context based on last observation
        # (Could be enhanced with LLM calls like Simulacra later)
        reflection = f"Considering the last observation: {self.last_observation}"
        emotional_analysis = {"emotional_update": self.current_emotional_state} # Simple placeholder

        # Prepare context for the standard decision prompt
        goals = self.persona.get("goals", ["Be present"])
        persona_state = self.persona # Pass the whole persona dict

        # Call the *correct* prompt manager method
        prompt = PromptManager.decide_action_prompt(
            reflection=reflection,
            emotional_analysis=emotional_analysis,
            goals=goals,
            immediate_environment=self.current_environment,
            persona_state=persona_state,
            retrieved_background="[NPC has no long-term memory access]", # Placeholder
            step_duration_minutes=step_duration_minutes, # Pass duration
            last_action_taken=self.last_action # Pass NPC's own last action
        )

        # Generate and validate the response
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=ActionDecisionResponse,
            system_instruction=f"You are {self.name}. Decide your next action based on the context. Respond ONLY with ActionDecisionResponse JSON.",
            operation_description=f"NPC {self.name} Action Decision"
        )

        if response_dict and "error" not in response_dict:
            action = response_dict.get("action")
            details = response_dict.get("action_details")

            # Format last_action string for the next cycle
            details_str = ""
            if isinstance(details, dict):
                details_parts = [f"{k}={v}" for k, v in details.items() if v is not None]
                if details_parts: details_str = f" ({', '.join(details_parts)})"
            elif isinstance(details, str): details_str = f" ({details})"
            self.last_action = f"{action}{details_str}" # Update NPC's last action tracker

            logger.info(f"NPC '{self.name}' decided action: {self.last_action}")
            # Return the standard action format expected by WorldEngine
            return {"action": action, "action_details": details}
        else:
            logger.error(f"NPC '{self.name}' failed to decide action: {response_dict}")
            self.last_action = "wait (decision failed)"
            return {"action": "wait", "action_details": {"reason": "Failed to decide action"}}

    def get_state(self) -> Dict:
        """Returns the current state of the NPC."""
        return {
            "name": self.name,
            "persona": self.persona,
            "history": self.history,
            "current_environment": self.current_environment, # Include env for context
            "last_observation": self.last_observation,
            "last_action": self.last_action
        }
