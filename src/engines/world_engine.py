# src/engines/world_engine.py
import asyncio
import base64
import datetime as dt
import json
import logging
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from google import genai
from google.genai import types
# import requests # Not used directly
import yaml  # Keep for potentially loading reaction profile dicts? Maybe not needed.
from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image  # Assumes Pillow is installed
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel # Import Panel

# --- Import from other modules in src ---
try:
    from src.llm_service import LLMService
    from src.models import \
        WorldProcessUpdateResponse, WorldStateChanges # Import necessary models
    from src.models import (ImmediateEnvironment, WorldReactionProfile,
                            WorldState, NPCEnvironmentStatus, ActionDecisionResponse) # Added NPCEnvironmentStatus and ActionDecisionResponse
    from src.npc import NPC  # Need the NPC class
    from src.prompt_manager import PromptManager
    # --- NEW IMPORTS ---
    from src.simulacra import Simulacra  # Need type hint
    from src.utils.llm_utils import generate_and_validate_llm_response
    from src.utils.world_utils import get_day_phase # Assuming this exists


    # --- END NEW IMPORTS ---
except ImportError as e:
     print(f"CRITICAL Error importing modules in world_engine.py: {e}")
     raise

logger = logging.getLogger(__name__)

M = TypeVar('M', bound=BaseModel)

# --- Default Structure (Keep as is) ---
DEFAULT_SIMULACRA_STATE = {
    "configuration": {
        "location": {"city": "NYC", "country": "USA", "region": "Northeast"},
    },
    "world_engine_state": {
        "world_state": None,
        "immediate_environment": None,
        "run_history": [],
        "reaction_profile": WorldReactionProfile().model_dump(),
        # <<< ADDED: Default for Day Arc >>>
        "current_day_arc": None
    },
    "simulacra_state": {
        "persona": None,
        "history": []
    }
}


class WorldEngine:
    """Manages the environment, physical laws, and agent interactions of the simulated world."""

    def __init__(self,
                 state_file_path: str,
                 console: Console,
                 reaction_profile: WorldReactionProfile,
                 llm_service: LLMService,
                 search_tool: Optional[DuckDuckGoSearchRun],
                 initial_config: Dict):
        """Basic synchronous initialization."""
        self.console = console
        self.state_path = state_file_path
        self.life_summary_path: Optional[str] = None
        self.run_history: List[Dict] = []
        self.world_state: Optional[Dict] = None
        self.immediate_environment: Optional[Dict] = None
        self.initial_narrative_context: Optional[str] = None
        self.llm_service = llm_service
        self.search_tool = search_tool
        self.reaction_profile = reaction_profile
        self.config = initial_config
        self.agents: Dict[str, Union[Simulacra, NPC]] = {}
        # Image settings from config
        image_settings = initial_config.get("image_settings", {})
        self.image_save_dir = Path(image_settings.get("save_directory", "images"))
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
        self.gemini_image_model_name = image_settings.get("model_name", "gemini-pro-vision") # Or your preferred model
        self.image_generation_enabled = image_settings.get("enabled", True)
        self.image_generation_frequency = image_settings.get("frequency", 1)

        # Cycle number for tracking simulation steps
        self.cycle_num = 0 # Initialize cycle number

        # <<< ADDED: Day Arc Attribute >>>
        self.current_day_arc: Optional[str] = None

    @classmethod
    async def create(cls,
                     state_file_path: str = "simulacra_state.json",
                     load_state: bool = True,
                     life_summary_path: Optional[str] = None,
                     console: Optional[Console] = None,
                     reaction_profile_arg: Union[str, Dict, WorldReactionProfile] = "balanced") -> "WorldEngine":
        """Asynchronously creates and initializes a WorldEngine instance."""
        console = console or Console()
        logger.info(f"Starting async WorldEngine creation from state file: {state_file_path}...")

        # Load Existing Combined State File
        full_state_data = {}
        if load_state and os.path.exists(state_file_path):
            try:
                with open(state_file_path, 'r', encoding='utf-8') as f:
                    full_state_data = json.load(f)
                logger.info(f"Successfully loaded combined state file: {state_file_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load or parse state file '{state_file_path}': {e}. Starting fresh.", exc_info=True)
                console.print(f"[red]Error loading state file '{state_file_path}'. Creating new state.[/red]")
                full_state_data = DEFAULT_SIMULACRA_STATE.copy()
        else:
            logger.warning(f"State file '{state_file_path}' not found or load_state=False. Creating new state.")
            console.print(f"[yellow]State file '{state_file_path}' not found or loading disabled. Creating new state.[/yellow]")
            full_state_data = DEFAULT_SIMULACRA_STATE.copy()

        # Ensure basic structure exists
        full_state_data.setdefault("configuration", DEFAULT_SIMULACRA_STATE["configuration"])
        full_state_data.setdefault("world_engine_state", DEFAULT_SIMULACRA_STATE["world_engine_state"])
        full_state_data.setdefault("simulacra_state", DEFAULT_SIMULACRA_STATE["simulacra_state"])

        current_config = full_state_data["configuration"]
        world_engine_saved_state = full_state_data["world_engine_state"]

        # Initialize Services
        llm_service = None
        try:
            llm_service = LLMService() # Assumes LLMService initializes API keys internally
            logger.info("LLMService created for WorldEngine.")
        except Exception as e:
            logger.critical(f"Failed to create LLMService for WorldEngine: {e}", exc_info=True)
            console.print(f"[bold red]CRITICAL Error: Failed create LLM Service: {e}.[/bold red]")
            raise RuntimeError("Cannot create WorldEngine without LLMService") from e

        search_tool = None
        try:
            search_tool = DuckDuckGoSearchRun() # Assumes API keys/setup handled internally
            logger.info("Search Tool created for WorldEngine.")
        except Exception as e:
            logger.warning(f"Failed to create DuckDuckGoSearchRun: {e}. News search might fail.", exc_info=True)

        # Resolve Reaction Profile
        profile_obj = None
        resolved_from_arg = False
        if reaction_profile_arg != "balanced": # Check if user provided a specific profile
            if isinstance(reaction_profile_arg, str):
                profile_obj = WorldReactionProfile.create_profile(reaction_profile_arg)
                resolved_from_arg = True
            elif isinstance(reaction_profile_arg, dict):
                try:
                    profile_obj = WorldReactionProfile.model_validate(reaction_profile_arg)
                    resolved_from_arg = True
                except ValidationError as e:
                    logger.warning(f"Invalid reaction profile dict from args: {e}. Trying loaded state.")
            elif isinstance(reaction_profile_arg, WorldReactionProfile):
                profile_obj = reaction_profile_arg
                resolved_from_arg = True
            else:
                logger.warning(f"Invalid reaction profile arg type: {type(reaction_profile_arg)}. Trying loaded state.")

        if not profile_obj and load_state and isinstance(world_engine_saved_state.get("reaction_profile"), dict):
             try:
                 profile_obj = WorldReactionProfile(**world_engine_saved_state["reaction_profile"])
                 logger.info("Loaded reaction profile from saved state.")
             except (ValidationError, TypeError) as e:
                 logger.warning(f"Invalid reaction profile in saved state: {e}. Using default.")

        if not profile_obj:
            profile_obj = WorldReactionProfile.create_profile("balanced")
            if resolved_from_arg:
                logger.warning("Provided reaction profile was invalid, reverted to balanced.")
            else:
                logger.info("Using default 'balanced' reaction profile.")
        logger.info(f"Final Reaction Profile: {profile_obj.model_dump_json(indent=2)}")


        # Create the basic synchronous instance
        engine_instance = cls(state_file_path=state_file_path, console=console, reaction_profile=profile_obj, llm_service=llm_service, search_tool=search_tool, initial_config=current_config)
        engine_instance.life_summary_path = life_summary_path


        # Perform Asynchronous Initialization Steps (World State & Environment)
        initialized_state = False
        if load_state and world_engine_saved_state: # Check if state exists and we should load it
            logger.info(f"Attempting to load existing world engine state from loaded data...")
            loaded_ws = world_engine_saved_state.get("world_state")
            loaded_env = world_engine_saved_state.get("immediate_environment")
            loaded_hist = world_engine_saved_state.get("run_history", [])
            loaded_cycle_num = world_engine_saved_state.get("cycle_num", 0) # Load cycle number
            # <<< ADDED: Load Day Arc >>>
            loaded_day_arc = world_engine_saved_state.get("current_day_arc")

            valid_ws = None
            valid_env = None
            if isinstance(loaded_ws, dict):
                try:
                    valid_ws = WorldState.model_validate(loaded_ws).model_dump()
                    logger.debug("Loaded world_state validated.")
                except ValidationError as e:
                    logger.warning(f"Invalid world_state in loaded data: {e}")
            if isinstance(loaded_env, dict):
                try:
                    # Validate NPC list structure first if necessary
                    if 'specific_npcs_present' in loaded_env and isinstance(loaded_env['specific_npcs_present'], list):
                         loaded_env['specific_npcs_present'] = [item for item in loaded_env['specific_npcs_present'] if isinstance(item, dict)]
                    valid_env = ImmediateEnvironment.model_validate(loaded_env).model_dump()
                    logger.debug("Loaded immediate_environment validated.")
                except ValidationError as e:
                    logger.warning(f"Invalid immediate_environment in loaded data: {e}")

            if valid_ws and valid_env:
                engine_instance.world_state = valid_ws
                engine_instance.immediate_environment = valid_env
                engine_instance.run_history = loaded_hist if isinstance(loaded_hist, list) else []
                engine_instance.cycle_num = loaded_cycle_num if isinstance(loaded_cycle_num, int) else 0 # Assign cycle number
                # <<< ADDED: Assign Loaded Day Arc >>>
                engine_instance.current_day_arc = loaded_day_arc if isinstance(loaded_day_arc, str) else None

                logger.info("Successfully loaded world engine state from state file.")
                console.print(f"[green]Loaded existing world state from [bold]{state_file_path}[/bold][/green]")
                if engine_instance.current_day_arc: # Print if loaded
                    console.print(f"[purple]Loaded Day Arc:[/purple] {engine_instance.current_day_arc}")
                initialized_state = True
            else:
                logger.warning("Existing world engine state in file was invalid or incomplete.")
                console.print(f"[yellow]Found existing state in {state_file_path}, but it was invalid. Attempting other initialization.[/yellow]")

        # Fallback initialization methods
        if not initialized_state and life_summary_path and os.path.exists(life_summary_path):
             logger.info(f"Attempting to initialize world state from life summary: {life_summary_path}")
             console.print(f"[cyan]Initializing world from life summary: [bold]{life_summary_path}[/bold][/cyan]")
             try:
                  await engine_instance._initialize_from_life_summary(life_summary_path)
                  if engine_instance.world_state and engine_instance.immediate_environment:
                       logger.info("Successfully initialized world state from life summary.")
                       console.print("[green]World state initialized from life summary.[/green]")
                       initialized_state = True
                       engine_instance.run_history = []
                       engine_instance.cycle_num = 0 # Reset cycle count
                       engine_instance.current_day_arc = None # No day arc on fresh init
                  else:
                       console.print("[red]Failed to initialize world state from life summary (method returned invalid state).[/red]")
             except Exception as e:
                  logger.error(f"Error during initialization from life summary '{life_summary_path}': {e}", exc_info=True)
                  console.print(f"[red]Error initializing from life summary: {e}[/red]")

        if not initialized_state:
             logger.info("No valid state loaded or initialized from summary. Initializing new world.")
             console.print("[cyan]Initializing new world based on current time and config...[/cyan]")
             try:
                  init_data = await engine_instance.initialize_new_world(engine_instance.config)
                  if engine_instance.world_state and engine_instance.immediate_environment:
                        logger.info("Successfully initialized new world.")
                        console.print("[green]New world initialized successfully.[/green]")
                        engine_instance.initial_narrative_context = init_data.get("narrative_context")
                        initialized_state = True
                        engine_instance.run_history = []
                        engine_instance.cycle_num = 0 # Reset cycle count
                        engine_instance.current_day_arc = None # No day arc on fresh init
                  else:
                        logger.error("Initialization of new world failed to set state correctly.")
                        console.print("[red]Failed to initialize new world state.[/red]")
             except Exception as e:
                  logger.error(f"Error during new world initialization: {e}", exc_info=True)
                  console.print(f"[red]Error initializing new world: {e}[/red]")

        # Ultimate fallback if all else fails
        if not initialized_state:
             logger.critical("WorldEngine failed to initialize state through any method. Using minimal defaults.")
             console.print("[bold red]CRITICAL: World Engine state could not be initialized! Using minimal defaults.[/bold red]")
             engine_instance.world_state = engine_instance._create_default_world_state(engine_instance.config)
             engine_instance.immediate_environment = await engine_instance._create_default_immediate_environment("Default Room")
             engine_instance.run_history = []
             engine_instance.cycle_num = 0
             engine_instance.current_day_arc = None
             engine_instance.initial_narrative_context = "[World Initialization Failed]"
             engine_instance.save_state() # Save the minimal default state

        logger.info("Async WorldEngine creation finished.")
        return engine_instance


    def register_agent(self, agent: Union[Simulacra, NPC]):
        """Adds a Simulacra or NPC instance to the engine's agent registry."""
        if not hasattr(agent, 'name') or not agent.name:
            logger.error("Cannot register agent: Agent lacks a 'name' attribute or name is empty.")
            return
        if agent.name in self.agents:
            logger.warning(f"Agent '{agent.name}' already registered. Overwriting.")
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: '{agent.name}' (Type: {type(agent).__name__})")
        self.console.print(f"[dim]Agent registered:[/dim] [bold magenta]{agent.name}[/bold magenta]")

    async def _generate_day_arc(self, persona: Dict, world_state: Dict) -> Optional[str]:
        """Generates a narrative arc for the current day using the LLM."""
        if not self.llm_service:
            logger.error("LLM Service not available for Day Arc generation.")
            return None
        if not persona or not world_state:
             logger.warning("Cannot generate Day Arc without persona or world state.")
             return None
        try:
            prompt = PromptManager.generate_day_arc_prompt(persona, world_state)

            # --- MODIFIED: Use generate_and_validate_llm_response ---
            # Pass response_model=None since we expect plain text (or dict with 'text' key)
            response_dict = await generate_and_validate_llm_response(
                llm_service=self.llm_service,
                prompt=prompt,
                response_model=None, # Expecting plain text or raw dict with 'text'
                # system_instruction="Generate a plausible narrative arc...", # REMOVED
                # operation_description="Day Arc Generation", # REMOVED
                # temperature=0.75 # REMOVED
            )
            # --- END MODIFICATION ---

            # Check the structure returned by generate_and_validate_llm_response
            # It likely still returns a dict, possibly with the raw 'text' or includes an 'error' key
            if response_dict and "error" not in response_dict and isinstance(response_dict.get('narrativeArc'), str):
                arc_text = response_dict['narrativeArc'].strip()
                # Basic validation - ensure it's not empty or just placeholder
                if arc_text and len(arc_text) > 10 and not arc_text.startswith("["):
                     logger.info(f"Generated Day Arc: {arc_text}")
                     return arc_text
                else:
                    logger.warning(f"Generated Day Arc seems invalid: '{arc_text}'")
                    return None
            elif response_dict and "error" in response_dict:
                 logger.error(f"LLM Error generating Day Arc: {response_dict['error']}")
                 return None
            else:
                # Handle cases where generate_and_validate might return the raw dict without 'text'
                # This might need adjustment based on generate_and_validate's exact behavior with response_model=None
                logger.warning(f"Invalid/Empty response for Day Arc generation (via helper): {response_dict}")
                return None
        except Exception as e:
            logger.error(f"Exception generating Day Arc: {e}", exc_info=True)
            return None

    async def _generate_npc_persona(self, npc_name: str, context: Dict) -> Dict:
        """Generates a basic persona for a newly created NPC."""
        location = context.get("immediate_environment", {}).get("current_location_name", "Unknown Location")
        # Basic placeholder - ideally LLM call using context
        base_persona = {
            "name": npc_name,
            "occupation": f"Person at {location}",
            "personality_traits": ["Neutral", "Observant"],
            "current_state": {"emotional": "Neutral"},
            "role": context.get("role", "Unknown Role") # Get role from environment if available
        }
        logger.info(f"Generated basic persona for NPC '{npc_name}' based on location '{location}'.")
        # Replace with actual LLM call if needed
        # prompt = PromptManager.generate_npc_persona_prompt(npc_name, context)
        # persona_dict = await generate_and_validate_llm_response(...)
        # return persona_dict or base_persona
        return base_persona


    async def _get_or_create_npc(self, npc_name: str) -> Optional[NPC]:
        """Gets an NPC from the registry or creates and registers a new one."""
        if npc_name in self.agents:
            agent = self.agents[npc_name]
            if isinstance(agent, NPC):
                return agent
            else:
                logger.error(f"Agent '{npc_name}' exists but is not an NPC (Type: {type(agent).__name__}). Cannot interact.")
                return None

        logger.info(f"NPC '{npc_name}' not found in registry. Creating new NPC instance.")
        self.console.print(f"[yellow]✨ Creating new NPC:[/yellow] [bold cyan]{npc_name}[/bold cyan]")

        # Find NPC's potential role from environment status
        npc_role = "Unknown Role"
        if self.immediate_environment:
            for npc_status in self.immediate_environment.get('specific_npcs_present', []):
                if isinstance(npc_status, dict) and npc_status.get('name') == npc_name:
                    npc_role = npc_status.get('role', 'Unknown Role')
                    break

        npc_initial_context = {
            "world_state": self.world_state,
            "immediate_environment": self.immediate_environment,
            "role": npc_role
        }
        npc_persona = await self._generate_npc_persona(npc_name, npc_initial_context)

        try:
            # Pass only necessary context to NPC constructor if needed
            new_npc = NPC(name=npc_name, initial_context=npc_initial_context, console=self.console)
            new_npc.update_persona(npc_persona) # Use update_persona method
            self.register_agent(new_npc)

            # Add/Update NPC in environment if not present or needs update
            if self.immediate_environment:
                 npcs_present = self.immediate_environment.setdefault('specific_npcs_present', [])
                 npc_entry_exists = False
                 for npc_entry in npcs_present:
                     if isinstance(npc_entry, dict) and npc_entry.get('name') == npc_name:
                          # Optionally update role/status if different
                          npc_entry['role'] = npc_persona.get('role', npc_entry.get('role', 'Person'))
                          npc_entry['status'] = npc_entry.get('status', 'Idle') # Keep existing status unless reset needed
                          npc_entry_exists = True
                          break
                 if not npc_entry_exists:
                      # Ensure we append a dictionary matching NPCEnvironmentStatus structure
                      npcs_present.append({"name": npc_name, "role": npc_persona.get('role', 'Person'), "status": "Idle"})
                      logger.info(f"Updated environment: Added '{npc_name}' to specific_npcs_present.")

            return new_npc
        except Exception as e:
            logger.error(f"Failed to create or register NPC '{npc_name}': {e}", exc_info=True)
            self.console.print(f"[red]Error creating NPC '{npc_name}'.[/red]")
            return None

    def _format_dialogue_observation(self, source_agent_name: str, target_agent_name: str, utterance: str) -> Dict:
        """Creates a structured dictionary for dialogue observations."""
        return {"type": "dialogue", "from": source_agent_name, "to": target_agent_name, "utterance": utterance}


    async def process_update(self, initiator_action: Dict[str, Any],
                           initiator_persona: Optional[Dict] = None,
                           step_duration_minutes: int = 1,
                           recent_narrative_updates: Optional[List[str]] = None,
                           initiator_reflection: Optional[str] = None,
                           initiator_thought_process: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes an action, handles NPC creation/reactions, generates narrative & image prompt,
        updates state, and returns perception data. World constraints handled via LLM prompt.
        """
        start_time = time.time() # For timing the cycle
        self.cycle_num += 1
        self.console.rule(f"[bold cyan]Cycle {self.cycle_num} - Processing Update[/bold cyan]")
        logger.info(f"--- Cycle {self.cycle_num} Start ---")
        logger.info(f"Initiator Action: {json.dumps(initiator_action, default=str)}")

        if not self.world_state or not self.immediate_environment:
            logger.error("Cannot process update: World not initialized.");
            return {"error": "World not initialized"} # Return early if world is broken

        if initiator_persona is None: initiator_persona = {}
        if recent_narrative_updates is None: recent_narrative_updates = [] # Default to empty list

        initiator_name = initiator_persona.get("name", "Unknown Agent")
        action_verb = initiator_action.get('action', 'unknown')
        action_details = initiator_action.get('action_details', {})

        self.console.print(f"\n[yellow]Processing action from '{initiator_name}' (Step: {step_duration_minutes} min):[/yellow] {action_verb} {json.dumps(action_details, default=str)}")
        logger.info(f"Processing action from '{initiator_name}' (Step: {step_duration_minutes} min): {action_verb} {action_details}")

        # Log the *attempted* action before the world reacts
        current_time_str = self.world_state.get("current_time", "?")
        current_date_str = self.world_state.get("current_date", "?")
        self.run_history.append({"timestamp": f"{current_date_str} {current_time_str}", "type": "action", "agent": initiator_name, "data": initiator_action, "cycle": self.cycle_num})

        # --- Stage 0: Update Time ---
        try:
            current_dt = dt.datetime.fromisoformat(f"{current_date_str}T{current_time_str}")
            new_dt = current_dt + dt.timedelta(minutes=step_duration_minutes)
            new_date_str = new_dt.strftime("%Y-%m-%d")
            new_time_str = new_dt.strftime("%H:%M")

            # <<< ADDED: Check for Day Change >>>
            day_changed = new_date_str != self.world_state['current_date']
            if day_changed:
                logger.info(f"Date changed from {self.world_state['current_date']} to {new_date_str}. Clearing day arc.")
                self.current_day_arc = None # Reset day arc for the new day

            self.world_state['current_date'] = new_date_str
            self.world_state['current_time'] = new_time_str
            self.world_state['day_phase'] = get_day_phase(new_dt.hour)
            logger.info(f"World time advanced to: {new_date_str} {new_time_str} ({self.world_state['day_phase']})")
        except ValueError as e:
            logger.error(f"Error parsing or updating time: {e}. Time may not advance.", exc_info=True)
            # Continue execution, but time won't advance this cycle

        # <<< ADDED: Day Arc Generation/Check >>>
        if day_changed or not self.current_day_arc: # Generate if day changed or no arc exists
             logger.info("No current day arc found or day changed, attempting to generate one.")
             self.current_day_arc = await self._generate_day_arc(initiator_persona, self.world_state)
             if self.current_day_arc:
                 self.console.print(f"[bold purple]🌅 New Day Arc:[/bold purple] {self.current_day_arc}")
             else:
                 self.console.print("[yellow]⚠️ Failed to generate day arc for today.[/yellow]")
        else:
             # Optionally print the existing arc for context if needed
             # logger.debug(f"Using existing day arc: {self.current_day_arc}")
             # self.console.print(f"[purple]Current Day Arc:[/purple] {self.current_day_arc}")
             pass


        # --- Stage 1: Initial World Reaction (Processes initiator action's immediate effects) ---
        # The prompt now includes world constraints for the LLM to enforce
        initial_prompt = PromptManager.process_update_prompt(
            world_state=self.world_state,
            immediate_environment=self.immediate_environment,
            simulacra_action=initiator_action, # Pass the original attempted action
            reaction_profile=self.reaction_profile,
            step_duration_minutes=step_duration_minutes,
            recent_narrative_updates=recent_narrative_updates,
            initiator_reflection=initiator_reflection,
            initiator_thought_process=initiator_thought_process
            # Day arc is implicitly handled via narrative prompts later
        )

        initial_world_update_raw = None
        final_consequences = []
        final_observations = []
        action_failed_constraints = False # Track if action failed

        try:
            self.console.print("[bold cyan]1. Generating world reaction (with constraint checks)...[/bold cyan]")
            initial_world_update_raw = await generate_and_validate_llm_response(
                llm_service=self.llm_service,
                prompt=initial_prompt,
                response_model=WorldProcessUpdateResponse,
                system_instruction=f"Simulate world/env reaction to action over {step_duration_minutes} min, ENFORCING constraints. Respond ONLY with specified JSON.",
                operation_description=f"World Reaction (incl. constraints) to {initiator_name}'s {action_verb[:30]} over {step_duration_minutes} min"
            )

            # --- Process the LLM's response ---
            if not initial_world_update_raw or "error" in initial_world_update_raw:
                 self.console.print("[red]World reaction generation failed.[/red]"); logger.error(f"World reaction failed: {initial_world_update_raw}"); final_consequences.append("World reaction generation failed.")
                 action_failed_constraints = True
            else:
                 # Apply environment changes (This should reflect the *actual* outcome)
                 new_env_data = initial_world_update_raw.get("updated_environment")
                 if isinstance(new_env_data, dict):
                      try:
                           # Ensure specific_npcs_present is list of dicts before full validation
                           if 'specific_npcs_present' in new_env_data and isinstance(new_env_data['specific_npcs_present'], list):
                               new_env_data['specific_npcs_present'] = [item for item in new_env_data['specific_npcs_present'] if isinstance(item, dict)]

                           validated_env = ImmediateEnvironment.model_validate(new_env_data)
                           self.immediate_environment = validated_env.model_dump()
                           logger.debug("Environment updated based on LLM reaction.")

                           # Log location change outcomes for debugging
                           original_target_location = initiator_action.get("action_details", {}).get("target_location") if initiator_action.get("action") == "move" else None
                           final_location = self.immediate_environment.get("current_location_name")
                           if original_target_location and original_target_location != final_location:
                               logger.info(f"Move action outcome: Attempted '{original_target_location}', ended at '{final_location}'. Constraint likely applied.")
                           elif original_target_location and original_target_location == final_location:
                               logger.info(f"Move action outcome: Successfully arrived at '{final_location}'.")

                      except ValidationError as e:
                           logger.error(f"Environment validation failed: {e}. Data: {new_env_data}", exc_info=True); self.console.print("[red]Environment update failed validation.")
                           # Keep old environment if update fails validation
                 else:
                      logger.warning(f"Missing/invalid 'updated_environment' in response: {new_env_data}")

                 # Apply world state changes
                 world_changes_dict = initial_world_update_raw.get("world_state_changes")
                 if isinstance(world_changes_dict, dict) and world_changes_dict:
                      temp_ws = self.world_state.copy(); changes_applied = 0; applied_change_dict = {}
                      for k, v in world_changes_dict.items():
                          if v is not None and k in temp_ws:
                              temp_ws[k] = v; changes_applied += 1; applied_change_dict[k] = v
                          elif k not in temp_ws:
                              logger.warning(f"LLM proposed invalid WorldState key '{k}' in world_state_changes.")
                      if changes_applied > 0:
                          try:
                              WorldState.model_validate(temp_ws) # Validate the merged state
                              self.world_state = temp_ws # Assign if valid
                              logger.debug(f"Initial world state updated ({changes_applied} changes applied)."); logger.info(f"Applied World State Changes: {applied_change_dict}")
                          except ValidationError as e:
                              logger.error(f"Initial WS change validation failed: {e}. Applied changes attempted: {applied_change_dict}", exc_info=True); self.console.print("[red]Initial WS changes failed validation after filtering Nones."); final_consequences.append("A ripple passes through the world, but the fundamental state remains.")

                 # Collect consequences and observations
                 final_consequences.extend(self._sanitize_string_list(initial_world_update_raw.get("consequences"), "initial_consequences"))
                 final_observations.extend(self._sanitize_string_list(initial_world_update_raw.get("observations"), "initial_observations"))
                 logger.info(f"Reaction processing: Cons={len(final_consequences)}, Obs={len(final_observations)}")

                 # Check observations for constraint failures
                 for obs in final_observations:
                     if isinstance(obs, str) and any(phrase in obs.lower() for phrase in ["closed", "locked", "cannot enter", "unable to", "failed to", "not possible", "ignores", "doesn't respond"]):
                         self.console.print(f"[orange3]Constraint likely enforced (Observation):[/orange3] {obs}")
                         action_failed_constraints = True # Mark action as failed
                         break # No need to check further observations for this

        except Exception as e:
            logger.error(f"Error during world reaction processing for '{action_verb}': {e}", exc_info=True); self.console.print(f"[red]Error processing initial world reaction: {e}[/red]"); final_consequences.append("Error processing initial world reaction.")
            action_failed_constraints = True # Assume failure on unexpected error

        # --- Stage 2: Handle Interaction / NPC Reactions ---
        self.console.print(f"\n[bold yellow]2. Processing NPC Reactions...[/bold yellow]")
        process_npc_turns = not action_failed_constraints or action_verb == "talk" # Process turns unless action failed (but allow talk)
        if not process_npc_turns:
             logger.info("Skipping NPC reactions as initiator's action likely failed due to constraints.")
             self.console.print("[dim]  -> Skipping NPC reactions due to action constraints.[/dim]")

        # Get list of active NPCs in the current environment (even if turns skipped, needed for conversation)
        present_npc_statuses = self.immediate_environment.get("specific_npcs_present", [])
        present_npc_names = [npc_status['name'] for npc_status in present_npc_statuses if isinstance(npc_status, dict) and 'name' in npc_status]
        logger.info(f"NPCs present in environment: {present_npc_names}")
        active_npcs: List[NPC] = []
        for npc_name in present_npc_names:
            npc_agent = await self._get_or_create_npc(npc_name)
            if npc_agent: active_npcs.append(npc_agent)
            else: logger.warning(f"Could not get or create NPC agent: {npc_name}")

        # --- Conversation Handling ---
        conversation_active = False
        conversation_target_npc: Optional[NPC] = None
        conversation_history: List = [] # Stores {'speaker': name, 'utterance': text}
        MAX_EXCHANGES = 5  # Limit conversation length
        exchange_count = 0
        target_npc_name = None
        initiator_utterance = None

        if isinstance(action_details, dict):
             target_npc_name = action_details.get("target")
             initiator_utterance = action_details.get("utterance")

        # Check if the *original* action was 'talk' and the target is present
        if action_verb == "talk" and target_npc_name and initiator_utterance:
             # Find the target NPC object among active NPCs
             target_npc_obj = next((npc for npc in active_npcs if npc.name == target_npc_name), None)
             if target_npc_obj:
                  # Check if Stage 1 observations indicate the talk failed
                  talk_failed_in_stage1 = False
                  for obs in final_observations:
                      if isinstance(obs, str) and target_npc_name in obs and any(p in obs.lower() for p in ["ignores", "doesn't respond", "walks away"]):
                          talk_failed_in_stage1 = True
                          logger.info(f"Talk action to '{target_npc_name}' seems to have failed based on Stage 1 observation: '{obs}'. Skipping conversation loop.")
                          break

                  if not talk_failed_in_stage1:
                       conversation_active = True
                       conversation_target_npc = target_npc_obj
                       conversation_history.append({"speaker": initiator_name, "utterance": initiator_utterance})
                       logger.info(f"Conversation initiated between {initiator_name} and {target_npc_name}.")
                       print(f"DEBUG: Starting conversation between {initiator_name} and {target_npc_name}")
                       self.console.print(f"💬 [bold]{initiator_name}[/bold] starts conversation with [bold]{target_npc_name}[/bold]: \"{initiator_utterance}\"")
                       # Update NPC status in the environment
                       if self.immediate_environment and 'specific_npcs_present' in self.immediate_environment:
                            self.immediate_environment['specific_npcs_present'] = [
                                {**npc_state, 'status': f"Talking to {initiator_name}"} if isinstance(npc_state, dict) and npc_state.get('name') == target_npc_name else npc_state
                                for npc_state in self.immediate_environment['specific_npcs_present']
                            ]
                            print(f"DEBUG: Set {target_npc_name} status to 'Talking'")
                  else:
                       self.console.print(f"[yellow] {target_npc_name} seems unresponsive or ignored the attempt to talk.[/yellow]")
             else:
                 logger.warning(f"Initiator tried to talk to NPC '{target_npc_name}', but they are not present or couldn't be created.")
                 self.console.print(f"[yellow] {target_npc_name} is not here to talk to.[/yellow]")
                 final_observations.append(f"{initiator_name} tried to talk to {target_npc_name}, but they weren't around.")


        # --- Shared Perception Data for NPCs ---
        npc_perception_data = {
            "world_state": self.world_state,
            "immediate_environment": self.immediate_environment, # Use potentially updated environment
            "observations": final_observations, # Include observations from Stage 1
            "consequences": final_consequences, # Include consequences from Stage 1
            "initiator": {"name": initiator_name, "action": initiator_action}
        }
        npc_actions: List[Dict] = [] # Store actions taken by NPCs this cycle

        # --- Conversation Loop ---
        try:
            while conversation_active and conversation_target_npc and exchange_count < MAX_EXCHANGES:
                exchange_count += 1
                logger.info(f"Conversation Exchange {exchange_count} (Max {MAX_EXCHANGES}) - Target: {conversation_target_npc.name}")
                print(f"DEBUG: Conversation exchange #{exchange_count}")

                # --- NPC's Turn ---
                self.console.print(f"   -> [bold]{conversation_target_npc.name}'s[/bold] turn...")
                last_utterance_info = conversation_history[-1] if conversation_history else {}
                dialogue_obs = self._format_dialogue_observation(
                    last_utterance_info.get('speaker', initiator_name), # Assume initiator spoke last if history empty (shouldn't happen here)
                    conversation_target_npc.name,
                    last_utterance_info.get('utterance', initiator_utterance or "...") # Use initial utterance if first exchange
                )
                # Update NPC perception with the latest dialogue
                npc_perception_data_conv = {**npc_perception_data, "observations": npc_perception_data["observations"] + [dialogue_obs]}
                print(f"DEBUG: NPC {conversation_target_npc.name} processing perception. Last said: '{dialogue_obs['utterance'][:50]}...'")
                await conversation_target_npc.process_perception(npc_perception_data_conv)

                # NPC decides action
                npc_response_action = await conversation_target_npc.decide_action(step_duration_minutes=step_duration_minutes)
                npc_actions.append({"npc_name": conversation_target_npc.name, **(npc_response_action or {})})
                print(f"DEBUG: NPC decided action: {npc_response_action}")

                if not npc_response_action:
                    logger.warning(f"NPC '{conversation_target_npc.name}' failed to produce a valid response.")
                    final_observations.append(f"{conversation_target_npc.name} seems unsure how to respond.")
                    conversation_active = False; break

                npc_action = npc_response_action.get("action", "")
                npc_details = npc_response_action.get("action_details", {})
                npc_utterance = npc_details.get("utterance") if isinstance(npc_details, dict) else None

                if npc_action == "talk" and npc_utterance:
                    # NPC continues conversation
                    dialogue_obs = self._format_dialogue_observation(conversation_target_npc.name, initiator_name, npc_utterance)
                    conversation_history.append({"speaker": conversation_target_npc.name, "utterance": npc_utterance})
                    final_observations.append(dialogue_obs) # Add to overall observations
                    self.console.print(f"   💬 [bold]{conversation_target_npc.name}[/bold] responds: \"{npc_utterance}\"")
                    print(f"DEBUG: Added NPC response to observations: '{npc_utterance}'")

                    if exchange_count >= MAX_EXCHANGES:
                        print("DEBUG: Max conversation exchanges reached.")
                        conversation_active = False; break

                    # --- Simulacra's Turn (Follow-up) ---
                    self.console.print(f"   -> [bold]{initiator_name}'s[/bold] turn (follow-up)...")
                    print("DEBUG: Generating Simulacra follow-up...")
                    simulacra_follow_up = await self._generate_simulacra_follow_up(
                        initiator_persona=initiator_persona,
                        npc_name=conversation_target_npc.name,
                        last_npc_utterance=npc_utterance,
                        # Pass only dialogue observations from history? Or all observations? Let's try just dialogue for focus.
                        conversation_history=[obs for obs in final_observations if isinstance(obs, dict) and obs.get('type') == 'dialogue'],
                        step_duration_minutes=step_duration_minutes
                    )
                    print(f"DEBUG: Simulacra follow-up generated: {simulacra_follow_up}")

                    if simulacra_follow_up and simulacra_follow_up.get("action") == "talk":
                        follow_up_details = simulacra_follow_up.get("action_details", {})
                        follow_up_utterance = follow_up_details.get("utterance")
                        if follow_up_utterance:
                            follow_up_obs = self._format_dialogue_observation(initiator_name, conversation_target_npc.name, follow_up_utterance)
                            conversation_history.append({"speaker": initiator_name, "utterance": follow_up_utterance})
                            final_observations.append(follow_up_obs)
                            self.console.print(f"   💬 [bold]{initiator_name}[/bold] follows up: \"{follow_up_utterance}\"")
                            print("DEBUG: Added Simulacra response to observations")
                            # Loop continues...
                        else:
                            logger.warning("Simulacra follow-up 'talk' action missing utterance.")
                            conversation_active = False
                    else:
                        # Simulacra chose to end the conversation
                        end_action = simulacra_follow_up.get('action', 'end') if simulacra_follow_up else 'end'
                        print(f"DEBUG: Simulacra chose to end conversation with action: {end_action}")
                        final_observations.append(f"{initiator_name} decides to {end_action} the conversation.")
                        conversation_active = False

                else: # NPC chose non-talk action
                    print(f"DEBUG: NPC chose action '{npc_action}' - ending conversation")
                    final_observations.append(f"{conversation_target_npc.name} {npc_action}s, ending the conversation.")
                    conversation_active = False

            # End of conversation loop handling
            if exchange_count >= MAX_EXCHANGES:
                final_observations.append(f"The conversation with {conversation_target_npc.name} eventually winds down.")
                final_consequences.append(f"Had an extended conversation with {conversation_target_npc.name}.")

            # Update NPC status after conversation
            if conversation_target_npc and self.immediate_environment and 'specific_npcs_present' in self.immediate_environment:
                last_npc_action = next((act for act in reversed(npc_actions) if act.get("npc_name") == conversation_target_npc.name), None)
                final_status = "Present" # Default
                if last_npc_action:
                     final_status = f"{last_npc_action.get('action', 'Present')} {json.dumps(last_npc_action.get('action_details', ''), default=str)}"
                print(f"DEBUG: Setting final status for {conversation_target_npc.name} to '{final_status}'")
                self.immediate_environment['specific_npcs_present'] = [
                    {**npc_state, 'status': final_status} if isinstance(npc_state, dict) and npc_state.get('name') == conversation_target_npc.name else npc_state
                    for npc_state in self.immediate_environment['specific_npcs_present']
                ]
            print(f"DEBUG: Conversation ended after {exchange_count} exchanges")

        except Exception as npc_err:
            logger.error(f"Error during conversation handling with '{target_npc_name}': {npc_err}", exc_info=True)
            print(f"DEBUG: Error during conversation: {npc_err}")
            self.console.print(f"[red]Error during conversation with '{target_npc_name}': {npc_err}[/red]")
            final_observations.append(f"The conversation with {target_npc_name} is interrupted unexpectedly.")
        # --- End Conversation Handling ---


        # --- NPCs NOT involved in conversation take their turns ---
        if process_npc_turns:
            for npc in active_npcs:
                 if npc == conversation_target_npc: continue # Skip the one who just conversed

                 self.console.print(f"   -> [bold]{npc.name}'s[/bold] turn (non-conversation)...")
                 print(f"DEBUG: Processing non-conversation turn for {npc.name}")
                 await npc.process_perception(npc_perception_data) # Process general perception
                 npc_action = await npc.decide_action(step_duration_minutes)
                 npc_actions.append({"npc_name": npc.name, **(npc_action or {})}) # Log action

                 if npc_action:
                      verb = npc_action.get('action', 'idle')
                      details = npc_action.get('action_details')
                      details_str = json.dumps(details, default=str) if details else ""
                      npc_status_str = f"{verb} {details_str}".strip()
                      self.console.print(f"   -> {npc.name} performs action: {npc_status_str}")
                      print(f"DEBUG: NPC {npc.name} action: {npc_status_str}")
                      # Update environment based on NPC action
                      final_observations.append(f"{npc.name} is {verb}ing" + (f" with details: {details}" if details else "."))
                      # Update this NPC's status in the environment
                      if self.immediate_environment and 'specific_npcs_present' in self.immediate_environment:
                          self.immediate_environment['specific_npcs_present'] = [
                              {**npc_state, 'status': npc_status_str} if isinstance(npc_state, dict) and npc_state.get('name') == npc.name else npc_state
                              for npc_state in self.immediate_environment['specific_npcs_present']
                          ]
                 else:
                      self.console.print(f"   -> {npc.name} does nothing notable.")
                      print(f"DEBUG: NPC {npc.name} did nothing notable.")
                      # Ensure status reflects inaction or previous state
                      if self.immediate_environment and 'specific_npcs_present' in self.immediate_environment:
                          self.immediate_environment['specific_npcs_present'] = [
                               {**npc_state, 'status': 'Idle'} if isinstance(npc_state, dict) and npc_state.get('name') == npc.name else npc_state
                               for npc_state in self.immediate_environment['specific_npcs_present']
                          ]
        # --- End NPC Reactions ---


        # --- Stage 3: Generate Image ---
        generated_image_prompt = None
        generated_image_path = None
        if self.image_generation_enabled and self.cycle_num % self.image_generation_frequency == 0:
            self.console.print(f"\n[bold blue]3. Generating Scene Image (Model: {self.gemini_image_model_name})...[/bold blue]")
            try:
                 # Ensure persona and action are passed correctly
                 generated_image_prompt, generated_image_path = await self._generate_scene_image(
                     initiator_name=initiator_name,
                     initiator_persona=initiator_persona,
                     initiator_action=initiator_action, # Use original intended action for context
                     cycle_num=self.cycle_num
                 )
                 if generated_image_path:
                     self.console.print(f"[blue] -> Image generated: {generated_image_path}[/blue]")
                 else:
                     self.console.print("[yellow] -> Image generation failed or skipped.[/yellow]")
            except Exception as img_gen_err:
                 logger.error(f"Error during image generation stage: {img_gen_err}", exc_info=True)
                 self.console.print(f"[red]Error generating image: {img_gen_err}[/red]")


        # --- Stage 4: Generate Final Narrative ---
        self.console.print(f"\n[bold magenta]4. Generating Final Narrative Update...[/bold magenta]")
        narrative_update = "[Narrative NA]"
        try:
            # Use history *before* the final perception log entry for context
            # Filter relevant recent history for the prompt
            action_history_strs = [
                f"{item['agent']}: {item['data'].get('action', '?')}"
                for item in self.run_history[-7:-1] # Look back a few cycles, exclude current perception
                if item.get('type') == 'action' and isinstance(item.get('data'), dict)
            ]

            narrative_update = await self._generate_updated_narrative(
                persona=initiator_persona,
                previous_actions=action_history_strs,
                world_state=self.world_state, # Use final world state
                immediate_environment=self.immediate_environment, # Use final environment state
                consequences=final_consequences, # Use consolidated consequences
                observations=final_observations, # Use consolidated observations
                step_duration_minutes=step_duration_minutes,
                recent_narrative_updates=recent_narrative_updates,
                initiator_reflection=initiator_reflection,
                initiator_thought_process=initiator_thought_process,
                # <<< ADDED: Pass Day Arc to Narrative Generation >>>
                day_arc=self.current_day_arc
            )
            self.console.print(Panel(narrative_update, title="Narrative Update", border_style="magenta"))
        except Exception as narr_err:
            logger.error(f"Narrative update failed: {narr_err}", exc_info=True)
            narrative_update = "[Narrative Error]"

        # --- Stage 5: Assemble final perception data ---
        # Ensure this reflects the FINAL state after all processing
        final_perception_data = {
            "world_state": self.world_state,
            "immediate_environment": self.immediate_environment,
            "consequences": final_consequences,
            "observations": final_observations,
            "narrative_update": narrative_update,
            "image_prompt": generated_image_prompt,
            "image_path": generated_image_path # Correct key based on image gen method
        }

        # --- Logging and Saving ---
        # Log the final perception data for this cycle
        self.run_history.append({"timestamp": f"{self.world_state.get('current_date')} {self.world_state.get('current_time')}", "type": "perception", "agent": initiator_name, "data": final_perception_data, "cycle": self.cycle_num})
        self.save_state() # Save the complete state at the end of the cycle

        end_time = time.time()
        cycle_duration = end_time - start_time
        logger.info(f"Finished processing update for '{initiator_name}'. Duration: {cycle_duration:.2f}s. Returning {len(final_observations)} observations.")
        logger.info(f"--- Cycle {self.cycle_num} End ---")

        # Return the final perception data needed by the main loop or Simulacra
        return final_perception_data


    async def _generate_simulacra_follow_up(self,
                                        initiator_persona: Dict,
                                        npc_name: str,
                                        last_npc_utterance: str,
                                        conversation_history: List[Dict], # Expect list of dialogue dicts
                                        step_duration_minutes: int) -> Optional[Dict]:
        """Generates a follow-up response from the Simulacra during an ongoing conversation."""
        initiator_name = initiator_persona.get('name', 'Simulacra')
        print(f"DEBUG: Generating follow-up response for {initiator_name} to {npc_name}")
        # Create a simple prompt for the follow-up response
        formatted_history = "\n".join([
            f"{obs.get('from', '?')}: {obs.get('utterance', '?')}"
            for obs in conversation_history # Already filtered to dialogue dicts
            if isinstance(obs, dict) # Double check type
        ])

        print(f"DEBUG: Formatted conversation history ({len(conversation_history)} entries):\n{formatted_history}")
        system_message = f"You are {initiator_name}, responding in conversation with {npc_name}. Generate a natural follow-up response or choose to end the conversation if it feels complete. Respond ONLY with ActionDecisionResponse JSON."

        prompt = f"""
Context:
You are {initiator_name}. Your personality traits are {initiator_persona.get('personality_traits', ['Neutral'])}. Your current goals are {initiator_persona.get('goals', ['Interact naturally'])}. Your emotional state is {initiator_persona.get('current_state', {}).get('emotional', 'Neutral')}.

Conversation History:
{formatted_history}

{npc_name} just said: "{last_npc_utterance}"

Task: Decide your response to {npc_name}.
Either:
1. Continue the conversation (action: "talk", with an "utterance" in action_details).
2. End the conversation (action: "wait", "move", or another appropriate non-talk action).

Consider the flow, your persona, and whether the conversation has reached a natural conclusion.

Respond ONLY with JSON matching this structure:
{{"thought_process": "Your reasoning for the response/action.", "action": "talk OR other action verb", "action_details": {{"utterance": "Your response text here (if talking)" }} or other details if not talking }}
"""
        print(f"DEBUG: Sending follow-up prompt to LLM (System: {system_message[:100]}... Prompt: {prompt[:100]}...)")

        try:
            # Use generate_and_validate for robustness
            follow_up_response = await generate_and_validate_llm_response(
                llm_service=self.llm_service,
                prompt=prompt,
                response_model=ActionDecisionResponse, # Use the Action model
                system_instruction=system_message,
                operation_description=f"Simulacra Conversation Follow-up ({initiator_name} -> {npc_name})",
                temperature=0.7, # Adjust temp as needed for conversation
            )
            print(f"DEBUG: Received raw response dict from validator: {follow_up_response}")

            if follow_up_response and "error" not in follow_up_response:
                 # Basic validation (already done by validator, but good practice)
                 if "action" in follow_up_response:
                      print(f"DEBUG: Valid follow-up response with action: {follow_up_response.get('action')}")
                      return follow_up_response
                 else:
                      logger.error(f"Validated follow-up response missing 'action': {follow_up_response}")
            else:
                 logger.error(f"Failed to generate or validate simulacra follow-up: {follow_up_response}")
                 print(f"DEBUG: Follow-up generation/validation failed: {follow_up_response}")

            # Fallback if LLM fails or validation issues persist
            return {"action": "wait", "thought_process": "Struggled to formulate response.", "action_details": {"reason": "Unable to formulate response"}}

        except Exception as e:
            logger.error(f"Error generating simulacra follow-up: {e}", exc_info=True)
            print(f"DEBUG: Exception in _generate_simulacra_follow_up: {e}")
            return None


    def _search_news(self, query: str) -> str:
        """Performs a news search using the configured tool."""
        if not self.search_tool:
            logger.warning("Search tool not initialized.")
            return "News search unavailable."
        try:
            logger.info(f"Performing news search with query: '{query}'")
            # Assuming search_tool.run is synchronous. If async, use await.
            results = self.search_tool.run(query)
            if not results or "No good DuckDuckGo Search Result found" in results:
                logger.info(f"No relevant news found for query: '{query}'")
                return "No relevant news found."
            logger.info(f"News search results for '{query}': {results[:150]}...")
            # Simple processing - maybe limit length or format
            return results.strip()
        except Exception as e:
            logger.error(f"Error during news search '{query}': {e}", exc_info=True)
            return "Error fetching news."


    async def _gather_comprehensive_news(self, location: Dict) -> str:
        """Gathers news from multiple queries."""
        city = location.get("city", "Unknown City")
        country = location.get("country", "Unknown Country")
        logger.info(f"Gathering comprehensive news for {city}, {country}")

        # Define relevant queries
        queries = [
            f"current events news {city} {country}",
            f"weather forecast {city} {country}",
            f"local economy news {city}",
            # Add more queries as needed (social issues, transportation, etc.)
        ]

        # Run searches concurrently (if tool allows/is thread-safe, or use asyncio.to_thread)
        # Using asyncio.to_thread assuming the tool's .run() is blocking I/O
        tasks = [asyncio.to_thread(self._search_news, query) for query in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        news_output = []
        for i, query in enumerate(queries):
            result = results_list[i]
            category = query.split()[1].capitalize() # Simple category extraction

            if isinstance(result, Exception):
                logger.error(f"News search failed for '{category}' (Query: '{query}'): {result}")
                news_output.append(f"--- {category} News ---\nError fetching news.\n")
            elif result and result not in ["No relevant news found.", "News search unavailable.", "Error fetching news."]:
                # Format the result slightly
                news_output.append(f"--- {category} News ---\n{result}\n")

        final_news = "\n".join(news_output).strip()
        logger.info(f"Finished gathering news. Length: {len(final_news)} chars.")
        return final_news if final_news else "No specific news could be retrieved."


    async def _determine_plausible_location(self, config: Dict) -> str:
        """Determines a plausible starting location using LLM or config."""
        city = config.get("location", {}).get("city", "DefaultCity")
        country = config.get("location", {}).get("country", "DefaultCountry")
        logger.info(f"Determining plausible starting location in {city}, {country}")

        # Try loading persona for context if available (adjust path if needed)
        persona_for_context = {}
        persona_path = self.state_path # Assumes persona is in the same combined file
        if os.path.exists(persona_path):
            try:
                with open(persona_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                if "simulacra_state" in data and isinstance(data["simulacra_state"], dict) and \
                   "persona" in data["simulacra_state"] and isinstance(data["simulacra_state"]["persona"], dict):
                     persona_for_context = data["simulacra_state"]["persona"]
                     logger.info(f"Using persona '{persona_for_context.get('name', 'Unknown')}' for location context.")
            except Exception as e:
                logger.warning(f"Could not load persona from {persona_path} for location context: {e}")

        persona_prompt_part = "[No specific character context]"
        if persona_for_context:
            persona_prompt_part = f"Character Context: Name: {persona_for_context.get('name', '?')} Age: {persona_for_context.get('age', '?')} Occupation: {persona_for_context.get('occupation', '?')}"

        prompt = f"Determine a realistic, specific starting location name for a simulation based on this context:\nCity: {city}, Country: {country}\n{persona_prompt_part}\nConsider typical character activities. Choose a specific place (e.g., 'Central Park Bench', 'Starbucks on 5th Ave', 'Office Lobby', 'Apartment Living Room'). Avoid overly generic terms. Return ONLY the location name. Example: Bryant Park Cafe Table"
        fallback_location = f"Downtown Square in {city}"

        # Use generate_content for plain text response
        response_dict = await self.llm_service.generate_content(
            prompt=prompt,
            system_instruction="Determine a realistic, specific location name. Respond ONLY with the name.",
            operation_description="Plausible Location Determination",
            temperature=0.6,
            response_model=None # Expect plain text
        )

        location = fallback_location
        if response_dict and isinstance(response_dict.get('text'), str):
             extracted_text = response_dict['text']
             # Clean potential quotes, newlines, etc.
             cleaned_location = extracted_text.strip('"\' \n\t').split('\n')[0].strip()
             if 0 < len(cleaned_location) < 100: # Basic sanity check
                 location = cleaned_location
             else:
                 logger.warning(f"Cleaned location seems invalid ('{cleaned_location}'). Using fallback.")
        elif response_dict and "error" in response_dict:
             logger.error(f"LLM error determining location: {response_dict['error']}")

        self.console.print(f"[green]Determined starting location:[/green] {location}")
        return location


    async def _generate_persona_for_location(self, city: str, country: str) -> Dict:
        """Generates a placeholder persona (should ideally be handled by Simulacra)."""
        logger.warning("WorldEngine generating placeholder persona. Ideally, Simulacra handles its own generation.")
        prompt = f"Create a realistic persona JSON for a character in {city}, {country}. Consider demographics, culture, lifestyle. Return ONLY the persona JSON object with keys: name, age, occupation, personality_traits (list), goals (list), current_state (dict with physical, emotional, mental), memory (dict with short_term, long_term lists)."
        default_persona = {
            "name": f"Resident of {city}", "age": 30, "occupation": "Local", "personality_traits": ["observant"],
            "goals": ["get through the day"], "current_state": {"physical": "ok", "emotional": "neutral", "mental": "aware"},
            "memory": {"short_term": [], "long_term": []}
        }

        # Use generate_and_validate expecting JSON output (but no specific model needed here)
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            system_instruction=f"Create a realistic persona JSON for {city}, {country}. Respond ONLY with the JSON.",
            operation_description="Placeholder Persona Generation",
            temperature=0.7,
            response_model=None # Expect raw JSON dict
        )

        if response_dict and "error" not in response_dict and isinstance(response_dict, dict):
            # Basic check for essential keys
            if "name" in response_dict and "age" in response_dict:
                return response_dict
            else:
                logger.error(f"Generated persona missing required fields: {response_dict}")
                return default_persona
        else:
            logger.error(f"Failed to generate placeholder persona: {response_dict}")
            return default_persona


    def get_current_datetime(self) -> Dict:
        """Gets the current UTC date and time."""
        now = dt.datetime.now(dt.timezone.utc)
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A")
        }


    async def _initialize_from_life_summary(self, file_path: str):
        """Initializes world state from a life summary file."""
        logger.info(f"Initializing world state from life summary: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                life_data = json.load(f)

            persona_details = life_data.get("persona_details")
            if not persona_details or not isinstance(persona_details, dict):
                raise ValueError("Missing/invalid 'persona_details' in life summary")

            # Determine Start Date/Time (complex logic, simplified here)
            age = persona_details.get("Age")
            birth_year = life_data.get("birth_year")
            birth_month = life_data.get("birth_month", 1)
            birth_day = life_data.get("birth_day", 1)
            start_time = "09:00" # Default start time
            start_date_str = None
            target_year = None

            if isinstance(age, int) and isinstance(birth_year, int):
                 target_year = birth_year + age
                 try:
                      # Basic date validation/clamping
                      birth_month = max(1, min(12, birth_month))
                      days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} # Ignore leap year for simplicity
                      birth_day = max(1, min(days_in_month.get(birth_month, 31), birth_day))
                      start_date = dt.date(target_year, birth_month, birth_day)
                      start_date_str = start_date.strftime("%Y-%m-%d")
                      logger.info(f"Calculated start date from life summary: {start_date_str}")
                 except ValueError as e:
                      logger.error(f"Date construction error from life summary ({target_year}-{birth_month}-{birth_day}): {e}. Using fallback date.");
                      start_date_str = f"{target_year}-01-01" if target_year else None

            if not start_date_str:
                 logger.warning("Cannot determine start date from life summary. Using current date/time.")
                 now = self.get_current_datetime()
                 start_date_str, start_time = now['date'], now['time']

            # Determine Location
            start_location_name = persona_details.get("Current_location", persona_details.get("Birthplace_City", "Unknown Location"))
            base_location = {
                "city": persona_details.get("Current_city", persona_details.get("Birthplace_City", "Unknown City")),
                "country": persona_details.get("Current_country", persona_details.get("Birthplace_Country", "Unknown Country")),
                "region": persona_details.get("Current_region", persona_details.get("Birthplace_Region", "Unknown Region"))
            }

            # Gather Context Summary (e.g., last year's events)
            context_summary = f"Initializing based on life summary. Character: {persona_details.get('Name', '?')} Age: {age}. Location: {start_location_name}. "
            if target_year and isinstance(life_data.get("yearly_summaries"), dict):
                summary_year = str(target_year - 1)
                latest_yearly = life_data["yearly_summaries"].get(summary_year, f"No summary available for year {summary_year}.")
                context_summary += f"Context from end of year {summary_year}: {latest_yearly}"
            else:
                context_summary += "No recent yearly summary context available."

            self.console.print(f"[yellow]Generating historical world state ({start_date_str} {start_time})...[/yellow]")
            generated_ws = await self._generate_world_state_from_context(date=start_date_str, time=start_time, location=base_location, context_summary=context_summary)
            self.world_state = generated_ws if generated_ws else self._create_default_world_state(self.config) # Fallback

            self.console.print(f"[yellow]Creating environment for '{start_location_name}'...[/yellow]")
            generated_env = await self._generate_environment_from_context(world_state=self.world_state, location_name=start_location_name)
            self.immediate_environment = generated_env if generated_env else await self._create_default_immediate_environment(start_location_name) # Fallback

            self.console.print("[yellow]Generating initial narrative context...[/yellow]")
            self.initial_narrative_context = await self._generate_narrative_context(persona_details, self.world_state, self.immediate_environment.get('current_location_name', start_location_name))
            logger.info(f"World state prepared based on life summary '{file_path}'.")

        except FileNotFoundError:
            logger.error(f"Life summary file not found: {file_path}")
            raise
        except (KeyError, ValueError, TypeError, ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Error processing life summary '{file_path}': {e}", exc_info=True)
            raise ValueError(f"Invalid format/data in life summary '{file_path}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing from life summary '{file_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize from life summary: {e}") from e


    async def initialize_new_world(self, config: Dict) -> Dict:
        """Initializes a new world state based on CURRENT time/news."""
        logger.info("Initializing new world based on current time/news...")
        location_config = config.get("location", {"city": "NYC", "country": "USA"}) # Use config or default

        # Defaults in case generation fails
        default_ws = self._create_default_world_state(config)
        default_env = await self._create_default_immediate_environment("Default Room")
        default_narrative = "World initialization failed. Starting default."
        default_observations = ["World feels generic."]

        try:
            current_datetime = self.get_current_datetime()
            self.console.print(f"[yellow]Gathering current news for {location_config.get('city')}...[/yellow]")
            news = await self._gather_comprehensive_news(location_config)

            self.console.print("[yellow]Generating current world state...[/yellow]")
            prompt_ws = PromptManager.initialize_world_state_from_news_prompt(news, config, current_datetime)
            gen_ws = await generate_and_validate_llm_response(
                llm_service=self.llm_service,
                prompt=prompt_ws,
                response_model=WorldState, # Expecting a full WorldState JSON
                operation_description="Current World State Generation"
            )
            self.world_state = gen_ws if gen_ws else default_ws
            # Ensure required time/date fields are present even if LLM omitted them
            self.world_state['current_date'] = self.world_state.get('current_date', current_datetime['date'])
            self.world_state['current_time'] = self.world_state.get('current_time', current_datetime['time'])
            self.world_state['day_phase'] = get_day_phase(dt.datetime.strptime(self.world_state['current_time'], "%H:%M").hour)


            # Determine plausible starting location
            start_loc = await self._determine_plausible_location(config)

            self.console.print(f"[yellow]Creating current environment for '{start_loc}'...[/yellow]")
            env_prompt = PromptManager.initialize_immediate_environment_prompt(self.world_state, start_loc)
            gen_env = await generate_and_validate_llm_response(
                llm_service=self.llm_service,
                prompt=env_prompt,
                response_model=ImmediateEnvironment, # Expecting ImmediateEnvironment JSON
                operation_description=f"Environment Generation for {start_loc}"
            )
            self.immediate_environment = gen_env if gen_env else default_env
            # Ensure location name matches what was determined
            self.immediate_environment['current_location_name'] = start_loc


            # Load persona if available for narrative context
            persona = {}
            path = self.state_path # Assumes combined state file
            if os.path.exists(path):
                 try:
                      with open(path, 'r', encoding='utf-8') as f:
                           data = json.load(f)
                           persona = data.get("simulacra_state", {}).get("persona", {})
                 except Exception as e:
                      logger.error(f"Failed loading persona from {path} for initial narrative: {e}")

            self.console.print("[yellow]Generating initial narrative context...[/yellow]")
            self.initial_narrative_context = await self._generate_narrative_context(
                persona, self.world_state, self.immediate_environment.get("current_location_name", start_loc)
            )
            self.initial_narrative_context = self.initial_narrative_context or default_narrative

            logger.info("New world (current time) initialized successfully.")
            return {"world_state": self.world_state, "immediate_environment": self.immediate_environment, "narrative_context": self.initial_narrative_context, "observations": [f"You are at {start_loc}.", f"Time is {self.world_state.get('current_time','?')}."]}

        except Exception as e:
            logger.critical(f"CRITICAL error during 'initialize_new_world': {e}", exc_info=True)
            self.world_state = default_ws
            self.immediate_environment = default_env
            self.initial_narrative_context = default_narrative
            return {"world_state": default_ws, "immediate_environment": default_env, "narrative_context": default_narrative, "observations": default_observations}


    async def _generate_world_state_from_context(self, date: str, time: str, location: Dict, context_summary: str) -> Optional[Dict]:
        """Generates world state from historical context."""
        prompt = PromptManager.initialize_world_state_from_context_prompt(date, time, location, context_summary)
        generated_data = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=WorldState, # Expecting full WorldState model
            operation_description=f"World State Generation for {date} {time}"
        )
        if generated_data and "error" not in generated_data:
            # Ensure date/time/location from context are preserved
            generated_data.update({"current_date": date, "current_time": time, **location})
            try:
                # Re-validate after adding context
                return WorldState.model_validate(generated_data).model_dump()
            except ValidationError as e:
                logger.error(f"Validation error after updating generated world state from context: {e}")
                return None
        else:
            logger.error(f"LLM failed to generate world state from context: {generated_data}")
            return None


    async def _generate_environment_from_context(self, world_state: Dict, location_name: str) -> Optional[Dict]:
        """Generates immediate environment from context."""
        prompt = PromptManager.initialize_immediate_environment_prompt(world_state, location_name)
        generated_data = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=ImmediateEnvironment, # Expecting ImmediateEnvironment model
            operation_description=f"Environment Generation for {location_name}"
        )
        if generated_data and "error" not in generated_data:
            # Ensure location name matches input context
            generated_data["current_location_name"] = location_name
            try:
                # Re-validate after adding context
                return ImmediateEnvironment.model_validate(generated_data).model_dump()
            except ValidationError as e:
                logger.error(f"Validation error after updating generated env from context: {e}")
                return None
        else:
            logger.error(f"LLM failed to generate environment from context: {generated_data}")
            return None

    def _create_default_world_state(self, config: Dict) -> Dict:
        """Creates a default WorldState dictionary."""
        now = dt.datetime.now(dt.timezone.utc)
        loc = config.get("location", {})
        city, country, region = loc.get("city", "DefaultCity"), loc.get("country", "DefaultCountry"), loc.get("region", "DefaultRegion")
        try:
            default_obj = WorldState(
                current_time=now.strftime("%H:%M"),
                current_date=now.strftime("%Y-%m-%d"),
                day_phase=get_day_phase(now.hour),
                city_name=city,
                country_name=country,
                region_name=region,
                district_neighborhood="Downtown",
                weather_condition="Clear",
                temperature_c=20.0,
                forecast="Stable",
                wind_description="Calm",
                precipitation_type="None",
                social_climate="Neutral",
                economic_condition="Stable",
                political_climate="Quiet",
                major_events=[],
                local_news=[],
                transportation_status="Normal",
                utility_status="Stable",
                public_announcements=[],
                trending_topics=[],
                current_cultural_events=[],
                public_health_status="Normal",
                public_safety_status="Safe"
            )
            logger.warning("Using default world state.")
            return default_obj.model_dump()
        except ValidationError as e:
            logger.critical(f"Failed validation creating DEFAULT WorldState: {e}", exc_info=True)
            return { # Minimal fallback dict
                "error": "Default WorldState creation failed",
                "current_time": now.strftime("%H:%M"),
                "current_date": now.strftime("%Y-%m-%d")
            }


    async def _create_default_immediate_environment(self, location_name: str) -> Dict:
        """Creates a default ImmediateEnvironment dictionary."""
        try:
            fallback_obj = ImmediateEnvironment(
                current_location_name=location_name,
                location_type="Generic Space",
                indoor_outdoor="Indoor",
                noise_level="Quiet",
                lighting="Fluorescent",
                temperature_feeling="Comfortable",
                # ambient_temperature_c=20.0, # Optional field
                air_quality="Filtered",
                crowd_density="Sparse",
                social_atmosphere="Neutral",
                seating_availability="Ample",
                # restroom_access="Available", # Optional field
                present_people=[],
                specific_npcs_present=[],
                ongoing_activities=[],
                nearby_objects=["Walls"],
                available_services=[],
                exit_options=["Door"],
                interaction_opportunities=[],
                points_of_interest=[],
                visible_features=["Plain"],
                audible_sounds=[],
                noticeable_smells=[],
                food_drink_options=[],
                recent_changes=[],
                ongoing_conversations=[],
                attention_drawing_elements=[]
            )
            logger.warning(f"Using hardcoded default environment for '{location_name}'.")
            return fallback_obj.model_dump()
        except ValidationError as e:
            logger.critical(f"Failed validation creating DEFAULT ImmediateEnvironment for '{location_name}': {e}", exc_info=True)
            return { # Minimal fallback dict
                 "error": "Default Env creation failed",
                 "current_location_name": location_name
            }

    def _sanitize_string_list(self, data: Any, field_name: str) -> List[str]:
        """Ensures data is a list of strings."""
        if isinstance(data, list):
            # Ensure all items are strings
            return [str(item) for item in data]
        elif isinstance(data, str):
            # Wrap single string in a list
            return [data] if data else []
        elif data is None:
            return []
        else:
            logger.warning(f"Invalid format for field '{field_name}' (expected list of str, got {type(data)}). Using []. Value: {data}")
            try:
                # Attempt conversion if possible (e.g., list of ints)
                return [str(item) for item in data]
            except TypeError:
                return []

    def save_state(self):
        """Saves the WorldEngine's current state including the Day Arc."""
        if not self.world_state or not self.immediate_environment:
            logger.error("Save state failed: WorldEngine state not fully initialized.")
            return

        # Prepare world engine specific state
        world_engine_current_state = {
            "world_state": self.world_state,
            "immediate_environment": self.immediate_environment,
            "run_history": self.run_history,
            "reaction_profile": self.reaction_profile.model_dump(),
            "cycle_num": self.cycle_num, # Save cycle number
            "current_day_arc": self.current_day_arc # Save Day Arc
        }

        # Load existing full state to update only the world engine part
        full_state_data = {}
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as file:
                    try:
                        full_state_data = json.load(file)
                        if not isinstance(full_state_data, dict):
                             logger.error(f"State file {self.state_path} is not a valid JSON object. Overwriting.")
                             full_state_data = {}
                    except json.JSONDecodeError:
                        logger.error(f"State file {self.state_path} corrupted. Overwriting.", exc_info=True)
                        full_state_data = {}
            else:
                logger.info(f"State file {self.state_path} not found. Creating new file.")
                full_state_data = {}

            # Ensure base structure exists
            full_state_data.setdefault("configuration", self.config if self.config else DEFAULT_SIMULACRA_STATE["configuration"])
            full_state_data.setdefault("simulacra_state", DEFAULT_SIMULACRA_STATE["simulacra_state"])

            # Update world engine state part
            full_state_data["world_engine_state"] = world_engine_current_state

            # --- Check directory path before creating ---
            dir_path = os.path.dirname(self.state_path)
            if dir_path: # Only create directories if dir_path is not empty
                os.makedirs(dir_path, exist_ok=True)
            # --- End Check ---

            # Save the updated full state
            with open(self.state_path, 'w', encoding='utf-8') as file:
                json.dump(full_state_data, file, indent=2, default=str)

            logger.info(f"WorldEngine state (incl. arc) saved to combined file: {self.state_path}")

        except Exception as e:
            logger.error(f"Error saving combined state to {self.state_path}: {e}", exc_info=True)
            self.console.print(f"[red]Error saving world state: {e}[/red]")

    def get_world_summary(self) -> str:
        """Generate a concise summary string of the current world state."""
        if not self.world_state:
            return "[bold red]World state NA.[/bold red]"
        ws = self.world_state
        try:
            time_date = f"[cyan]{ws.get('current_time', '?')} {ws.get('current_date', '?')} ({ws.get('day_phase', '?')})[/]"
            loc = f"[green]{ws.get('city_name', '?')}, {ws.get('district_neighborhood', ws.get('region_name', '?'))}[/]"
            temp_c = ws.get('temperature_c', '?')
            temp_str = f"{temp_c}°C" if isinstance(temp_c, (int, float)) else '?'
            weather = f"[yellow]{ws.get('weather_condition', '?')} {temp_str} {ws.get('wind_description', '')}[/]"

            events = ws.get('major_events', [])
            valid_events = [str(e) for e in events if e and str(e).strip().lower() not in ["none", "n/a", ""]]
            event_str = f"[magenta]Events:[/magenta] {', '.join(valid_events[:2])}" if valid_events else "[i]No major events.[/i]"

            # Conditions summary
            conditions = []
            if ws.get('economic_condition') and "unstable" in str(ws['economic_condition']).lower(): conditions.append("Eco. unstable")
            if ws.get('social_climate') and "tension" in str(ws['social_climate']).lower(): conditions.append("Social tension")
            if ws.get('political_climate') and "unrest" in str(ws['political_climate']).lower(): conditions.append("Pol. unrest")
            precip = ws.get('precipitation_type')
            if precip and precip != "None": conditions.append(f"{precip}")
            condition_str = f"[orange3]Conditions:[/orange3] {', '.join(conditions)}" if conditions else ""

            # <<< ADDED: Include Day Arc in Summary >>>
            day_arc_str = f"\n[purple]Day Arc:[/purple] {self.current_day_arc}" if self.current_day_arc else ""

            return f"[b]World:[/b] {time_date} {loc}. {weather}. {event_str} {condition_str}{day_arc_str}".strip()
        except Exception as e:
            logger.error(f"World summary generation error: {e}", exc_info=True)
            return "[red]Err Generating World Summary[/red]"

    async def _generate_scene_image(self,
                                   initiator_name: str,
                                   initiator_persona: Dict,
                                   initiator_action: Dict,
                                   cycle_num: int) -> tuple[Optional[str], Optional[str]]:
        """
        Generates an image prompt based on the current state and requests an image via Gemini.

        Args:
            initiator_name: Name of the character initiating the action (focus of the scene).
            initiator_persona: Dictionary containing the initiator's current state (emotions, etc.).
            initiator_action: Dictionary containing the action the initiator just took.
            cycle_num: The current simulation cycle number.

        Returns:
            A tuple containing (image_path_str, generated_prompt) or (None, generated_prompt) on failure.
            image_path_str is the string path if successful, None otherwise.
        """
        generated_prompt = "[Image prompt generation failed]" # Default prompt on error
        image_filepath_str = None

        if not self.world_state or not self.immediate_environment:
            logger.error("Cannot generate image: World state or environment not initialized.")
            return image_filepath_str, generated_prompt
        if not self.llm_service:
             logger.error("Cannot generate image: LLMService not available.")
             return image_filepath_str, generated_prompt
        # Check if Gemini client is available (adjust based on how LLMService handles it)
        if not hasattr(self.llm_service, 'gemini_client') or not self.llm_service.gemini_client:
             logger.error("Cannot generate image: Gemini client not available in LLMService.")
             # Alternatively, try initializing it here if feasible:
             # try: genai.configure(...) except Exception: pass
             return image_filepath_str, generated_prompt

        try:
            # --- Extract relevant details for prompt ---
            ws = self.world_state
            env = self.immediate_environment
            action_type = initiator_action.get('action', 'being present')
            action_details = initiator_action.get('action_details', {}) # Should be dict or None
            action_target = action_details.get('target') if isinstance(action_details, dict) else None
            action_manner = action_details.get('manner') if isinstance(action_details, dict) else None
            action_utterance = action_details.get('utterance') if isinstance(action_details, dict) else None

            persona_state = initiator_persona.get('current_state', {})
            emotion = persona_state.get('emotional', 'neutral')
            physical = persona_state.get('physical', 'normal appearance') # More descriptive default

            location_name = env.get('current_location_name', 'Unknown Location')
            location_type = env.get('location_type', 'Generic Space')
            time_str = ws.get('current_time', '?')
            day_phase = ws.get('day_phase', 'daytime')
            weather = ws.get('weather_condition', 'clear')
            lighting_desc = env.get('lighting', 'average lighting')
            atmosphere = env.get('social_atmosphere', 'neutral')

            # Describe the main action
            action_desc = f"{initiator_name} is {action_type}"
            if action_type == 'talk' and action_utterance: action_desc = f"{initiator_name} is talking, saying '{action_utterance[:50]}...'"
            elif action_type == 'move' and isinstance(action_details, dict) and action_details.get('target_location'): action_desc = f"{initiator_name} is moving towards {action_details['target_location']}"
            elif action_type == 'use' and isinstance(action_details, dict) and action_details.get('item'): action_desc = f"{initiator_name} is using {action_details['item']}"
            elif action_target: action_desc += f" targeting {action_target}"
            if action_manner: action_desc += f" ({action_manner})"


            # Describe the people present concisely
            npcs = env.get('specific_npcs_present', [])
            people_desc = "The place is quiet."
            if isinstance(npcs, list) and npcs:
                npc_descs = []
                for npc in npcs[:3]: # Limit number of NPCs described
                     if isinstance(npc, dict):
                          npc_name = npc.get('name', 'Someone')
                          npc_role = npc.get('role', 'person')
                          npc_status = npc.get('status', 'present')
                          if npc_name != initiator_name: # Don't describe initiator here
                              npc_descs.append(f"{npc_name} ({npc_role}, {npc_status})")
                if npc_descs: people_desc = f"Nearby people: {'; '.join(npc_descs)}."
            elif env.get('present_people'): people_desc = f"General types of people present: {', '.join(env.get('present_people',[]))}."

            # Describe key visual features
            scene_details_list = env.get('visible_features', []) + env.get('points_of_interest', [])
            scene_details = f"Key visual elements: {', '.join(scene_details_list[:3])}." if scene_details_list else "The scene has simple features."

            # Assemble the prompt
            # Refined prompt structure
            generated_prompt = (
                f"Anime illustration capturing a moment in time. Focus: {initiator_name}, appearing {physical} and feeling {emotion}. "
                f"Action: {action_desc}. "
                f"Setting: {location_name} ({location_type}), {day_phase} ({time_str}). Lighting is {lighting_desc}. "
                f"Atmosphere: {atmosphere}. Weather outside (if visible): {weather}. "
                f"Context: {people_desc} {scene_details} "
                f"Style: High detail, expressive anime art style, cinematic angle."
            )
            self.console.print(f"[cyan]Image Prompt (Cycle {cycle_num}): {generated_prompt[:150]}...[/cyan]")
            logger.info(f"Image Prompt Cycle {cycle_num}: {generated_prompt}")

            # --- Call Gemini Image Generation ---
            # Ensure the client is accessed correctly via llm_service
            client = self.llm_service.gemini_client
            response = client.generate_images( # Use correct method name
                model=self.gemini_image_model_name, # Use configured model name
                prompt=generated_prompt,
                # Optional: Add negative prompt, aspect ratio etc. via config
                # number_of_images=1, # Assuming default is 1 or using config
            )

            # Process the response (assuming response structure from SDK)
            # This structure might vary based on SDK version
            generated_image_data = None
            if hasattr(response, 'generated_images') and response.generated_images:
                # Accessing the image data - adjust if structure differs
                img_obj = response.generated_images[0]
                if hasattr(img_obj, 'image') and hasattr(img_obj.image, 'image_bytes'):
                     generated_image_data = img_obj.image.image_bytes
                # Handle other potential response structures if necessary
            elif hasattr(response, 'images') and response.images: # Check alternative structure
                 img_obj = response.images[0]
                 if hasattr(img_obj, '_blob') and hasattr(img_obj._blob, 'data'): # Check for _blob structure
                      generated_image_data = img_obj._blob.data

            if generated_image_data:
                try:
                    image = Image.open(BytesIO(generated_image_data))
                    filename = f"cycle_{cycle_num:04d}_{int(time.time())}.png"
                    image_filepath_obj = self.image_save_dir / filename
                    image.save(image_filepath_obj)
                    image_filepath_str = str(image_filepath_obj) # Store path as string
                    logger.info(f"Successfully generated and saved image: {image_filepath_str}")
                    # self.console.print(f"[green]✅ Image generated: {image_filepath_str}[/green]") # Already printed above
                except Exception as pil_err:
                    logger.error(f"Failed to process or save image data with Pillow: {pil_err}", exc_info=True)
                    self.console.print(f"[red]❌ Failed to save generated image: {pil_err}[/red]")
            else:
                logger.error(f"Image generation failed: No valid image data found in response. Response structure: {response}")
                self.console.print("[red]❌ Image generation failed (no valid data in response).[/red]")

        except Exception as img_err:
            # Catch potential API errors (rate limits, auth issues, etc.)
            logger.error(f"Gemini image generation API call failed: {img_err}", exc_info=True)
            self.console.print(f"[red]❌ Image generation API call failed: {img_err}[/red]")
            # Return the generated prompt even if API call fails

        return image_filepath_str, generated_prompt # Return path (or None) and the prompt used


    def get_environment_summary(self) -> str:
        """Generate a concise summary string of the immediate environment."""
        if not self.immediate_environment:
            return "[bold red]Environment NA.[/bold red]"
        env = self.immediate_environment
        try:
            loc = f"[cyan]{env.get('current_location_name', '?')} ({env.get('location_type', '?')})[/]"
            setting = f"[green]{env.get('indoor_outdoor', '?')}[/]"
            atmos = f"[yellow]{env.get('social_atmosphere', '?')} ({env.get('crowd_density', '?')})[/]"

            npcs_list = env.get('specific_npcs_present', [])
            npcs_names = []
            if isinstance(npcs_list, list):
                npcs_names = [f"{npc.get('name', '?')}({npc.get('status', '?')})" for npc in npcs_list if isinstance(npc, dict)]
            npc_str = f"[m]NPCs:[/m] {', '.join(npcs_names[:2])}" + ('...' if len(npcs_names) > 2 else '') if npcs_names else "[i]No NPCs.[/i]"

            activities = env.get('ongoing_activities', [])
            act_str = f"[blue]Acts:[/blue] {', '.join(map(str, activities[:2]))}" + ('...' if len(activities) > 2 else '') if activities else "[i]Quiet.[/i]"

            pois = env.get('points_of_interest', [])
            poi_str = f"[o]POI:[/o] {', '.join(map(str, pois[:1]))}" + ('...' if len(pois) > 1 else '') if pois else ""

            return f"[b]Env:[/b] {loc} {setting}. {atmos}. {npc_str} {act_str} {poi_str}".strip()
        except Exception as e:
            logger.error(f"Environment summary generation error: {e}", exc_info=True)
            return "[red]Err Generating Env Summary[/red]"

    async def _generate_narrative_context(self, persona: Dict, world_state: Dict, location_name: str) -> str:
        """Generates the initial narrative context."""
        # Ensure environment is passed if available
        env_for_prompt = self.immediate_environment if self.immediate_environment else {"current_location_name": location_name}
        narrative_prompt = PromptManager.generate_initial_narrative_prompt(persona, world_state, env_for_prompt)
        # Use generate_content for plain text
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=narrative_prompt,
            system_instruction="Generate immersive narrative context (2-3 paragraphs). Respond ONLY with narrative text.",
            operation_description="Initial Narrative Context Generation",
            temperature=0.7,
            response_model=None # Expect plain text
        )

        narrative = "[Narrative Gen Failed]"
        if response_dict and isinstance(response_dict.get('text'), str):
            narrative = response_dict['text'].strip()
            if not narrative: narrative = "[Narrative Gen Returned Empty]"
        elif response_dict and "error" in response_dict:
            narrative = f"[Narrative Err: {response_dict['error']}]"
        return narrative


    # <<< MODIFIED: Added day_arc parameter >>>
    async def _generate_updated_narrative(self, persona: Dict, previous_actions: List[str],
                                          world_state: Dict, immediate_environment: Dict,
                                          consequences: List[str], observations: List[str],
                                          step_duration_minutes: int = 1,
                                          recent_narrative_updates: Optional[List[str]] = None,
                                          initiator_reflection: Optional[str] = None,
                                          initiator_thought_process: Optional[str] = None,
                                          day_arc: Optional[str] = None) -> str:
        """Generates an updated narrative segment, considering the Day Arc."""
        if recent_narrative_updates is None: recent_narrative_updates = []

        # Call the prompt manager function, now expecting day_arc
        narrative_prompt = PromptManager.generate_narrative_update_prompt(
            persona=persona,
            previous_actions=previous_actions,
            world_state=world_state,
            immediate_environment=immediate_environment,
            consequences=consequences,
            observations=observations,
            step_duration_minutes=step_duration_minutes,
            recent_narrative_updates=recent_narrative_updates,
            initiator_reflection=initiator_reflection,
            initiator_thought_process=initiator_thought_process,
            day_arc=day_arc # <<< PASS ARC HERE >>>
        )

        # Call LLM, parse response as plain text
        response_dict = await generate_and_validate_llm_response(
             llm_service=self.llm_service,
             prompt=narrative_prompt,
             # System instruction mentions the arc implicitly via the prompt content
             system_instruction=f"Generate narrative update ({step_duration_minutes} min), maintaining continuity and considering context. Respond ONLY with narrative text.",
             operation_description="Narrative Update Generation",
             temperature=0.7, # Adjust temperature as needed
             response_model=None # Expect plain text
        )

        narrative = "[Narrative Update Failed]"
        if response_dict and isinstance(response_dict.get('narrative'), str):
            narrative = response_dict['narrative'].strip()
            if not narrative: narrative = "[Narrative Update Returned Empty]"
        elif response_dict and "error" in response_dict:
            narrative = f"[Narrative Update Err: {response_dict['error']}]"

        # Log the generated narrative (or failure)
        logger.debug(f"Generated Narrative Update (Day Arc: {'Yes' if day_arc else 'No'}): {narrative[:100]}...")
        return narrative