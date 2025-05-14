# src/config.py - Simulation Configuration and Constants

import os
from typing import Optional # Added for RANDOM_SEED type hint

# --- Core API and Model Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-2.0-flash")
SEARCH_AGENT_MODEL_NAME = os.getenv("SEARCH_AGENT_MODEL_NAME", "gemini-2.0-flash")
APP_NAME = "TheSimulationAsync"
USER_ID = "player1" # Keep USER_ID
SIMULACRA_KEY = "simulacra_profiles" # Changed from "simulacra"

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", 1))
UPDATE_INTERVAL = float(os.getenv("UPDATE_INTERVAL", 0.1))
MAX_SIMULATION_TIME = float(os.getenv("MAX_SIMULATION_TIME", 96000.0)) # Default to 30 minutes simulation time
MEMORY_LOG_CONTEXT_LENGTH = 10
MAX_MEMORY_LOG_ENTRIES = 500

# --- Agent Self-Reflection Parameters (for simulacra_agent_task_llm) ---
AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS = float(os.getenv("AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS", 120.0)) # How often self-reflection is considered
LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS = float(os.getenv("LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS", 300.0)) # Min duration for self-reflection
INTERJECTION_COOLDOWN_SIM_SECONDS = float(os.getenv("INTERJECTION_COOLDOWN_SIM_SECONDS", 450.0)) # Cooldown for any type of interjection for an agent
PROB_INTERJECT_AS_SELF_REFLECTION = float(os.getenv("PROB_INTERJECT_AS_SELF_REFLECTION", 0.60)) # Chance of self-reflection if conditions met for simulacra_agent_task_llm
AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS = float(os.getenv("AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS", 0.5)) # How often agent tasks check their status

# --- Dynamic Interruption Task Parameters (for dynamic_interruption_task) ---
DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS = float(os.getenv("DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS", 5.0)) # How often the dynamic interruption task checks, in real-world seconds
DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS = float(os.getenv("DYNAMIC_INTERruption_TARGET_DURATION_SECONDS", 600.0)) # Sim duration of an action at which target probability is met
DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION = float(os.getenv("DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION", 0.35)) # Target probability (e.g., 5%) per check
DYNAMIC_INTERRUPTION_MIN_PROB = float(os.getenv("DYNAMIC_INTERRUPTION_MIN_PROB", 0.005)) # Minimum probability for eligible actions (e.g., 0.5%) per check
DYNAMIC_INTERRUPTION_MAX_PROB_CAP = float(os.getenv("DYNAMIC_INTERRUPTION_MAX_PROB_CAP", 0.25)) # Absolute maximum probability cap per check (e.g., 15%)
MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK = float(os.getenv("MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK", 60.0)) # Actions shorter than this (sim seconds) won't be interrupted by this task
# --- World Information Gatherer Parameters ---
WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS = float(os.getenv("WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS", 3600.0)) # How often world feeds are updated
MAX_WORLD_FEED_ITEMS = int(os.getenv("MAX_WORLD_FEED_ITEMS", 5))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
WORLD_CONFIG_DIR = os.path.join(BASE_DIR, "data") # World configs are in the main data dir

# --- Random Seed ---
RANDOM_SEED_VALUE = os.getenv("RANDOM_SEED_VALUE") # Get from env
RANDOM_SEED: Optional[int] = int(RANDOM_SEED_VALUE) if RANDOM_SEED_VALUE and RANDOM_SEED_VALUE.isdigit() else None

# --- Narrative Image Generation Parameters ---
ENABLE_NARRATIVE_IMAGE_GENERATION = os.getenv("ENABLE_NARRATIVE_IMAGE_GENERATION", "False").lower() == "true"
IMAGE_GENERATION_INTERVAL_REAL_SECONDS = float(os.getenv("IMAGE_GENERATION_INTERVAL_REAL_SECONDS", 1800.0)) # How often to generate an image
IMAGE_GENERATION_MODEL_NAME = os.getenv("IMAGE_GENERATION_MODEL_NAME", "gemini-2.0-flash-preview-image-generation") # Or your preview model, e.g., "gemini-2.0-flash-preview-image-generation"
IMAGE_GENERATION_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "narrative_images")

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details"
SIMULACRA_PROFILES_KEY = "simulacra_profiles" # This is the old constant, SIMULACRA_KEY now points to this value.
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location" # Used in world_config and state[WORLD_TEMPLATE_DETAILS_KEY]
DEFAULT_HOME_LOCATION_NAME = "Home_01"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."

# Ensure directories exist (safe check, though loop_utils also does this)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
os.makedirs(IMAGE_GENERATION_OUTPUT_DIR, exist_ok=True) # Create image output directory
