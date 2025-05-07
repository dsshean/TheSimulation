# src/config.py - Simulation Configuration and Constants

import os

# --- Core API and Model Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-2.0-flash")
SEARCH_AGENT_MODEL_NAME = os.getenv("SEARCH_AGENT_MODEL_NAME", "gemini-2.0-flash")
APP_NAME = "TheSimulationAsync"
USER_ID = "player1"

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", 1))
UPDATE_INTERVAL = float(os.getenv("UPDATE_INTERVAL", 0.1))
MAX_SIMULATION_TIME = float(os.getenv("MAX_SIMULATION_TIME", 1800.0))
MEMORY_LOG_CONTEXT_LENGTH = 10
MAX_MEMORY_LOG_ENTRIES = 500

# --- Agent Interruption / Reflection Parameters ---
AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS = float(os.getenv("AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS", 120.0))
LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS = float(os.getenv("LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS", 300.0))
INTERJECTION_COOLDOWN_SIM_SECONDS = float(os.getenv("INTERJECTION_COOLDOWN_SIM_SECONDS", 450.0))
PROB_INTERJECT_AS_SELF_REFLECTION = float(os.getenv("PROB_INTERJECT_AS_SELF_REFLECTION", 0.60))
PROB_INTERJECT_AS_NARRATIVE = float(os.getenv("PROB_INTERJECT_AS_NARRATIVE", 0.05))
AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS = float(os.getenv("AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS", 0.5))

# --- World Information Gatherer Parameters ---
WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS = float(os.getenv("WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS", 3600.0))
SIMPLE_TIMER_INTERJECTION_INTERVAL_SIM_SECONDS = float(os.getenv("SIMPLE_TIMER_INTERJECTION_INTERVAL_SIM_SECONDS", 3600.0))
MAX_WORLD_FEED_ITEMS = int(os.getenv("MAX_WORLD_FEED_ITEMS", 5))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
WORLD_CONFIG_DIR = os.path.join(BASE_DIR, "data") # World configs are in the main data dir

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location" # Used in world_config and state[WORLD_TEMPLATE_DETAILS_KEY]
DEFAULT_HOME_LOCATION_NAME = "At home"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."

# Ensure directories exist (safe check, though loop_utils also does this)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
