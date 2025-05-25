# src/config.py - Simulation Configuration and Constants

import os
from typing import Optional

# --- Core API and Model Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-2.0-flash")
SEARCH_AGENT_MODEL_NAME = os.getenv("SEARCH_AGENT_MODEL_NAME", "gemini-2.0-flash")
APP_NAME = "TheSimulationAsync"
USER_ID = "player1"
SIMULACRA_KEY = "simulacra_profiles"

# --- ADK Configuration ---
ADK_SESSION_ID = os.getenv("ADK_SESSION_ID", "simulation_session_001")
ADK_MAX_HISTORY_LENGTH = int(os.getenv("ADK_MAX_HISTORY_LENGTH", "100"))
# --- ADK Specific ---
ADK_STATE_SAVE_INTERVAL = int(os.getenv("ADK_STATE_SAVE_INTERVAL", "50")) # Number of ADK ticks between saves
# --- Social Media Configuration (Consolidated) ---
ENABLE_BLUESKY_POSTING = os.getenv("ENABLE_BLUESKY_POSTING", "False").lower() == "true"
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD")

ENABLE_TWITTER_POSTING = os.getenv("ENABLE_TWITTER_POSTING", "False").lower() == "true"
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

SOCIAL_POST_HASHTAGS = os.getenv("SOCIAL_POST_HASHTAGS", "#TheSimulation #AI #DigitalTwin #ProceduralStorytelling")
SOCIAL_POST_TEXT_LIMIT = int(os.getenv("SOCIAL_POST_TEXT_LIMIT", "300"))

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", "1"))
UPDATE_INTERVAL = float(os.getenv("UPDATE_INTERVAL", "0.1"))
MAX_SIMULATION_TIME = float(os.getenv("MAX_SIMULATION_TIME", "9996000.0"))
MAX_SIMULATION_TICKS = int(os.getenv("MAX_SIMULATION_TICKS", "1000"))
MEMORY_LOG_CONTEXT_LENGTH = 10
MAX_MEMORY_LOG_ENTRIES = 500

# --- Agent Self-Reflection Parameters ---
AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS = float(os.getenv("AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS", "120.0"))
LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS = float(os.getenv("LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS", "300.0"))
INTERJECTION_COOLDOWN_SIM_SECONDS = float(os.getenv("INTERJECTION_COOLDOWN_SIM_SECONDS", "450.0"))
PROB_INTERJECT_AS_SELF_REFLECTION = float(os.getenv("PROB_INTERJECT_AS_SELF_REFLECTION", "0.60"))
AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS = float(os.getenv("AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS", "0.5"))

# --- Dynamic Interruption Task Parameters ---
DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS = float(os.getenv("DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS", "5.0"))
DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS = float(os.getenv("DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS", "600.0"))
DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION = float(os.getenv("DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION", "0.35"))
DYNAMIC_INTERRUPTION_MIN_PROB = float(os.getenv("DYNAMIC_INTERRUPTION_MIN_PROB", "0.005"))
DYNAMIC_INTERRUPTION_MAX_PROB_CAP = float(os.getenv("DYNAMIC_INTERRUPTION_MAX_PROB_CAP", "0.25"))
MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK = float(os.getenv("MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK", "60.0"))

# --- World Information Gatherer Parameters ---
WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS = float(os.getenv("WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS", "3600.0"))
MAX_WORLD_FEED_ITEMS = int(os.getenv("MAX_WORLD_FEED_ITEMS", "5"))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
WORLD_CONFIG_DIR = os.path.join(BASE_DIR, "data")

# --- Random Seed ---
RANDOM_SEED_VALUE = os.getenv("RANDOM_SEED_VALUE")
RANDOM_SEED: Optional[int] = int(RANDOM_SEED_VALUE) if RANDOM_SEED_VALUE and RANDOM_SEED_VALUE.isdigit() else None

# --- Narrative Image Generation Parameters ---
ENABLE_NARRATIVE_IMAGE_GENERATION = os.getenv("ENABLE_NARRATIVE_IMAGE_GENERATION", "False").lower() == "true"
IMAGE_GENERATION_INTERVAL_REAL_SECONDS = float(os.getenv("IMAGE_GENERATION_INTERVAL_REAL_SECONDS", "1800.0"))
IMAGE_GENERATION_MODEL_NAME = os.getenv("IMAGE_GENERATION_MODEL_NAME", "imagen-3.0-generate-002")
IMAGE_GENERATION_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "narrative_images")
BLUESKY_MAX_IMAGE_SIZE_BYTES = int(os.getenv("BLUESKY_MAX_IMAGE_SIZE_BYTES", str(976 * 1024))) # 976KB default

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
INITIAL_LOCATION_DEFINITIONS_KEY = "initial_location_definitions"
LOCATION_KEY = "location"
DEFAULT_HOME_LOCATION_NAME = "Home_01"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."
WORLD_FEEDS_KEY = "world_feeds"
WORLD_CONTEXT_KEY = "world_context"

# Ensure directories exist
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
os.makedirs(IMAGE_GENERATION_OUTPUT_DIR, exist_ok=True)
