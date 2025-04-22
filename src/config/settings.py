# src/config/settings.py
import os
from dotenv import load_dotenv

# Load variables from .env file at the project root
# Assumes .env is in the parent directory of src/
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f".env file loaded from {dotenv_path} (if exists).")

# --- API Key Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Model Constants ---
# Fetches from .env or uses default values
MODEL_GEMINI_PRO = os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-pro-latest")
MODEL_GEMINI_FLASH = os.getenv("MODEL_GEMINI_FLASH", "gemini-1.5-flash-latest")
DEBUG_MODE = False
# --- Session/App Constants ---
APP_NAME = "world_simulation_app"
USER_ID = "player_1"
SESSION_ID = "sim_session_001"

print(f"Settings loaded: Using MODEL_GEMINI_PRO={MODEL_GEMINI_PRO}, MODEL_GEMINI_FLASH={MODEL_GEMINI_FLASH}")

# Check if API key is loaded
if not GOOGLE_API_KEY:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: GOOGLE_API_KEY not found in environment or .env file.")
    print("!!! Please create a .env file with GOOGLE_API_KEY=YOUR_KEY")
    print("!!! Or set the environment variable.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")