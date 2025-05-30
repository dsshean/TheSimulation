import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

console = Console()

from src.simulation_async import APP_NAME, run_simulation

# Assuming logger_config.py is in the same directory or accessible via PYTHONPATH
# and contains the setup_unique_logger function.
try:
    from src.logger_config import (  # MODIFIED: Added setup_event_logger
        setup_event_logger, setup_unique_logger)
except ImportError:
    print("ERROR: logger_config.py not found. Please ensure it exists and is in your PYTHONPATH.")
    print("Using basic fallback logging to console only.")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(APP_NAME if 'APP_NAME' in globals() else "FallbackLogger")
    # Fallback for event logger setup
    def setup_event_logger(instance_uuid=None, log_dir="logs/events"): # Dummy fallback
        print(f"WARNING: setup_event_logger not found in logger_config. Event logging to JSONL will be disabled.")
        return None, "event_log_disabled.jsonl"
    # Fallback unique_log_filename to avoid error later if setup_unique_logger failed
    unique_log_filename = "fallback_console_only.log" 

# --- Unique Logging Setup ---
logger, unique_log_filename = setup_unique_logger(
    logger_name=APP_NAME,
    file_prefix=APP_NAME, # Results in filenames like YourAppName_YYYYMMDD_HHMMSS.log
    file_level=logging.DEBUG,    # As per original file log level
    console_level=logging.WARNING, # As per original console log level
    file_formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Original file format
    console_formatter_str='%(levelname)s: %(message)s', # Original console format
    file_mode='w' # Original file mode (overwrite if somehow the same timestamped file existed)
)
logger.info(f"--- Application Start ({APP_NAME}) --- Logging to: {unique_log_filename}")

# Disable verbose logging for third-party modules
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google_adk').setLevel(logging.WARNING) 
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('geopy').setLevel(logging.WARNING)

# --- Event Logger Setup ---
event_logger: Optional[logging.Logger] = None # Initialize to None, type hint for clarity
event_log_filename: str = "event_log_not_initialized.jsonl"
 # args will be parsed below, we'll use args.instance_uuid for the event logger
 # The actual setup will happen after arg parsing.

def initialize_event_logger(instance_uuid_for_log: Optional[str] = None):
    global event_logger, event_log_filename
    event_logger, event_log_filename = setup_event_logger(instance_uuid=instance_uuid_for_log, log_dir="logs/events")
    logger.info(f"Structured event logging to: {event_log_filename if event_logger else 'DISABLED'}")
# --- Entry Point ---
if __name__ == "__main__":
    # Use APP_NAME imported from simulation_async
    parser = argparse.ArgumentParser(description=f"Run {APP_NAME} simulation instance.")
    parser.add_argument(
        "--instance-uuid", type=str,
        help="Specify the UUID of the simulation instance to load. If omitted, the latest instance is loaded.",
        default=None
    )
    # --- ADDED: Override Arguments ---
    parser.add_argument(
        "--override-location", type=str,
        help="Override the primary location (e.g., 'London, UK', 'Mars Colony 7').",
        default=None
    )
    parser.add_argument(
        "--override-mood", type=str,
        help="Override the initial mood for ALL simulacra (e.g., 'happy', 'anxious').",
        default=None
    )
    # --- END ADDED ---
    args = parser.parse_args()

    # Initialize the event logger now that we have args.instance_uuid
    initialize_event_logger(args.instance_uuid)

    try:
        if sys.platform == "win32":
             # This policy might be needed on Windows for Rich Live display
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # --- MODIFIED: Pass override arguments to run_simulation ---
        asyncio.run(run_simulation(
            instance_uuid_arg=args.instance_uuid,
            location_override_arg=args.override_location,
            mood_override_arg=args.override_mood,
            event_logger_instance=event_logger # Pass the event logger
        ))
        # --- END MODIFIED ---
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        console.print("\n[orange_red1]Simulation interrupted.[/]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main execution block: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False) # Show traceback using Rich
    finally:
        logging.shutdown() # Ensure logs are flushed
        console.print("Application finished.")
