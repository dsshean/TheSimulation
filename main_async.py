import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

console = Console()

from src.simulation_async import (APP_NAME,  # Import APP_NAME too
                                      run_simulation)
# --- Logging Setup ---
# Keep basic logging setup here at the entry point
log_filename = "simulation_async.log"
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to capture all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w' # Overwrite log file on each run
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING) # Show warnings and above on console
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)
logger = logging.getLogger(__name__)
logger.info(f"--- Application Start (main_async.py) --- Logging to: {log_filename}")

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

    try:
        if sys.platform == "win32":
             # This policy might be needed on Windows for Rich Live display
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # --- MODIFIED: Pass override arguments to run_simulation ---
        asyncio.run(run_simulation(
            instance_uuid_arg=args.instance_uuid,
            location_override_arg=args.override_location,
            mood_override_arg=args.override_mood
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

