import argparse
import asyncio
import logging
import os # Keep os for makedirs
import sys

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
console = Console()
# Ensure src directory is in Python path or use relative import if appropriate
from src.simulation_async import run_simulation, APP_NAME # Import APP_NAME too

# --- Logging Setup ---
# Keep basic logging setup here at the entry point
log_filename = "simulation_async.log"
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to capture all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
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
    args = parser.parse_args()

    try:
        if sys.platform == "win32":
             # This policy might be needed on Windows for Rich Live display
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_simulation(instance_uuid_arg=args.instance_uuid))
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        console.print("\n[orange_red1]Simulation interrupted.[/]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main execution block: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False)
    finally:
        logging.shutdown()
        console.print("Application finished.")
