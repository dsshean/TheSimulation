#!/usr/bin/env python3
"""
Simple version of main_async.py without Textual dashboard.
Based on the "main branch upgraded to adk V1.0" commit.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


from src.simulation_async import APP_NAME, run_simulation

# Setup logging
try:
    from src.logger_config import setup_event_logger, setup_unique_logger
except ImportError:
    print("ERROR: logger_config.py not found. Using basic logging.")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(APP_NAME if 'APP_NAME' in globals() else "TheSimulation")
    def setup_event_logger(instance_uuid=None, log_dir="logs/events"):
        print("Event logging disabled (logger_config not found)")
        return None, "event_log_disabled.jsonl"
    unique_log_filename = "fallback_console_only.log"
else:
    # Proper logging setup
    logger, unique_log_filename = setup_unique_logger(
        logger_name=APP_NAME,
        file_prefix=APP_NAME,
        file_level=logging.DEBUG,
        console_level=logging.WARNING,
        file_formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        console_formatter_str='%(levelname)s: %(message)s',
        file_mode='w'
    )
    logger.info(f"--- Application Start ({APP_NAME}) --- Logging to: {unique_log_filename}")

# Disable verbose third-party logging
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google_adk').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

event_logger: Optional[logging.Logger] = None
event_log_filename: str = "event_log_not_initialized.jsonl"

def initialize_event_logger(instance_uuid_for_log: Optional[str] = None):
    global event_logger, event_log_filename
    event_logger, event_log_filename = setup_event_logger(instance_uuid=instance_uuid_for_log, log_dir="logs/events")
    logger.info(f"Structured event logging to: {event_log_filename if event_logger else 'DISABLED'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run {APP_NAME} simulation instance.")
    parser.add_argument(
        "--instance-uuid", type=str,
        help="Specify the UUID of the simulation instance to load. If omitted, the latest instance is loaded.",
        default=None
    )
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
    args = parser.parse_args()

    # Initialize event logger
    initialize_event_logger(args.instance_uuid)

    print(f"[bold green]Starting {APP_NAME} Simulation[/bold green]")
    print(f"Instance UUID: {args.instance_uuid or 'Latest'}")
    if args.override_location:
        print(f"Location Override: {args.override_location}")
    if args.override_mood:
        print(f"Mood Override: {args.override_mood}")
    
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(run_simulation(
            instance_uuid_arg=args.instance_uuid,
            location_override_arg=args.override_location,
            mood_override_arg=args.override_mood,
            event_logger_instance=event_logger
        ))
        
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        print("\n[orange_red1]Simulation interrupted.[/]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        # print_exception(show_locals=False)
    finally:
        logging.shutdown()
        print("Application finished.")