import argparse
import asyncio
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Removed Rich console - using pure Textual

from src.simulation_async import APP_NAME, run_simulation
from src.textual_dashboard import SimulationDashboard

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
        # Silence unused parameter warnings
        _ = instance_uuid, log_dir
        return None, "event_log_disabled.jsonl"
    # Fallback unique_log_filename to avoid error later if setup_unique_logger failed
    unique_log_filename = "fallback_console_only.log" 

# --- Unique Logging Setup (File only for Textual mode) ---
logger, unique_log_filename = setup_unique_logger(
    logger_name=APP_NAME,
    file_prefix=APP_NAME, # Results in filenames like YourAppName_YYYYMMDD_HHMMSS.log
    file_level=logging.DEBUG,    # As per original file log level
    console_level=logging.CRITICAL, # Disable console logging to prevent interfering with Textual
    file_formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Original file format
    console_formatter_str='%(levelname)s: %(message)s', # Original console format
    file_mode='w' # Original file mode (overwrite if somehow the same timestamped file existed)
)

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

class TheSimulationApp(SimulationDashboard):
    """Enhanced Textual app that runs the simulation integrated with the dashboard."""
    
    def __init__(self, instance_uuid: Optional[str] = None, location_override: Optional[str] = None, mood_override: Optional[str] = None):
        # Find latest state and event files automatically
        super().__init__(state_file=None)
        self.instance_uuid = instance_uuid
        self.location_override = location_override
        self.mood_override = mood_override
        self.simulation_task: Optional[asyncio.Task] = None
        self.simulation_state_ref = None  # Reference to live simulation state
        
    def on_ready(self) -> None:
        """Called when the app is ready - start simulation and dashboard."""
        # Call parent ready to set up dashboard
        super().on_ready()
        
        # Initialize event logger
        initialize_event_logger(self.instance_uuid)
        
        # Enable INFO level for simulation modules to capture agent outputs
        logging.getLogger('src.simulation_async').setLevel(logging.INFO)
        logging.getLogger('src.core_tasks').setLevel(logging.INFO)
        logging.getLogger('src.agents').setLevel(logging.INFO)
        
        # Add startup message
        if hasattr(self, 'main_event_log') and self.main_event_log:
            self.main_event_log.add_event({
                'type': 'INFO',
                'description': f'Starting {APP_NAME} simulation...'
            })
        
        # Start the simulation as a background task
        asyncio.create_task(self._run_simulation())
    
    async def _run_simulation(self):
        """Run the simulation in the background."""
        try:
            if hasattr(self, 'main_event_log') and self.main_event_log:
                self.main_event_log.add_event({
                    'type': 'INFO',
                    'description': 'Initializing simulation...'
                })
            
            await run_simulation(
                instance_uuid_arg=self.instance_uuid,
                location_override_arg=self.location_override,
                mood_override_arg=self.mood_override,
                event_logger_instance=event_logger,
                dashboard_app=self
            )
        except asyncio.CancelledError:
            if hasattr(self, 'main_event_log') and self.main_event_log:
                self.main_event_log.add_event({
                    'type': 'WARNING',
                    'description': 'Simulation cancelled by user'
                })
        except Exception as e:
            error_msg = f"Simulation crashed: {e}"
            logger.error(error_msg, exc_info=True)
            if hasattr(self, 'main_event_log') and self.main_event_log:
                self.main_event_log.add_event({
                    'type': 'ERROR',
                    'description': error_msg
                })
    
    def action_quit(self) -> None:
        """Enhanced quit that stops simulation first."""
        super().action_quit()
    
    def update_live_simulation_data(self, state_data: dict, event_bus_size: int = 0, narration_queue_size: int = 0):
        """Update dashboard with live simulation data."""
        self.simulation_state_ref = state_data
        
        # Update panels with live data using correct attribute names and methods
        if hasattr(self, 'world_state_panel') and self.world_state_panel:
            self.world_state_panel.update_state(state_data)
        
        if hasattr(self, 'detailed_status_panel') and self.detailed_status_panel:
            self.detailed_status_panel.update_state(state_data, event_bus_size, narration_queue_size)
        
        if hasattr(self, 'agent_status_table') and self.agent_status_table:
            self.agent_status_table.update_state(state_data)
        
        if hasattr(self, 'location_table') and self.location_table:
            self.location_table.update_location_data(state_data)

async def main(instance_uuid: Optional[str] = None, location_override: Optional[str] = None, mood_override: Optional[str] = None):
    """Main entry point that runs the simulation with Textual UI."""
    app = TheSimulationApp(
        instance_uuid=instance_uuid,
        location_override=location_override,
        mood_override=mood_override
    )
    
    try:
        await app.run_async()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.shutdown()

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

    if sys.platform == "win32":
         # This policy might be needed on Windows for Rich Live display
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function (event logger will be initialized inside the app)
    asyncio.run(main(
        instance_uuid=args.instance_uuid,
        location_override=args.override_location,
        mood_override=args.override_mood
    ))