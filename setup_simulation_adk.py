# c:\Users\dshea\Desktop\TheSimulation\src\setup_simulation_adk.py
import asyncio
import json
import logging
import os
import random  # Added for sim_id generation
import string  # Added for sim_id generation
import uuid
from datetime import datetime, timezone

# For user input, similar to setup_simulation.py
from rich.console import Console
from rich.panel import Panel  # For displaying config summary
from rich.rule import Rule  # For section breaks

from src.config import APP_NAME
# Import from the new ADK-enabled life generator
from src.generation.life_generator_adk import generate_new_simulacra_background
# Import the correct logging setup function and APP_NAME for logger naming
from src.logger_config import setup_unique_logger
from src.state_loader import parse_location_string  # Import from state_loader

# Setup logging
logger, log_filename = setup_unique_logger(
    logger_name=APP_NAME + "_SetupADK",
    file_prefix="setup_simulation_adk",
    console_level=logging.CRITICAL # Set console level high to minimize output
)
logger.info(f"Logging initialized for setup_simulation_adk.py. Log file: {log_filename}")
logging.getLogger("google.adk").setLevel(logging.WARNING)
# --- Configuration ---
NUMBER_OF_SIMULACRA_TO_GENERATE = 1 # Or however many you want

console = Console() # For get_user_input

def get_user_input(prompt: str, valid_options: list = None, allow_empty: bool = False, input_type: type = str, default_value: any = None) -> any:
    """Gets validated user input."""
    while True:
        default_prompt = f" (default: {default_value})" if default_value is not None else ""
        user_input = console.input(f"[bold cyan]{prompt}[/bold cyan]{default_prompt}: ").strip()
        
        if not user_input and default_value is not None:
            return default_value
        if not user_input and not allow_empty:
            console.print("[bold red]Input cannot be empty.[/bold red]")
            continue
        # Case-insensitive check for valid_options
        if valid_options and (user_input.lower() not in [str(opt).lower() for opt in valid_options]):
            console.print(f"[bold red]Invalid option. Please choose from: {', '.join(map(str,valid_options))}.[/bold red]")
            continue
        try:
            return input_type(user_input)
        except ValueError:
            console.print(f"[bold red]Invalid input type. Expected {input_type.__name__}.[/bold red]")

async def setup_new_simulation_environment():
    """
    Sets up a new simulation environment by:
    1. Getting configuration from the user.
    2. Saving the world configuration.
    3. Generating a specified number of simulacra for that world configuration.
    """
    console.print(Rule("ADK Simulation Setup", style="bold blue"))
    logger.info("Starting new ADK simulation environment setup...")

    instance_uuid = str(uuid.uuid4())
    console.print(f"Generated Simulation Instance UUID: [cyan]{instance_uuid}[/cyan]")

    # Replicate the config_data structure from setup_simulation.py
    config_data = {
        "world_instance_uuid": instance_uuid,
        "world_type": None,
        "sub_genre": None,
        "description": None,
        "rules": {
            "allow_teleportation": False,
            "time_progression_rate": 1.0,
            "weather_effects_travel": True,
            "historical_date": None
        },
        "location": {
            "city": None, "state": None, "country": None,
            "coordinates": {"latitude": None, "longitude": None}
        },
        "setup_timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z"
    }

    # --- Get User Input for Configuration ---
    console.print(Rule("World Configuration", style="yellow"))
    sim_type_input = get_user_input(
        "Select World Type (RealWorld, SciFi, Fantasy, Custom) [Default: RealWorld]:",
        ["RealWorld", "SciFi", "Fantasy", "Custom"],
        default_value="RealWorld"
    ).lower()

    type_mapping = {"realworld": "real", "scifi": "scifi", "fantasy": "fantasy", "custom": "custom"}
    config_data["world_type"] = type_mapping.get(sim_type_input) # Use .get with exact key

    if config_data["world_type"] == "real":
        realworld_type = get_user_input(
            "Select Real World Sub-Genre (RealTime, Historical) [Default: RealTime]:",
            ["RealTime", "Historical"],
            default_value="RealTime"
        )
        config_data["sub_genre"] = realworld_type.lower()

        if config_data["sub_genre"] == "realtime":
            location_str = get_user_input(
                "Enter Primary Location (e.g., 'Asheville, NC', 'Tokyo, Japan') [Default: New York, New York]:",
                allow_empty=True,
                default_value="New York, New York")
            config_data["location"] = parse_location_string(location_str)
            config_data["description"] = f"A simulation mirroring the current real world in {location_str}. Standard physics apply."
            config_data["rules"]["time_progression_rate"] = 1.0
        else: # Historical
            period_desc = get_user_input("Enter Historical Period/Location (e.g., 'Ancient Rome, 100 CE', 'Victorian London'):", allow_empty=False)
            config_data["location"]["city"] = period_desc # Store full desc as city for simplicity here
            config_data["description"] = f"An ancestral simulation set in {period_desc}. Historical context applies."
            rate_input = get_user_input("Enter time progression rate (e.g., 1.0 for real-time, leave empty for 1.0):", allow_empty=True, input_type=float, default_value=1.0)
            config_data["rules"]["time_progression_rate"] = rate_input
            start_date = get_user_input("Enter optional historical start date (YYYY-MM-DD or leave empty):", allow_empty=True, default_value=None)
            if start_date:
                try:
                    datetime.strptime(start_date, "%Y-%m-%d")
                    config_data["rules"]["historical_date"] = f"{start_date}T00:00:00Z"
                except ValueError:
                    logger.warning(f"Invalid date format '{start_date}'. Ignoring historical start date.")
                    console.print(f"[yellow]Warning:[/yellow] Invalid date format '{start_date}'. Ignoring.")

    elif config_data["world_type"] in ["scifi", "fantasy", "custom"]:
        sub_genre_desc = get_user_input(f"Enter a Sub-Genre for {sim_type_input} (e.g., 'Space Opera', 'High Fantasy', 'Cyberpunk'):", allow_empty=False)
        config_data["sub_genre"] = sub_genre_desc
        description = get_user_input(f"Enter a brief description for your {sim_type_input} - {sub_genre_desc} setting:", allow_empty=False)
        config_data["description"] = description
        location_concept = get_user_input("Enter a primary location or region name (e.g., 'Starbase Alpha', 'Elvenwood Forest'):", allow_empty=False)
        config_data["location"]["city"] = location_concept # Store concept as city
        teleport_input = get_user_input("Allow teleportation? (yes/no) [Default: no]:", ["yes", "no"], default_value="no")
        config_data["rules"]["allow_teleportation"] = (teleport_input.lower() == "yes")
        rate_input = get_user_input("Enter time progression rate (e.g., 1.0 for real-time speed, leave empty for 1.0):", allow_empty=True, input_type=float, default_value=1.0)
        config_data["rules"]["time_progression_rate"] = rate_input

    # Ensure description is set if somehow missed (e.g., custom path without explicit ask)
    if not config_data["description"]:
        config_data["description"] = get_user_input(
            f"Enter a brief description for your {config_data['world_type']} world:",
            allow_empty=True,
            default_value=f"A generic {config_data['world_type']} world."
        )

    # Simulacra generation specific inputs
    console.print(Rule("Simulacra Configuration", style="yellow"))
    number_of_simulacra = get_user_input("How many simulacra to generate for this world? [Default: 1]:", input_type=int, default_value=1)
    
    # Determine if real-world context should be allowed based on world type for life generation
    # This matches the logic in setup_simulation.py where it's implicitly tied to "real" world type.
    allow_real_context_for_life_generation = (config_data["world_type"] == "real")
    if config_data["world_type"] != "real": # Ask only if not a "real" world, as for "real" it's implied true
        allow_real_context_input = get_user_input(
            f"Allow ADK Google Search for context during life generation for this '{config_data['world_type']}' world? (yes/no) [Default: no]:",
            valid_options=["yes", "no"], default_value="no"
        ).lower()
        allow_real_context_for_life_generation = allow_real_context_input == "yes"
    
    gender_preference_input = get_user_input(
        "Enter gender preference for simulacra (e.g., male, female, any) [Default: any]:",
        valid_options=["male", "female", "any"], default_value="any"
    ).lower()
    if gender_preference_input == "any":
        gender_preference_input = None # Pass None if "any" for life_generator_adk

    logger.info(f"World Instance UUID: {instance_uuid}")
    logger.info(f"World Type: {config_data['world_type']}, Sub-Genre: {config_data['sub_genre']}, Description: {config_data['description']}")
    logger.info(f"Allow real context for life gen: {allow_real_context_for_life_generation}")
    logger.info(f"Simulacra gender preference: {gender_preference_input or 'any'}")

    # Save the world configuration (similar to setup_simulation.py)
    world_config_dir = "world_configurations"
    os.makedirs(world_config_dir, exist_ok=True)
    world_config_filename = f"{world_config_dir}/world_config_{instance_uuid}.json"
    # config_data already holds all the necessary fields
    with open(world_config_filename, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"World configuration saved to: {world_config_filename}")
    console.print(f"\nSimulation config saved successfully to: [green]{world_config_filename}[/green]")
    console.print(Panel(json.dumps(config_data, indent=2), title="World Config Summary", border_style="green"))

    # Generate Simulacra for the new world
    if instance_uuid:
        console.print(Rule("Simulacra Generation", style="yellow"))
        if not os.getenv("GOOGLE_API_KEY"):
            console.print("[bold yellow]Warning:[/bold yellow] GOOGLE_API_KEY environment variable not set. LLM calls will likely fail.")
            logger.warning("GOOGLE_API_KEY environment variable not set.")
            proceed = get_user_input("Proceed without GOOGLE_API_KEY? (yes/no) [Default: no]:", ["yes", "no"], default_value="no")
            if proceed.lower() != 'yes':
                console.print("Aborting simulacra generation.")
                return

        if number_of_simulacra <= 0:
            console.print("Number of simulacra must be positive. Skipping generation.")
            return
            
        logger.info(f"Generating {number_of_simulacra} simulacra for world UUID: {instance_uuid}...")
        console.print(f"Attempting to generate [bold]{number_of_simulacra}[/bold] simulacra...")

        sim_ids_for_tasks = [] # Initialize the list to store sim_ids
        results = [] # Store results or exceptions

        for i in range(number_of_simulacra):
            sim_id = f"sim_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
            sim_num_display = i + 1
            console.print(Rule(f"Generating Simulacra {sim_num_display}/{number_of_simulacra} (ID: {sim_id})", style="magenta"))
            logger.info(f"--- Generating Simulacra {i+1}/{number_of_simulacra} (SimID: {sim_id}) ---")
            sim_ids_for_tasks.append(sim_id) # Store the sim_id

            try:
                # Call and await the generation function directly
                result = await generate_new_simulacra_background(
                    sim_id=sim_id,
                    world_instance_uuid=instance_uuid,
                    world_type=config_data["world_type"],
                    world_description=config_data["description"],
                    allow_real_context=allow_real_context_for_life_generation,
                    gender_preference=gender_preference_input,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Exception during sequential generation of simulacra {sim_num_display}: {e}", exc_info=True)
                results.append(e) # Store the exception to be handled like gather does

        console.print(Rule("Simulacra Generation Complete", style="blue"))
        success_count = 0
        fail_count = 0
        for i, result in enumerate(results):
            sim_num_display = i + 1 # For display
            try:
                if isinstance(result, Exception):
                    raise result # Re-raise if gather caught it
                elif result is None:
                    raise ValueError("Generation returned None") # Treat None as failure
                else:
                    success_count += 1
                    sim_id_generated = result.get("sim_id", f"Unknown_ID_{sim_num_display}")
                    persona_name = result.get("persona_details", {}).get("Name", "Unknown Name")
                    logger.info(f"Successfully generated life summary for SimID: {sim_id_generated}, Persona Name: {persona_name}")
                    console.print(f"[green]✔ Simulacra {sim_num_display} ({sim_id_generated} - {persona_name}) generated successfully.[/green]")
            except Exception as e:
                # This block will now primarily catch exceptions stored in `results`
                # or if the result itself was an error not caught by the try/except in the loop above.
                fail_count += 1
                # The robust sim_id retrieval is already in place from previous changes
                failed_sim_id_str = sim_ids_for_tasks[i] if i < len(sim_ids_for_tasks) else f"TaskIndex_{sim_num_display}"
                logger.error(f"Simulacra {sim_num_display} (SimID: {failed_sim_id_str}) generation failed: {e}", exc_info=e)
                console.print(f"[bold red]Simulacra {sim_num_display} (SimID: {failed_sim_id_str}) generation failed:[/bold red] {e}")

        console.print(f"\nGeneration Summary: {success_count} succeeded, {fail_count} failed.")
    else:
        logger.warning("Skipping simulacra generation as world instance creation failed.")

    console.print(Rule("ADK Setup Complete", style="bold blue"))
    logger.info("ADK Simulation environment setup process finished.")


if __name__ == "__main__":
    # To run the setup:
    try:
        asyncio.run(setup_new_simulation_environment())
    except KeyboardInterrupt:
        console.print("\nSetup interrupted by user.")
        logger.info("Setup interrupted by user.")
    except Exception as e:
        logger.critical(f"A critical error occurred during ADK setup: {e}", exc_info=True)
        console.print(f"\n[bold red]A critical error occurred during ADK setup:[/bold red] {e}")
