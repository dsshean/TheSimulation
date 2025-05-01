import json
import os
import uuid
import asyncio # Added for async operations
import logging # Added for logging within generator
import random # Added for sim_id generation
import string # Added for sim_id generation
from datetime import datetime
from typing import Optional # Added for type hinting

# --- Import the generator function ---
# Ensure this path is correct relative to your project structure
try:
    from src.generation.life_generator import generate_new_simulacra_background
except ImportError:
    print("ERROR: Could not import 'generate_new_simulacra_background'.")
    print("Ensure 'src/generation/life_generator.py' exists and is importable.")
    generate_new_simulacra_background = None # Set to None to prevent runtime errors later

# --- Rich Console ---
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# --- Basic Logging Setup ---
# Configure logging to capture info from the generator
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("setup_simulation.log"),
                              logging.StreamHandler()]) # Log to file and console stream
logger = logging.getLogger(__name__)


def get_user_input(prompt: str, valid_options: list = None, allow_empty: bool = False, input_type: type = str) -> any:
    """Gets validated user input with type checking."""
    while True:
        user_input_str = input(f"{prompt} ").strip()
        if not user_input_str and not allow_empty:
            print("Input cannot be empty.")
            continue

        # Handle type conversion
        try:
            if input_type == int:
                if not user_input_str and allow_empty:
                    return None # Or a default int if appropriate
                user_input = int(user_input_str)
            elif input_type == float:
                 if not user_input_str and allow_empty:
                    return None # Or a default float
                 user_input = float(user_input_str)
            else: # Default to string
                user_input = user_input_str
        except ValueError:
            print(f"Invalid input type. Please enter a valid {input_type.__name__}.")
            continue

        # Handle validation options if provided
        if valid_options:
            # Case-insensitive check for string options
            if isinstance(user_input, str):
                if user_input.lower() in [str(opt).lower() for opt in valid_options]:
                    # Return the matching option in its original case (for consistency)
                    for opt in valid_options:
                        if user_input.lower() == str(opt).lower():
                            return opt
                elif allow_empty and not user_input:
                     return "" # Return empty string if allowed
                else:
                    print(f"Invalid option. Please choose from: {', '.join(map(str, valid_options))}")
                    continue # Re-prompt if invalid option
            else: # For non-string types with options (e.g., int range - though not used here yet)
                 if user_input in valid_options:
                     return user_input
                 else:
                     print(f"Invalid option. Please choose from: {', '.join(map(str, valid_options))}")
                     continue
        else:
            # No options, just return the type-converted input
            return user_input


def parse_location_string(location_str: str) -> dict:
    """Attempts to parse 'City, State' or 'City, Country'."""
    parts = [p.strip() for p in location_str.split(',')]
    location_data = {"city": None, "state": None, "country": None, "coordinates": {"latitude": None, "longitude": None}}
    if len(parts) >= 1:
        location_data["city"] = parts[0]
    if len(parts) == 2:
        # Basic check: if 2 letters, assume state (US/Canada), otherwise country
        if len(parts[1]) == 2 and parts[1].isalpha():
             location_data["state"] = parts[1].upper()
             location_data["country"] = "United States" # Default assumption, could be improved
        else:
             location_data["country"] = parts[1]
    elif len(parts) == 3: # Assume City, State, Country
        location_data["state"] = parts[1].upper()
        location_data["country"] = parts[2]

    # If only city was given, try to guess country based on common knowledge or leave null
    if location_data["city"] and not location_data["country"]:
         # Simple examples, could be expanded or use a library
         if location_data["city"].lower() in ["london", "paris", "tokyo"]:
             location_data["country"] = {"london": "United Kingdom", "paris": "France", "tokyo": "Japan"}[location_data["city"].lower()]
         # else: leave country as None

    # Ensure city is set if only country was somehow derived
    if not location_data["city"] and location_str:
        location_data["city"] = location_str # Fallback to using the whole string as city/region name

    return location_data

# --- Make main async ---
async def main():
    """Guides the user through setting up simulation parameters and generating simulacra."""
    console.print(Rule("Simulation Setup", style="bold blue"))

    instance_uuid = str(uuid.uuid4())
    console.print(f"Generated Simulation Instance UUID: [cyan]{instance_uuid}[/cyan]")

    # Structure matching world_config.json
    config_data = {
        "world_instance_uuid": instance_uuid,
        "world_type": None, # "real", "scifi", "fantasy", "custom"
        "sub_genre": None, # "realtime", "historical", "alternate_history", "space_opera", "high_fantasy", etc.
        "description": None,
        "rules": {
            "allow_teleportation": False,
            "time_progression_rate": 1.0,
            "weather_effects_travel": True,
            "historical_date": None # ISO format string or null
        },
        "location": { # Primary location details
            "city": None,
            "state": None,
            "country": None,
            "coordinates": { # Keep null for now
                "latitude": None,
                "longitude": None
            }
        },
        "setup_timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }

    # --- World Config Input ---
    console.print(Rule("World Configuration", style="yellow"))
    sim_type_input = get_user_input(
        "Select World Type (RealWorld, SciFi, Fantasy, Custom):",
        ["RealWorld", "SciFi", "Fantasy", "Custom"]
    )

    type_mapping = {"realworld": "real", "scifi": "scifi", "fantasy": "fantasy", "custom": "custom"}
    config_data["world_type"] = type_mapping.get(sim_type_input.lower())

    if config_data["world_type"] == "real":
        realworld_type = get_user_input(
            "Select Real World Sub-Genre (RealTime, Historical):",
            ["RealTime", "Historical"]
        )
        config_data["sub_genre"] = realworld_type.lower()

        if config_data["sub_genre"] == "realtime":
            location_str = get_user_input("Enter Primary Location (e.g., 'Asheville, NC', 'Tokyo, Japan'):")
            config_data["location"] = parse_location_string(location_str)
            config_data["description"] = f"A simulation mirroring the current real world in {location_str}. Standard physics apply."
            config_data["rules"]["time_progression_rate"] = 1.0
        else: # Historical
            period_desc = get_user_input("Enter Historical Period/Location (e.g., 'Ancient Rome, 100 CE', 'Victorian London'):")
            config_data["location"]["city"] = period_desc
            config_data["description"] = f"An ancestral simulation set in {period_desc}. Historical context applies."
            rate_input = get_user_input("Enter time progression rate (e.g., 1.0 for real-time, 3600 for 1hr/sec, leave empty for 1.0):", allow_empty=True, input_type=float)
            config_data["rules"]["time_progression_rate"] = rate_input if rate_input is not None else 1.0
            start_date = get_user_input("Enter optional historical start date (YYYY-MM-DD or leave empty):", allow_empty=True)
            if start_date:
                 try:
                     # Validate date format briefly
                     datetime.strptime(start_date, "%Y-%m-%d")
                     config_data["rules"]["historical_date"] = f"{start_date}T00:00:00Z"
                 except ValueError:
                     logger.warning(f"Invalid date format '{start_date}'. Ignoring historical start date.")
                     console.print(f"[yellow]Warning:[/yellow] Invalid date format '{start_date}'. Ignoring.")

    elif config_data["world_type"] in ["scifi", "fantasy", "custom"]:
        sub_genre_desc = get_user_input(f"Enter a Sub-Genre for {sim_type_input} (e.g., 'Space Opera', 'High Fantasy', 'Cyberpunk'):")
        config_data["sub_genre"] = sub_genre_desc
        description = get_user_input(f"Enter a brief description for your {sim_type_input} - {sub_genre_desc} setting:")
        config_data["description"] = description
        location_concept = get_user_input("Enter a primary location or region name (e.g., 'Starbase Alpha', 'Elvenwood Forest'):")
        config_data["location"]["city"] = location_concept
        teleport = get_user_input("Allow teleportation? (yes/no):", ["yes", "no"])
        config_data["rules"]["allow_teleportation"] = (teleport.lower() == "yes")
        rate_input = get_user_input("Enter time progression rate (e.g., 1.0 for real-time speed, leave empty for 1.0):", allow_empty=True, input_type=float)
        config_data["rules"]["time_progression_rate"] = rate_input if rate_input is not None else 1.0

    # --- Save the world config data ---
    output_dir = "data"
    output_filename = f"world_config_{instance_uuid}.json"
    output_file = os.path.join(output_dir, output_filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(config_data, f, indent=2)
        console.print(f"\nSimulation config saved successfully to: [green]{output_file}[/green]")
        console.print(Panel(json.dumps(config_data, indent=2), title="World Config Summary", border_style="green"))

    except IOError as e:
        logger.error(f"Error saving config file: {e}", exc_info=True)
        console.print(f"\n[bold red]Error saving config file:[/bold red] {e}")
        return # Stop if world config cannot be saved
    except Exception as e:
        logger.error(f"An unexpected error occurred during config save: {e}", exc_info=True)
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
        return

    # --- Simulacra Generation ---
    console.print(Rule("Simulacra Generation", style="yellow"))

    # Check if generator function was imported
    if generate_new_simulacra_background is None:
        console.print("[bold red]Cannot proceed with simulacra generation: generator function not found.[/bold red]")
        return

    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
         console.print("[bold yellow]Warning:[/bold yellow] GOOGLE_API_KEY environment variable not set. LLM calls will likely fail.")
         logger.warning("GOOGLE_API_KEY environment variable not set.")
         # Optionally, ask the user if they want to proceed anyway
         proceed = get_user_input("Proceed without GOOGLE_API_KEY? (yes/no):", ["yes", "no"])
         if proceed.lower() != 'yes':
             console.print("Aborting simulacra generation.")
             return

    num_simulacra = get_user_input("How many simulacra to generate for this world?", input_type=int)

    if num_simulacra <= 0:
        console.print("Number of simulacra must be positive. Skipping generation.")
        return

    console.print(f"Attempting to generate [bold]{num_simulacra}[/bold] simulacra...")

    generation_tasks = []
    for i in range(num_simulacra):
        # Generate a simple unique ID for the simulacra
        sim_id = f"sim_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
        console.print(f"\n--- Preparing generation for Simulacra {i+1}/{num_simulacra} (ID: {sim_id}) ---")

        # Determine if real-world context should be allowed based on world type
        allow_real_context = (config_data["world_type"] == "real")

        # Create the async task for generation
        task = generate_new_simulacra_background(
            sim_id=sim_id,
            world_instance_uuid=instance_uuid,
            world_type=config_data["world_type"], # Pass world type for context
            world_description=config_data["description"],
            allow_real_context=allow_real_context # Pass the flag
            # age_range uses default (18-45)
        )
        generation_tasks.append(task)

    # Run all generation tasks concurrently
    console.print(f"\n[bold]Starting generation for {len(generation_tasks)} simulacra...[/bold] (This may take some time)")
    results = await asyncio.gather(*generation_tasks, return_exceptions=True)
    console.print(Rule("Simulacra Generation Complete", style="bold blue"))

    # Process results
    success_count = 0
    fail_count = 0
    for i, result in enumerate(results):
        sim_num = i + 1
        if isinstance(result, Exception):
            fail_count += 1
            logger.error(f"Simulacra {sim_num} generation failed with exception: {result}", exc_info=result)
            console.print(f"[bold red]Simulacra {sim_num} generation failed:[/bold red] {result}")
        elif result is None:
            fail_count += 1
            logger.error(f"Simulacra {sim_num} generation returned None (check logs for details).")
            console.print(f"[bold red]Simulacra {sim_num} generation failed (returned None).[/bold red]")
        else:
            # Result should be the life_data dictionary
            success_count += 1
            sim_id_generated = result.get("simulacra_id", f"Unknown_ID_{sim_num}")
            persona_name = result.get("persona_details", {}).get("Name", "Unknown Name")
            console.print(f"[green]✔ Simulacra {sim_num} ({sim_id_generated} - {persona_name}) generated successfully.[/green]")
            # Life summary file saving is handled within generate_new_simulacra_background

    console.print(f"\nGeneration Summary: {success_count} succeeded, {fail_count} failed.")
    console.print(Rule("Setup Complete", style="bold blue"))


if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
    except Exception as e:
        logger.critical(f"A critical error occurred during setup: {e}", exc_info=True)
        console.print(f"\n[bold red]A critical error occurred:[/bold red] {e}")

