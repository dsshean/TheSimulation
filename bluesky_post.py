import json
import uuid
import re
import os
import glob
from io import BytesIO
import argparse # Import argparse for command-line arguments
from typing import Optional, List, Dict, Any # Added import for Optional, List, Dict, Any type hints
from PIL import Image # For image compression
from dotenv import load_dotenv
from rich.console import Console # Added for console output in script
from atproto import Client, models as atproto_models # Aliased import
import google.generativeai as genai

# --- Configuration ---
WORLD_SIMULATION_UUID = None # Will be loaded dynamically
MAX_POST_LENGTH = 300 # Bluesky's character limit (graphemes approx) - align with SOCIAL_POST_TEXT_LIMIT
IMAGE_ACTUAL_DIR = "" # Will be set to the correct data/narrative_images path

# Constants for Bluesky posting (mirroring parts of your project's config.py)
BLUESKY_MAX_IMAGE_SIZE_BYTES = 976 * 1024  # 976KB
SOCIAL_POST_HASHTAGS = os.getenv("SOCIAL_POST_HASHTAGS", "#TheSimulation #AI #DigitalTwin #ProceduralStorytelling")
# MAX_POST_LENGTH is used for the text limit

# --- Bluesky API Client (initialized later if needed) ---
bluesky_client = None

# --- Google AI Clients (initialized later if needed) ---
text_gen_client = None
img_gen_client = None

console = Console() # Initialize Rich console for script output

def find_latest_file_in_dir(directory: str, pattern: str) -> Optional[str]:
    """Finds the most recently modified file matching the pattern in a given directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found for latest file search: {directory}")
        return None
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_world_simulation_uuid_from_config_filename(jsonl_filepath: str) -> bool:
    # This function remains largely the same, but uses console for output
    """
    Determines the World Simulation UUID by finding the latest world_config_*.json
    file in the project's 'data' directory and extracting the UUID from its filename.
    """
    global WORLD_SIMULATION_UUID
    try:
        # Navigate from JSONL path to project's 'data' directory
        # Assumes JSONL is in project_root/logs/events/ or a similar structure
        events_dir = os.path.dirname(jsonl_filepath)
        logs_dir = os.path.dirname(events_dir)
        project_root_dir = os.path.dirname(logs_dir)
        data_dir = os.path.join(project_root_dir, "data")

        if not os.path.isdir(data_dir):
            console.print(f"[red]Error:[/red] 'data' directory not found at expected path: {data_dir}")
            return False

        latest_config_file_path = find_latest_file_in_dir(data_dir, "world_config_*.json") # Use the helper

        if not latest_config_file_path:
            print(f"Error: No 'world_config_*.json' files found in {data_dir}.")
            return False

        filename = os.path.basename(latest_config_file_path)
        # Regex to extract UUID from "world_config_UUID.json"
        match = re.match(r"world_config_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.json", filename)
        if match:
            WORLD_SIMULATION_UUID = match.group(1)
            console.print(f"[green]Loaded World Simulation UUID:[/green] {WORLD_SIMULATION_UUID} from config file: {filename}")
            
            # Optional: Verify UUID inside the file
            try:
                with open(latest_config_file_path, 'r', encoding='utf-8') as f_cfg:
                    config_content = json.load(f_cfg)
                    if config_content.get("world_instance_uuid") != WORLD_SIMULATION_UUID:
                        console.print(f"[yellow]Warning:[/yellow] UUID in filename ({WORLD_SIMULATION_UUID}) does not match 'world_instance_uuid' in file ({config_content.get('world_instance_uuid')}). Using filename UUID.")
            except Exception as e_verify:
                console.print(f"[yellow]Warning:[/yellow] Could not verify UUID inside {filename}: {e_verify}")
            return True
        else:
            print(f"Error: Could not extract UUID from filename: {filename}")
            return False

    except Exception as e:
        print(f"An unexpected error occurred while loading UUID from config filename: {e}")
        return False


def _initialize_bluesky_client():
    """Initializes and logs in the Bluesky client if not already done."""
    global bluesky_client
    if bluesky_client is None:
        load_dotenv() # Load .env file
        bsky_handle = os.environ.get("BLUESKY_HANDLE")
        bsky_password = os.environ.get("BLUESKY_APP_PASSWORD")

        if not bsky_handle or not bsky_password:
            console.print("[red]Error:[/red] BLUESKY_HANDLE or BLUESKY_APP_PASSWORD not found in .env file.")
            console.print("[yellow]Actual posting to Bluesky will be skipped.[/yellow]")
            return False
        
        try:
            client = Client()
            client.login(bsky_handle, bsky_password)
            bluesky_client = client
            console.print(f"[green]Successfully logged into Bluesky as {bsky_handle}.[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error logging into Bluesky:[/red] {e}")
            print("Actual posting to Bluesky will be skipped.")
            bluesky_client = None 
            return False
    return True


def _send_to_bluesky_api(text_content, image_path=None, alt_text=None):
    """
    Actually posts to Bluesky using the atproto library.
    """
    if not bluesky_client:
        print("Bluesky client not initialized. Cannot post.")
        return

    try:
        embed_to_post = None
        if image_path:
            # Construct full_image_path using IMAGE_ACTUAL_DIR
            # image_path is just the filename from the event log (now includes UUID subdir)
            full_image_path = os.path.join(IMAGE_ACTUAL_DIR, image_path) if IMAGE_ACTUAL_DIR and not os.path.isabs(image_path) else image_path

            if not os.path.exists(full_image_path):
                console.print(f"[yellow]Image file not found:[/yellow] {full_image_path}. Posting text only.")
            else:
                image_bytes_for_upload = None
                original_file_size = os.path.getsize(full_image_path)

                if original_file_size > BLUESKY_MAX_IMAGE_SIZE_BYTES:
                    print(f"Image {full_image_path} ({original_file_size / (1024*1024):.2f}MB) exceeds Bluesky limit. Attempting to compress.")
                    try:
                        img_pil = Image.open(full_image_path)
                        if img_pil.mode == 'RGBA':
                            img_pil = img_pil.convert('RGB')
                        
                        temp_image_buffer = BytesIO()
                        quality = 85
                        while quality >= 50:
                            temp_image_buffer.seek(0)
                            temp_image_buffer.truncate()
                            img_pil.save(temp_image_buffer, format="JPEG", quality=quality, optimize=True)
                            if temp_image_buffer.tell() <= BLUESKY_MAX_IMAGE_SIZE_BYTES:
                                console.print(f"[green]Compressed image to[/green] {temp_image_buffer.tell() / 1024:.2f}KB (JPEG quality {quality}).")
                                image_bytes_for_upload = temp_image_buffer.getvalue()
                                break
                            quality -= 10
                        if not image_bytes_for_upload:
                            console.print(f"[red]Could not compress image sufficiently.[/red] Final attempt size: {temp_image_buffer.tell() / 1024:.2f}KB.")
                    except Exception as e_compress:
                        print(f"Error during image compression: {e_compress}")
                else:
                    with open(full_image_path, 'rb') as f_img:
                        image_bytes_for_upload = f_img.read()

                if image_bytes_for_upload:
                    console.print(f"[blue]Uploading image:[/blue] {full_image_path} ({len(image_bytes_for_upload)/1024:.2f} KB)...")
                    safe_alt_text = (alt_text or "")[:1000] # Bluesky alt text limit
                    
                    upload_response = bluesky_client.com.atproto.repo.upload_blob(image_bytes_for_upload)
                    
                    if not upload_response or not hasattr(upload_response, 'blob') or not upload_response.blob:
                        print("Failed to upload image blob. Posting text only.")
                    else:
                        image_to_embed = atproto_models.AppBskyEmbedImages.Image(
                            alt=safe_alt_text, image=upload_response.blob
                        )
                        embed_to_post = atproto_models.AppBskyEmbedImages.Main(images=[image_to_embed])
                        print("Image uploaded successfully.")
                elif image_path: # Only print this if an image path was provided but preparation failed
                    print("Image preparation failed. Posting text only.")

        print("Creating post on Bluesky...")
        # Ensure text_content includes hashtags if desired (already handled by chunking logic)
        bluesky_client.com.atproto.repo.create_record(
            atproto_models.ComAtprotoRepoCreateRecord.Data(
                repo=bluesky_client.me.did, # Requires login
                collection=atproto_models.ids.AppBskyFeedPost,
                record=atproto_models.AppBskyFeedPost.Main(
                    created_at=bluesky_client.get_current_time_iso(),
                    text=text_content,
                    embed=embed_to_post,
                    # langs=['en'] # Optional
                ),
            )
        )
        print("Post successfully created on Bluesky.")
    except Exception as e:
        print(f"Error during Bluesky API call: {e}")

def _load_events_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Loads and parses JSONL events from a file."""
    raw_events = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    raw_events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]Skipping malformed JSON line:[/yellow] {line.strip()} - Error: {e}")
                    continue
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found at {filepath}")
        return []
    except Exception as e:
        console.print(f"[red]Error:[/red] An unexpected error occurred while reading {filepath}: {e}")
        return []

    if not raw_events:
        console.print("[yellow]No events found in file.[/yellow]")

    return raw_events

def _extract_image_info_from_events(events: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    """Extracts image generation events and maps them by simulation time."""
    image_info_map = {}
    filename_time_regex = re.compile(r"_T(\d+(?:\.\d+)?)_")

    for event in events:
        if event.get("agent_id") == "ImageGenerator" and event.get("event_type") == "image_generation":
            data = event.get("data", {})
            filename = data.get("image_filename")
            prompt = data.get("prompt_snippet") # This is the refined prompt, good for alt text
            sim_time = event.get("sim_time_s") # Use the sim_time from the event itself

            if filename and prompt and sim_time is not None:
                 # Use the sim_time from the event data directly, it's more reliable
                 # than parsing the filename again.
                 image_info_map[sim_time] = {
                     "path": filename,
                     "alt": prompt, # Use the refined prompt as alt text
                     "used": False
                 }
                 # Optional: Still check filename time for consistency/debugging
                 match = filename_time_regex.search(filename)
                 if match:
                     try:
                         file_sim_time = float(match.group(1))
                         if abs(file_sim_time - sim_time) > 0.5: # Check if they are significantly different
                             console.print(f"[yellow]Warning:[/yellow] Sim time in event ({sim_time:.1f}s) differs from filename ({file_sim_time:.1f}s) for image {filename}.")
                     except ValueError: pass # Ignore parsing errors
            
    return image_info_map

def post_to_bluesky(text_content, image_path=None, alt_text=None):
    """
    Shows a confirmation and then calls the actual Bluesky posting function.
    """
    print("--- PROPOSED BLUESKY POST ---")
    console.print(f"[bold]Content[/] ({len(text_content)} chars):\n{text_content}")
    if image_path:
        full_image_path = image_path
        # Construct full_image_path using IMAGE_ACTUAL_DIR for display
        # image_path is just the filename from the event log (now includes UUID subdir)
        full_image_path = os.path.join(IMAGE_ACTUAL_DIR, image_path) if IMAGE_ACTUAL_DIR and not os.path.isabs(image_path) else image_path
        
        console.print(f"[bold]Attaching Image:[/bold] {full_image_path}")
        if os.path.exists(full_image_path):
            console.print("[dim](Image file exists on disk)[/dim]")
        else:
            print(f"(Image file NOT FOUND at: {full_image_path})")
        print(f"Alt Text: {alt_text}")
    print("--- END PROPOSED POST ---")

    confirm = input("Post this to Bluesky? (y/n): ").lower()
    if confirm == 'y':
        if _initialize_bluesky_client():
            print("Proceeding with post...")
            _send_to_bluesky_api(text_content, image_path, alt_text)
    else:
        print("Skipping post as per user confirmation.")
    print()

def process_events_file(filepath):
    global IMAGE_BASE_DIR
    IMAGE_BASE_DIR = os.path.dirname(filepath) # Images are relative to the JSONL file's location

    if not load_world_simulation_uuid_from_config_filename(filepath):
        console.print("[red]Failed to load World Simulation UUID. Aborting.[/red]")
        return

    image_info_map = {}

    raw_events = _load_events_from_file(filepath)
    if not raw_events:
        return # Loading failed or file was empty, message handled in _load_events_from_file

    image_info_map = _extract_image_info_from_events(raw_events)
            
    for event in raw_events:
        agent_id = event.get("agent_id")
        event_type = event.get("event_type")
        data = event.get("data", {})
        current_event_sim_time = event.get("sim_time_s") 

        text_to_post_content = None

        if agent_id == "Narrator" and event_type == "narration":
            # The 'narrative' field from Narrator output already includes the "At <time>..." prefix
            # We might want to strip it if WORLD_SIMULATION_UUID already implies context,
            # or keep it if it's useful. For now, let's keep it.
            text_to_post_content = data.get("narrative") 
        elif agent_id == "sim_sustdj" and event_type == "intent":
            text_to_post_content = data.get("details") # This is usually the "what I want to say" part
        
        if not text_to_post_content:
            continue

        attached_image_path = None
        attached_alt_text = None
        time_matching_tolerance = 0.1 
        best_match_time_diff = float('inf')
        matched_image_key = None

        if current_event_sim_time is not None:
            for img_ref_sim_time, img_data in image_info_map.items():
                if not img_data["used"]:
                    current_diff = abs(img_ref_sim_time - current_event_sim_time)
                    if current_diff < time_matching_tolerance and current_diff < best_match_time_diff:
                        best_match_time_diff = current_diff
                        matched_image_key = img_ref_sim_time
            
            if matched_image_key is not None:
                image_data_to_use = image_info_map[matched_image_key]
                attached_image_path = image_data_to_use["path"]
                attached_alt_text = image_data_to_use["alt"]
                image_info_map[matched_image_key]["used"] = True
            else:
                # No matching image found - ask if user wants to generate one
                console.print("[yellow]No matching image found for this narrative.[/yellow]")
                console.print(f"[cyan]Narrative text:[/cyan] {text_to_post_content[:100]}...")
                
                # Check if AI clients are initialized
                enable_image_generation = os.environ.get("ENABLE_ON_DEMAND_IMAGE_GENERATION", "true").lower() == "true"
                if enable_image_generation and not (text_gen_client and img_gen_client):
                    initialize_ai_clients()
                
                if text_gen_client and img_gen_client:
                    generate_img = input("Generate an image for this narrative? (y/n): ").lower().strip()
                    if generate_img == 'y':
                        # Extract weather if mentioned in text
                        weather_match = re.search(r"(sunny|cloudy|rainy|snowy|clear|overcast|foggy|misty)", 
                                                text_to_post_content.lower())
                        weather = weather_match.group(1) if weather_match else "clear"
                        
                        # Generate the image
                        attached_image_path, attached_alt_text = generate_image_for_narrative(
                            narrative_text=text_to_post_content,
                            sim_time=current_event_sim_time,
                            weather=weather,
                            world_mood="ordinary"  # Could extract from data if available
                        )
                        
                        if attached_image_path:
                            console.print(f"[green]Successfully generated image:[/green] {attached_image_path}")
                        else:
                            console.print("[red]Failed to generate image.[/red]")
                else:
                    console.print("[yellow]Image generation is not available. Enable it by setting ENABLE_ON_DEMAND_IMAGE_GENERATION=true in .env file.[/yellow]")

        words = text_to_post_content.split()
        if not words:
            continue
        
        # Calculate space for WorldID, part indicator, and hashtags
        # Example: "WorldID: <uuid> (10/10) #Tag1 #Tag2 "
        hashtags_len = len(SOCIAL_POST_HASHTAGS) + (len(SOCIAL_POST_HASHTAGS.split()) if SOCIAL_POST_HASHTAGS else 0) # Add spaces for hashtags
        max_header_len = len(f"WorldID: {WORLD_SIMULATION_UUID} (00/00) ") + hashtags_len
        content_char_limit_per_chunk = MAX_POST_LENGTH - max_header_len

        if content_char_limit_per_chunk <= 20:
            print(f"Warning: content_char_limit_per_chunk is very small ({content_char_limit_per_chunk}). May result in poor chunking.")
            content_char_limit_per_chunk = max(20, content_char_limit_per_chunk)

        raw_content_chunks = []
        current_chunk_words = []

        for word in words:
            potential_chunk_words = current_chunk_words + [word]
            potential_text = " ".join(potential_chunk_words)
            
            if len(potential_text) <= content_char_limit_per_chunk:
                current_chunk_words = potential_chunk_words
            else:
                if current_chunk_words:
                    raw_content_chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [word]
        
        if current_chunk_words:
            raw_content_chunks.append(" ".join(current_chunk_words))
        
        if not raw_content_chunks and text_to_post_content:
            raw_content_chunks.append(text_to_post_content)

        num_total_chunks = len(raw_content_chunks)

        for i, chunk_text_part in enumerate(raw_content_chunks):
            current_header_prefix = f"WorldID: {WORLD_SIMULATION_UUID}"
            current_post_header = ""
            if num_total_chunks > 1:
                current_post_header = f"{current_header_prefix} ({i+1}/{num_total_chunks}) "
            else:
                current_post_header = f"{current_header_prefix} "
            
            # Add hashtags to the end of the text part, before final length check
            text_with_hashtags = chunk_text_part
            if SOCIAL_POST_HASHTAGS:
                text_with_hashtags += f"\n\n{SOCIAL_POST_HASHTAGS}" # Add hashtags on new lines

            full_post_text = current_post_header + text_with_hashtags
            
            if len(full_post_text) > MAX_POST_LENGTH:
                # Recalculate space for content, considering header and hashtags are now part of text_with_hashtags
                # The header is fixed, so we need to truncate text_with_hashtags
                space_for_text_with_hashtags = MAX_POST_LENGTH - len(current_post_header)
                
                if space_for_text_with_hashtags <= 3: # Not enough for "..."
                    truncated_part = text_with_hashtags[:space_for_text_with_hashtags]
                else:
                    sub_part_for_trunc = text_with_hashtags[:space_for_text_with_hashtags - 3]
                    last_space = sub_part_for_trunc.rfind(' ')
                    if last_space > 0 :
                         truncated_part = sub_part_for_trunc[:last_space] + "..."
                    else:
                         truncated_part = sub_part_for_trunc + "..."
                full_post_text = current_post_header + truncated_part
                if len(full_post_text) > MAX_POST_LENGTH: 
                    full_post_text = full_post_text[:MAX_POST_LENGTH -3] + "..."

            img_for_this_post = None
            alt_for_this_post = None
            if i == 0 and attached_image_path: # Image only on the first post of a thread
                img_for_this_post = attached_image_path
                alt_for_this_post = attached_alt_text
            
            post_to_bluesky(full_post_text, image_path=img_for_this_post, alt_text=alt_for_this_post)

    unused_image_count = 0
    for ref_time, img_data in image_info_map.items():
        if not img_data["used"]:
            unused_image_count += 1
            print(f"Info: Unused image (intended for sim_time ~{ref_time}): {img_data['path']}")
    if unused_image_count > 0:
        print(f"Info: Total {unused_image_count} pre-processed images were not attached to any post.")


# Initialize Google AI clients
def initialize_ai_clients():
    """Initialize Google AI clients for text and image generation."""
    global img_gen_client, text_gen_client
    
    load_dotenv()  # Make sure we load .env variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] GOOGLE_API_KEY not found in .env file.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        img_gen_client = genai
        text_gen_client = genai.GenerativeModel(
            os.environ.get("MODEL_NAME", "gemini-2.0-flash")
        )
        console.print("[green]Successfully initialized AI clients.[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error initializing AI clients:[/red] {e}")
        return False

def get_random_style_combination(num_general=0, num_lighting=1, num_color=1, 
                                num_technique=1, num_composition=1, num_atmosphere=1):
    """Generate a random style combination for image prompts."""
    import random
    
    general_styles = ["dramatic", "cinematic", "photorealistic", "detailed", "vivid", "rich", "expressive", "artistic"]
    lighting_styles = ["golden hour", "soft lighting", "morning light", "evening light", "natural lighting", "dramatic lighting", "diffused light", "ambient lighting", "warm lighting", "cool lighting"]
    color_styles = ["vibrant colors", "muted colors", "warm tones", "cool tones", "high contrast", "low contrast", "monochromatic", "complementary colors", "analogous colors"]
    techniques = ["depth of field", "bokeh", "shallow focus", "sharp focus", "tilt-shift", "high dynamic range", "low key", "high key"]
    composition_styles = ["rule of thirds", "leading lines", "symmetrical", "asymmetrical", "centered composition", "negative space", "framing", "diagonal composition", "golden ratio"]
    atmosphere_styles = ["serene", "melancholic", "nostalgic", "tense", "peaceful", "mysterious", "playful", "solemn", "ethereal", "dramatic atmosphere"]
    
    style_parts = []
    if num_general > 0:
        style_parts.extend(random.sample(general_styles, min(num_general, len(general_styles))))
    if num_lighting > 0:
        style_parts.extend(random.sample(lighting_styles, min(num_lighting, len(lighting_styles))))
    if num_color > 0:
        style_parts.extend(random.sample(color_styles, min(num_color, len(color_styles))))
    if num_technique > 0:
        style_parts.extend(random.sample(techniques, min(num_technique, len(techniques))))
    if num_composition > 0:
        style_parts.extend(random.sample(composition_styles, min(num_composition, len(composition_styles))))
    if num_atmosphere > 0:
        style_parts.extend(random.sample(atmosphere_styles, min(num_atmosphere, len(atmosphere_styles))))
    
    return ", ".join(style_parts)

def generate_image_for_narrative(narrative_text, sim_time=None, weather="clear", world_mood="ordinary"):
    """Generate an image for a narrative text using Google's image generation API."""
    if not text_gen_client or not img_gen_client:
        console.print("[yellow]AI clients not initialized. Cannot generate image.[/yellow]")
        return None, None
    
    console.print("[blue]Generating image for narrative...[/blue]")
    time_string = "afternoon" if not sim_time else f"{sim_time}"
    
    # Step 1: Refine the narrative into a concise image prompt
    console.print("[dim]Refining narrative text into image prompt...[/dim]")
    actor_name = "character"  # Extract from narrative if possible
    
    # Extract actor name with regex (optional)
    import re
    actor_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+) (?:looks|walks|sits|stands|observes|examines)", narrative_text)
    if actor_match:
        actor_name = actor_match.group(1)
    
    prompt_for_refinement = f"""You are an expert at transforming narrative text into concise, visually descriptive prompts ideal for an image generation model. Your goal is to focus on a single, clear subject, potentially with a naturally blurred background.
Original Narrative Context: "{narrative_text}"
Current Time: "{time_string}"
Current Weather: "{weather}"
World Mood: "{world_mood}"
Instructions for Refinement:
1. Identify a single, compelling visual element or a very brief, static moment from the 'Original Narrative Context'.
2. Describe this single subject clearly and vividly. Use descriptive language for the subject and its relationship to any implied background.
3. If appropriate, suggest a composition that would naturally lead to a blurred background (e.g., "A close-up of...", "A detailed shot of...", "A lone figure with the background softly blurred...").
4. Keep the refined description concise (preferably 1-2 sentences).
5. The refined description should be purely visual and directly usable as an image prompt.
6. Do NOT include any instructions for the image generation model itself (like "Generate an image of..."). Just provide the refined descriptive text.
7. The "single subject" should be an object, a part of the environment, or an abstract concept from the narrative. DO NOT make the actor ({actor_name}) the primary subject of the visual description.
Refined Visual Description:"""

    try:
        response = text_gen_client.generate_content(prompt_for_refinement)
        refined_narrative = response.text.strip()
        console.print(f"[green]Refined image prompt:[/green] {refined_narrative}")
        
        # Step 2: Generate the image using the refined prompt
        console.print("[dim]Generating image...[/dim]")
        
        random_style = get_random_style_combination(
            num_general=0, num_lighting=1, num_color=1, 
            num_technique=1, num_composition=1, num_atmosphere=1
        )
        
        prompt_for_image_gen = f"""Generate a high-quality, visually appealing, **photo-realistic** photograph depicting a scene or subject directly related to the following narrative context. The viewpoint should be observational, focusing on the environment or key elements described.
Narrative Context: "{refined_narrative}"
Style: "{random_style}"
Instructions for the Image:
The image should feature:
- Time of Day: Reflect the lighting and atmosphere typical of "{time_string}".
- Weather: Depict the conditions described by "{weather}".

ABSOLUTELY CRUCIAL EXCLUSIONS: No digital overlays, UI elements, watermarks, or logos. The actor ({actor_name}) or ANY human figures MUST NOT be visible in the image. The focus is SOLELY on the described scene, objects, or atmosphere.
Generate this image."""

        # Generate the image
        response = img_gen_client.GenerativeModel(
            os.environ.get("IMAGE_GENERATION_MODEL_NAME", "imagen-3.0-generate-002")
        ).generate_images(
            prompt=prompt_for_image_gen,
            # adjust parameters as needed
        )
        
        # Save the image
        if not response.images:
            console.print("[yellow]No images were generated.[/yellow]")
            return None, None
            
        # Create path and save image
        if not WORLD_SIMULATION_UUID:
            console.print("[yellow]No World UUID set. Using 'generated' as fallback.[/yellow]")
            uuid_dir = "generated"
        else:
            uuid_dir = WORLD_SIMULATION_UUID
            
        # Make sure the image output directory exists
        narrative_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "data", "narrative_images")
        os.makedirs(narrative_images_dir, exist_ok=True)
        
        uuid_image_output_dir = os.path.join(narrative_images_dir, uuid_dir)
        os.makedirs(uuid_image_output_dir, exist_ok=True)
        
        # Save image
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_time_str = f"T{int(sim_time) if sim_time else 0}"
        image_filename = f"narrative_{sim_time_str}_{timestamp_str}.png"
        full_image_path = os.path.join(uuid_image_output_dir, image_filename)
        
        # Convert base64 to image and save
        from PIL import Image
        import base64
        
        if response.images:
            with open(full_image_path, "wb") as f:
                # Assuming images are returned as base64 or PIL objects
                # Modify this part according to the actual API response
                f.write(response.images[0])
                
            console.print(f"[green]Image saved to:[/green] {full_image_path}")
            
            # Return relative path for event log and alt text
            relative_path = os.path.join(uuid_dir, image_filename)
            return relative_path, refined_narrative
            
    except Exception as e:
        console.print(f"[red]Error generating image:[/red] {e}")
        import traceback
        traceback.print_exc()
        
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process simulation event logs and post to Bluesky.")
    parser.add_argument(
        "logfile",
        type=str,
        nargs='?', # Makes the argument optional
        help="Path to the specific .jsonl event log file to process. If omitted, the latest 'events_latest_*.jsonl' in the default directory will be used."
    )
    args = parser.parse_args()

    target_jsonl_file_path = args.logfile
    
    if not target_jsonl_file_path:
        console.print("[yellow]No specific log file provided. Looking for the latest event log file...[/yellow]")
        # Default behavior: find the latest log file
        events_log_directory = r".\logs\events" # Default directory
        events_file_pattern = "events_latest_*.jsonl"
        console.print(f"[blue]Looking in:[/blue] {events_log_directory} with pattern: {events_file_pattern}")
        target_jsonl_file_path = find_latest_file_in_dir(events_log_directory, events_file_pattern)

        if not target_jsonl_file_path:
            console.print(f"[bold red]FATAL ERROR:[/bold red] No event log files matching '{events_file_pattern}' found in {events_log_directory}.")
            exit(1)

    if not os.path.exists(target_jsonl_file_path):
        print(f"FATAL ERROR: Specified log file not found: {target_jsonl_file_path}")
    else:
        print(f"Processing events from log file: {target_jsonl_file_path}")
        process_events_file(target_jsonl_file_path)
