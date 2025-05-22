import json
import uuid
import re
import os
import glob
from io import BytesIO
import argparse # Import argparse for command-line arguments
from typing import Optional # Added import for Optional type hint
from PIL import Image # For image compression
from dotenv import load_dotenv
from atproto import Client, models as atproto_models # Aliased import

# --- Configuration ---
WORLD_SIMULATION_UUID = None # Will be loaded dynamically
MAX_POST_LENGTH = 300 # Bluesky's character limit (graphemes approx) - align with SOCIAL_POST_TEXT_LIMIT
IMAGE_BASE_DIR = "" # Dynamically set based on input JSONL file path

# Constants for Bluesky posting (mirroring parts of your project's config.py)
BLUESKY_MAX_IMAGE_SIZE_BYTES = 976 * 1024  # 976KB
SOCIAL_POST_HASHTAGS = os.getenv("SOCIAL_POST_HASHTAGS", "#TheSimulation #AI #DigitalTwin #ProceduralStorytelling")
# MAX_POST_LENGTH is used for the text limit

# --- Bluesky API Client (initialized later if needed) ---
bluesky_client = None

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
    """
    Determines the World Simulation UUID by finding the latest world_config_*.json
    file in the project's 'data' directory and extracting the UUID from its filename.
    """
    global WORLD_SIMULATION_UUID
    try:
        # Navigate from JSONL path to project's 'data' directory
        # Assumes JSONL is in project_root/logs/events/
        events_dir = os.path.dirname(jsonl_filepath)
        logs_dir = os.path.dirname(events_dir)
        project_root_dir = os.path.dirname(logs_dir)
        data_dir = os.path.join(project_root_dir, "data")

        if not os.path.isdir(data_dir):
            print(f"Error: 'data' directory not found at expected path: {data_dir}")
            return False

        latest_config_file_path = find_latest_file_in_dir(data_dir, "world_config_*.json")

        if not latest_config_file_path:
            print(f"Error: No 'world_config_*.json' files found in {data_dir}.")
            return False

        filename = os.path.basename(latest_config_file_path)
        # Regex to extract UUID from "world_config_UUID.json"
        match = re.match(r"world_config_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.json", filename)
        if match:
            WORLD_SIMULATION_UUID = match.group(1)
            print(f"Loaded World Simulation UUID: {WORLD_SIMULATION_UUID} from config file: {filename}")
            
            # Optional: Verify UUID inside the file
            try:
                with open(latest_config_file_path, 'r', encoding='utf-8') as f_cfg:
                    config_content = json.load(f_cfg)
                    if config_content.get("world_instance_uuid") != WORLD_SIMULATION_UUID:
                        print(f"Warning: UUID in filename ({WORLD_SIMULATION_UUID}) does not match 'world_instance_uuid' in file ({config_content.get('world_instance_uuid')}). Using filename UUID.")
            except Exception as e_verify:
                print(f"Warning: Could not verify UUID inside {filename}: {e_verify}")
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
            print("Error: BLUESKY_HANDLE or BLUESKY_APP_PASSWORD not found in .env file.")
            print("Actual posting to Bluesky will be skipped.")
            return False
        
        try:
            client = Client()
            client.login(bsky_handle, bsky_password)
            bluesky_client = client
            print(f"Successfully logged into Bluesky as {bsky_handle}.")
            return True
        except Exception as e:
            print(f"Error logging into Bluesky: {e}")
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
            full_image_path = image_path
            if IMAGE_BASE_DIR and not os.path.isabs(image_path):
                full_image_path = os.path.join(IMAGE_BASE_DIR, image_path)

            if not os.path.exists(full_image_path):
                print(f"Image file not found: {full_image_path}. Posting text only.")
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
                                print(f"Compressed image to {temp_image_buffer.tell() / 1024:.2f}KB (JPEG quality {quality}).")
                                image_bytes_for_upload = temp_image_buffer.getvalue()
                                break
                            quality -= 10
                        if not image_bytes_for_upload:
                            print(f"Could not compress image sufficiently. Final attempt size: {temp_image_buffer.tell() / 1024:.2f}KB.")
                    except Exception as e_compress:
                        print(f"Error during image compression: {e_compress}")
                else:
                    with open(full_image_path, 'rb') as f_img:
                        image_bytes_for_upload = f_img.read()

                if image_bytes_for_upload:
                    print(f"Uploading image: {full_image_path} ({len(image_bytes_for_upload)/1024:.2f} KB)...")
                    safe_alt_text = (alt_text or "")[:1000] # Bluesky alt text limit
                    
                    upload_response = bluesky_client.com.atproto.repo.upload_blob(image_bytes_for_upload)
                    
                    if not upload_response or not hasattr(upload_response, 'blob'):
                        print("Failed to upload image blob. Posting text only.")
                    else:
                        image_to_embed = atproto_models.AppBskyEmbedImages.Image(
                            alt=safe_alt_text, image=upload_response.blob
                        )
                        embed_to_post = atproto_models.AppBskyEmbedImages.Main(images=[image_to_embed])
                        print("Image uploaded successfully.")
                else:
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


def post_to_bluesky(text_content, image_path=None, alt_text=None):
    """
    Shows a confirmation and then calls the actual Bluesky posting function.
    """
    print("--- PROPOSED BLUESKY POST ---")
    print(f"Content ({len(text_content)} chars): {text_content}")
    if image_path:
        full_image_path = image_path
        if IMAGE_BASE_DIR and not os.path.isabs(image_path):
            full_image_path = os.path.join(IMAGE_BASE_DIR, image_path)
        
        print(f"Attaching Image: {full_image_path}")
        if os.path.exists(full_image_path):
            print("(Image file exists on disk)")
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
            print("Skipping post due to Bluesky client initialization failure.")
    else:
        print("Skipping post as per user confirmation.")
    print()


def process_events_file(filepath):
    global IMAGE_BASE_DIR
    IMAGE_BASE_DIR = os.path.dirname(filepath) # Images are relative to the JSONL file's location

    if not load_world_simulation_uuid_from_config_filename(filepath):
        print("Failed to load World Simulation UUID. Aborting.")
        return
    if not WORLD_SIMULATION_UUID:
        print("World Simulation UUID is not set after attempting load. Aborting.")
        return

    raw_events = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    raw_events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    if not raw_events:
        print("No events found in file.")
        return

    image_info_map = {}
    filename_time_regex = re.compile(r"_T(\d+(?:\.\d+)?)_")

    for event in raw_events:
        if event.get("agent_id") == "ImageGenerator":
            data = event.get("data", {})
            filename = data.get("image_filename")
            prompt = data.get("prompt_snippet") # This is the refined prompt, good for alt text
            if filename and prompt:
                match = filename_time_regex.search(filename)
                if match:
                    try:
                        ref_sim_time_str = match.group(1)
                        ref_sim_time = float(ref_sim_time_str)
                        image_info_map[ref_sim_time] = {
                            "path": filename,
                            "alt": prompt, # Use the refined prompt as alt text
                            "used": False
                        }
                    except ValueError:
                        print(f"Warning: Could not parse sim_time from filename: {filename}")
                else:
                    print(f"Warning: Could not extract sim_time signature (e.g., _T123.45_) from filename: {filename}")
            
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
            unused_image_count +=1
            print(f"Info: Unused image (intended for sim_time ~{ref_time}): {img_data['path']}")
    if unused_image_count > 0:
        print(f"Info: Total {unused_image_count} pre-processed images were not attached to any post.")


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
        print("No specific log file provided. Looking for the latest event log file...")
        # Default behavior: find the latest log file
        events_log_directory = r"c:\Users\dshea\Desktop\TheSimulation\logs\events" # Default directory
        events_file_pattern = "events_latest_*.jsonl"
        print(f"Looking in: {events_log_directory} with pattern: {events_file_pattern}")
        target_jsonl_file_path = find_latest_file_in_dir(events_log_directory, events_file_pattern)

        if not target_jsonl_file_path:
            print(f"FATAL ERROR: No event log files matching '{events_file_pattern}' found in {events_log_directory}.")
            exit(1)

    if not os.path.exists(target_jsonl_file_path):
        print(f"FATAL ERROR: Specified log file not found: {target_jsonl_file_path}")
    else:
        print(f"Processing events from log file: {target_jsonl_file_path}")
        process_events_file(target_jsonl_file_path)
