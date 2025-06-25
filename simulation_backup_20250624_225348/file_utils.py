# src/file_utils.py
import os
import logging

logger = logging.getLogger(__name__)

# This assumes that file_utils.py is in the 'src' directory,
# and the project root is the parent of 'src'.
_BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_dir_exists(path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")
    elif not os.path.isdir(path):
        logger.error(f"Path exists but is not a directory: {path}")
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")

def get_data_dir() -> str:
    """Returns the absolute path to the main data directory."""
    return os.path.join(_BASE_PROJECT_DIR, "data")

def get_states_dir() -> str:
    """Returns the absolute path to the states subdirectory within the data directory."""
    return os.path.join(get_data_dir(), "states")

def get_world_config_dir() -> str:
    """Returns the absolute path to the directory containing world_config files (which is the main data directory)."""
    return get_data_dir()

def get_life_summary_dir() -> str:
    """Returns the absolute path to the life_summaries subdirectory within the data directory."""
    return os.path.join(get_data_dir(), "life_summaries")