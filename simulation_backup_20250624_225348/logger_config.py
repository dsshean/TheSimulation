import logging
import datetime # This is correct for datetime.datetime.now()
import sys
import os # Added for os.makedirs
import json # Added for JSON operations
 
def setup_unique_logger(
    logger_name="TheSimulationApp",
    file_prefix="TheSimulation",
    file_level=logging.DEBUG,  # Default file logging level
    console_level=logging.INFO,  # Default console logging level
    file_formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    console_formatter_str='%(levelname)s: %(message)s',
    file_mode='a'  # Default to append for log files
):
    """
    Sets up a logger that writes to a unique, timestamped file and the console.
 
    Args:
        logger_name (str): The name for the logger instance.
        file_prefix (str): The prefix for the log filename.
        file_level (int): Logging level for the file handler.
        console_level (int): Logging level for the console handler.
        file_formatter_str (str): Format string for file logs.
        console_formatter_str (str): Format string for console logs.
        file_mode (str): File mode for the FileHandler ('a' for append, 'w' for write).
 
    Returns:
        tuple[logging.Logger, str]: The configured logger instance and the log filename.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{file_prefix}_{timestamp}.log"

    # Get the root logger to configure it globally
    root_logger = logging.getLogger()
    # Set root logger level to the most verbose of its handlers
    # The handlers themselves will filter based on their own levels.
    root_logger.setLevel(min(file_level, console_level))
 
    # Prevent adding multiple handlers to the root logger if this function is called multiple times
    # (e.g., in tests or if the module is reloaded, though less common for root).
    # A more robust check might involve checking handler types if adding selectively.
    # For simplicity, if root has no handlers, we add ours.
    # If it has handlers, we assume it's already configured (e.g., by a previous call to this function).
    if not root_logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(file_formatter_str)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
 
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(console_formatter_str)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
 
        # Log the initialization message using a specific logger for clarity if needed
        init_logger = logging.getLogger(logger_name) # Use the app-specific logger for this message
        init_logger.info(f"Root logger configured by '{logger_name}'. Logging to file: {log_filename} and console.")
    else: # If root_logger already has handlers
        # Attempt to find the actual log_filename if root logger was already configured
        actual_log_filename_if_exists = log_filename # Default to initially calculated name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename'):
                actual_log_filename_if_exists = handler.baseFilename
                log_filename = actual_log_filename_if_exists # Update log_filename to be returned
                break
        # Log that the logger (identified by logger_name) is using the already configured root handlers.
        already_init_logger = logging.getLogger(logger_name)
        already_init_logger.info(f"Logger '{logger_name}' is using already configured root logger. Log file likely: {actual_log_filename_if_exists}")

    # Return the specifically named logger for direct use, and the determined log_filename
    return logging.getLogger(logger_name), log_filename

# --- Added for Structured Event Logging ---

class JsonFormatter(logging.Formatter):
    """
    A custom formatter that formats the log record's message directly.
    Assumes the message is already a JSON string.
    """
    def format(self, record):
        # We expect the message to be a pre-formatted JSON string
        return record.getMessage()

def setup_event_logger(instance_uuid: str = None, log_dir: str = "logs/events"):
    """
    Sets up a dedicated logger for structured JSON events.
    Events are written to a .jsonl file (one JSON object per line).

    Args:
        instance_uuid (str, optional): The UUID of the simulation instance.
                                       If None, a timestamped log for 'latest' run is created.
        log_dir (str): The directory to store event logs.

    Returns:
        tuple: (logging.Logger instance, str path to log file) or (None, None) if setup fails.
    """
    logger_name = f"EventLogger_{instance_uuid or 'latest'}"
    event_logger = logging.getLogger(logger_name)
    event_logger.setLevel(logging.INFO)  # Log all event messages passed to it
    event_logger.propagate = False    # Don't send to parent (main) loggers (especially the root logger)

    os.makedirs(log_dir, exist_ok=True) # Ensure the event log directory exists

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Consistent with setup_unique_logger
    log_filename = f"events_{instance_uuid}_{timestamp}.jsonl" if instance_uuid else f"events_latest_{timestamp}.jsonl"
    file_path = os.path.join(log_dir, log_filename)

    fh = logging.FileHandler(file_path, mode='a', encoding='utf-8') # Append mode
    fh.setFormatter(JsonFormatter())
    event_logger.addHandler(fh)
    return event_logger, file_path