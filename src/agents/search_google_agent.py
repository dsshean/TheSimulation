import logging
from google.adk.agents import Agent
from google.adk.tools import google_search
from src.config import settings # Assuming settings.MODEL_GEMINI_FLASH_SEARCH points to gemini-2.0-flash or similar
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# --- Define a simple instruction for the search agent ---
SEARCH_AGENT_INSTRUCTION = """
You are a search assistant. Your ONLY task is to use the provided 'google_search' tool to find information based on the user's query.
Return the search results concisely. Do not add any conversational text, just the information found.
"""

search_google_agent = None
try:
    # --- Use a model specifically compatible with google_search, like gemini-2.0-flash ---
    # --- Ensure settings.MODEL_GEMINI_FLASH_SEARCH is defined in config.py ---
    search_model = settings.MODEL_GEMINI_FLASH_SEARCH or "gemini-2.0-flash" # Fallback
    if not settings.MODEL_GEMINI_FLASH_SEARCH:
        logger.warning("settings.MODEL_GEMINI_FLASH_SEARCH not defined, using default 'gemini-2.0-flash'.")

    search_google_agent = Agent(
        name="SearchGoogleAgent",
        model=search_model,
        description="Performs Google searches based on provided queries.",
        instruction=SEARCH_AGENT_INSTRUCTION,
        tools=[
            google_search # Only this tool
        ],
    )
    console.print(f"Agent '[bold cyan]{search_google_agent.name}[/bold cyan]' defined (Dedicated Search Role). Model: {search_model}")
    logger.info(f"SearchGoogleAgent initialized with model {search_model} and google_search tool.")

except Exception as e:
    console.print(f"[bold red]Error creating SearchGoogleAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    search_google_agent = None # Ensure it's None on error
