import logging
from google.adk.tools.agent_tool import AgentTool # Corrected import
from src.agents.search_google_agent import search_google_agent # Import the agent instance
from rich.console import Console
from typing import Optional # Import Optional

console = Console()
logger = logging.getLogger(__name__)

# --- Function to create the AgentTool ---
def create_search_agent_tool() -> Optional[AgentTool]:
    """
    Creates and returns the AgentTool wrapping the search_google_agent.
    Returns None if the search_google_agent is not initialized or if creation fails.
    """
    # Explicitly check if search_google_agent is initialized
    if search_google_agent:
        try:
            tool = AgentTool(
                agent=search_google_agent,
                name="call_search_agent",
                description="Calls a dedicated search agent to perform a Google search for the given query. Input should be the search query string.",
                # Define input/output schema if needed, but for simple text query/response, defaults might suffice.
                # input_schema=...,
                # output_schema=...
            )
            logger.info(f"Successfully created AgentTool '{tool.name}' wrapping agent '{search_google_agent.name}'.")
            console.print(f"Tool '[bold purple]{tool.name}[/bold purple]' created (wraps SearchGoogleAgent).")
            return tool
        except Exception as e:
            # Catch potential errors during AgentTool creation itself
            logger.error(f"Error during AgentTool instantiation for search_google_agent: {e}", exc_info=True)
            console.print(f"[bold red]Error creating AgentTool instance:[/bold red] {e}")
            return None
    else:
        # This case means search_google_agent was None when this function was called
        logger.error("SearchGoogleAgent instance is None. Cannot create AgentTool.")
        console.print("[bold red]Error:[/bold red] Cannot create 'call_search_agent' tool because SearchGoogleAgent is None.")
        return None

# --- Create the tool instance by calling the function ---
# This attempts to create the tool when this module is loaded.
search_agent_tool: Optional[AgentTool] = create_search_agent_tool()

# Add a log to confirm if the tool was created or not at the end of this module's execution
if search_agent_tool:
    logger.info("agent_tools.py loaded: search_agent_tool created successfully.")
else:
    logger.warning("agent_tools.py loaded: search_agent_tool creation FAILED.")
