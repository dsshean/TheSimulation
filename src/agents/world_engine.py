# src/agents/world_engine.py
from google.adk.agents import Agent
from src.config import settings
from src.tools import world_engine_tools
from src.prompts import world_engine_instructions # Import instructions
from rich.console import Console

console = Console()

def create_agent():
    """Factory function to create the World Engine agent."""
    try:
        agent = Agent(
            name="world_engine",
            model=settings.MODEL_GEMINI_FLASH,
            description=(
                "Provides setting details and processes movement/time updates when instructed by the Narrator. "
                "Uses its internal knowledge to enhance descriptions and provide additional context."
            ),
            instruction=(
                f"{world_engine_instructions.WORLD_ENGINE_AGENT_INSTRUCTION}\n\n"
                "When providing setting details, use your internal knowledge about the location to generate descriptions if tools do not provide sufficient information."
            ),
            tools=[
                world_engine_tools.get_setting_details,
                world_engine_tools.process_movement,
                world_engine_tools.advance_time
            ],
        )
        console.print(f"Agent '[bold green]{agent.name}[/bold green]' created.")
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating world_engine_agent:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None