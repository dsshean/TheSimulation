from google.adk.agents import Agent, BaseAgent # Ensure BaseAgent is imported if used in type hint
from google.adk.tools import FunctionTool
from google.adk.sessions import Session # Import Session for type hinting
from src.config import settings
from src.tools import simulacra_tools
from src.prompts import simulacra_instructions
from rich.console import Console
from typing import Dict, Optional, Any # Add necessary types

console = Console()

def create_agent(sim_id: str, persona: Dict, session: Session) -> Optional[BaseAgent]:
    """Factory function to create a specific Simulacra agent instance."""
    try:
        persona_details = persona.get("persona_details", persona)
        name = persona_details.get("Name", sim_id)
        occupation = persona_details.get("Occupation", "Unknown")
        traits = persona_details.get("Personality_Traits", [])
        background = persona_details.get("Background", "N/A")

        instructions = simulacra_instructions.SIMULACRA_AGENT_INSTRUCTION.format(
            agent_name=name,
            agent_id=sim_id, # Make agent_id available to the prompt template
            occupation=occupation,
            traits=', '.join(traits),
            background=background
        )

        agent = Agent(
            name=sim_id,
            model=settings.MODEL_GEMINI_PRO,
            description=f"Simulacra agent representing {name}.", # Dynamic description
            instruction=instructions, # Use the formatted instructions
            tools=[
                FunctionTool(simulacra_tools.generate_internal_monologue),
                FunctionTool(simulacra_tools.attempt_move_to),
                FunctionTool(simulacra_tools.attempt_talk_to),
                FunctionTool(simulacra_tools.check_self_status),
                FunctionTool(simulacra_tools.attempt_interact_with),
                FunctionTool(simulacra_tools.update_self_goal) 
            ],
        )
        return agent
    except Exception as e:
        console.print(f"[bold red]Error creating simulacra_agent for ID {sim_id}:[/bold red] {e}")
        console.print_exception(show_locals=False) # Keep locals off unless debugging deep
        return None