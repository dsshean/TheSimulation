# src/agents/world_engine.py
from google.adk.agents import LlmAgent
from google.genai import types # <<< CORRECTED IMPORT: Changed 'google.adk' to 'google.genai'
from src.config import settings # Assuming settings defines the model
from src.prompts import world_engine_instructions # Import the NEW instructions
from rich.console import Console
import json # Import json for schema

console = Console()

# Define the expected output schema as a dictionary
# Based on the instructions, it should be a JSON dictionary
# Use types.JsonSchema for validation if ADK supports it directly,
# otherwise use dict for basic type hint.
# For more robustness, define a Pydantic model if using elsewhere.
OUTPUT_SCHEMA = dict

# Create the agent instance
# No factory function needed if directly instantiated
world_engine = LlmAgent(
    name="world_engine",
    model=settings.MODEL_GEMINI_PRO, # Use a model strong in reasoning
    description=(
        "Enforces physical rules and constraints of the simulation world based on "
        "provided context and proposed actions. Validates action feasibility and estimates durations."
    ),
    instruction=world_engine_instructions.WORLD_ENGINE_INSTRUCTION, # Use the NEW instruction
    tools=[], # <<<< CORE CHANGE: No external state-fetching tools by default
              # Add calculation/physics tools only if internal reasoning is insufficient.
    output_key="validation_result", # Optional: Save the validation output dictionary
)

# console.print(f"Agent '[bold green]{world_engine.name}[/bold green]' defined (Rule Enforcement Role).")


# Example of how you might define a more specific schema using types.JsonSchema
# (Requires understanding ADK's specific schema handling capabilities)
# try:
#     output_schema_detail = types.JsonSchema(
#         type=types.JsonSchema.Type.OBJECT,
#         properties={
#             "validation_status": types.JsonSchema(type=types.JsonSchema.Type.STRING, enum=["approved", "rejected", "modified"]),
#             "reasoning": types.JsonSchema(type=types.JsonSchema.Type.STRING),
#             "estimated_duration_seconds": types.JsonSchema(type=types.JsonSchema.Type.INTEGER),
#             "adjusted_outcome": types.JsonSchema(type=types.JsonSchema.Type.STRING)
#         },
#         required=["validation_status", "reasoning", "estimated_duration_seconds"]
#     )
#     world_engine_with_schema = LlmAgent(
#         # ... other params ...
#         output_schema=output_schema_detail,
#         response_format="json",
#         output_key="validation_result",
#     )
#     console.print(f"Agent '[bold green]{world_engine.name}[/bold green]' defined with detailed schema.")
# except Exception as e:
#      console.print(f"[yellow]Could not define detailed output schema for world_engine: {e}. Using basic dict.[/yellow]")
#      # Fallback to the simpler definition above if schema definition fails