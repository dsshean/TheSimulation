# src/tools/npc_tools.py (Updated for LLM Responses)

from google.adk.tools.tool_context import ToolContext
from rich.console import Console

# --- Add LLMService and settings imports ---
from src.generation.llm_service import LLMService # Or adjust path as needed
from src.config import settings # To get the model name
# --- End Add ---


console = Console()

def generate_npc_response(
    received_message: str,
    npc_name: str,
    npc_role: str,
    tool_context: ToolContext # Added ToolContext parameter
) -> dict:
    """
    Generates a response for the NPC using an LLM, based on the received message,
    the NPC's name, role, and current simulation context. Saves the response
    to the 'last_npc_interaction' state key.

    :param received_message: The message received by the NPC from the Simulacra.
    :param npc_name: The name of the NPC.
    :param npc_role: The role or identity of the NPC (e.g., person, animal, object).
    :param tool_context: The tool context for accessing state.
    :return: A dictionary containing the generated response under the key 'result'.
    """
    console.print(f"[dim magenta]--- Tool: Generating response for NPC: [i]{npc_name}[/i] (Role: {npc_role}) ---[/dim magenta]")
    console.print(f"[dim magenta]   Received Message: '{received_message}'[/dim magenta]")

    # --- Prepare Context for LLM ---
    # Get relevant context from state using tool_context
    current_location = tool_context.state.get("simulacra_location", "an unknown location")
    current_time = tool_context.state.get("world_time", "an unknown time")
    # Optionally add other relevant state if needed, e.g., last event summary
    # last_narration = tool_context.state.get("last_narration", "")

    # --- Construct Prompt for LLM ---
    prompt = (
        f"You are role-playing as an NPC in a simulation named '{npc_name}'.\n"
        f"Your assigned role is: '{npc_role}'.\n"
        f"The current simulation time is '{current_time}' and the location is '{current_location}'.\n"
        f"The main character (Simulacra) just said the following to you: '{received_message}'\n\n"
        f"Instructions:\n"
        f"- Generate a brief, in-character response based on your name ({npc_name}), your role ({npc_role}), the message you received, and the current context.\n"
        f"- If your role is 'person', respond naturally.\n"
        f"- If your role is 'animal', respond with descriptive text representing sounds or actions appropriate for that animal.\n"
        f"- If your role is 'object', respond in a way that reflects the object's nature (e.g., describing a sensation, a passive observation, or silence if appropriate).\n"
        f"- If your role is 'abstract entity', respond creatively or cryptically, staying relevant.\n"
        f"- Keep the response concise (1-3 sentences usually).\n"
        f"- Do NOT break character. Respond ONLY as the NPC would.\n\n"
        f"NPC Response:"
    )

    # --- Call LLMService ---
    generated_text = f"({npc_name} does not respond.)" # Default response
    try:
        llm_service = LLMService(model_name=settings.MODEL_GEMINI_FLASH) # Use Flash or Pro as needed
        raw_llm_response = llm_service.generate_content_text(prompt=prompt)

        if raw_llm_response:
            # Clean up potential LLM artifacts like quoting its own response
            generated_text = raw_llm_response.strip().strip('"').strip("'")
        else:
            console.print(f"[yellow]Warning: LLM returned empty response for NPC {npc_name}.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]--- Tool Error (generate_npc_response LLM call): {e} ---[/bold red]")
        generated_text = f"({npc_name} seems unable to respond due to an error.)"

    # --- Format Final Output and Update State ---
    # Add NPC name for clarity, unless the response already implies it strongly
    # You might adjust this formatting based on how you want it presented
    if npc_role == "person" and not generated_text.lower().startswith(f"{npc_name.lower()} says"):
         final_response_string = f"{npc_name}: \"{generated_text}\""
    elif npc_role == "animal" or npc_role == "object" or npc_role == "abstract entity":
         final_response_string = f"({npc_name} - {npc_role}): {generated_text}" # Describe non-verbal response
    else: # Default or unknown role
         final_response_string = f"{npc_name}: {generated_text}"


    console.print(f"[dim magenta]--- Tool: Generated NPC Response: {final_response_string} ---[/dim magenta]")

    # Save the final response to state for the Narrator (Step 7/8)
    tool_context.state['last_npc_interaction'] = final_response_string

    # Return the result dictionary as expected by the framework/Narrator
    return {"result": final_response_string}