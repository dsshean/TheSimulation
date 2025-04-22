# src/tools/npc_tools.py
from google.adk.tools.tool_context import ToolContext
from rich.console import Console

console = Console()

def generate_npc_response(received_message: str, npc_name: str, npc_role: str) -> dict:
    """
    Generates a response for the NPC based on the received message and its role.
    :param received_message: The message received by the NPC.
    :param npc_name: The name of the NPC.
    :param npc_role: The role or identity of the NPC (e.g., person, animal, object).
    :return: A dictionary containing the generated response.
    """
    console.print(f"[bold cyan]NPC Agent received message:[/bold cyan] {received_message}")
    console.print(f"[bold cyan]NPC Role:[/bold cyan] {npc_role}")

    # Example response logic based on the NPC's role
    if npc_role == "person":
        response = (
            f"{npc_name} says: 'Thank you for your question. Here's what I can tell you: "
            f"Based on my knowledge, {received_message.lower()}.'"
        )
    elif npc_role == "animal":
        response = (
            f"{npc_name} (an animal) responds with a series of gestures or sounds that seem to convey: "
            f"'I understand your question, but I can only respond in my own way.'"
        )
    elif npc_role == "object":
        response = (
            f"{npc_name} (an object) seems to emit a faint hum or vibration, as if acknowledging: "
            f"'I am just an object, but I sense your presence.'"
        )
    elif npc_role == "abstract entity":
        response = (
            f"{npc_name} (an abstract entity) responds in a cryptic tone: "
            f"'The answer lies within the question itself: {received_message}.'"
        )
    else:
        response = (
            f"{npc_name} (unknown role) says: 'I am not sure how to respond to that, but I will try my best: "
            f"{received_message}.'"
        )

    console.print(f"[bold green]NPC Agent response:[/bold green] {response}")
    return {"result": response}