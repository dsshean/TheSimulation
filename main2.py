import asyncio
import copy
import logging
import os
import random
import re  # For parsing agent names
import sys  # For exit
from typing import Any, Dict, List, Optional

from google.adk.agents import (BaseAgent, LlmAgent, ParallelAgent,
                               SequentialAgent)
# Import MemoryService implementations
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
# Import necessary tool components
from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.genai import types
from google.genai.types import Content, FunctionCall, FunctionResponse, Part

# --- Rich Console Setup ---
try:
    from rich.console import Console
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.rule import Rule
    console = Console()
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print(f"\n--- {args[0] if args else ''} ---")
        def padding(self, *args, **kwargs): print(*args) # Dummy padding
    console = DummyConsole()
    print("Rich console not found, using basic print for separators.")


# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="debate_simulation.log",
    filemode="w"
)
logger = logging.getLogger(__name__)


APP_NAME = "complex_debate_simulation"
USER_ID = "debate_user"
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest")
logger.info(f"Using model: {MODEL_NAME}")
console.print(f"Using model: [cyan]{MODEL_NAME}[/cyan]")

# --- Constants for Phases and Turns ---
MAX_INTERNAL_PLANNING_STEPS = 6 # How many back-and-forth steps
MAX_DEBATE_TURNS = 10 # Total back-and-forth turns between teams (5 per team)

# --- Potential Names for Team Members ---
POTENTIAL_NAMES = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]

# --- State Keys ---
DEBATE_TOPIC_KEY = "debate_topic"
TEAM_A_PLAN_KEY = "team_a_plan"
TEAM_B_PLAN_KEY = "team_b_plan"
TEAM_A_LATEST_ARG_KEY = "team_a_latest_argument"
TEAM_B_LATEST_ARG_KEY = "team_b_latest_argument"
JUDGEMENT_KEY = "judgement"
CURRENT_PHASE_KEY = "current_phase"
DEBATE_TURN_NUMBER_KEY = "debate_turn_number"

# --- Tools (Keep as they are, logging/printing handled in main loop) ---
async def save_team_plan_func(team_id: str, plan_or_initial_arg: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Saves the team's plan/initial argument after internal discussion."""
    if team_id not in ('A', 'B'): return {"status": "error", "message": "Invalid team_id."}
    state_key = TEAM_A_PLAN_KEY if team_id == 'A' else TEAM_B_PLAN_KEY
    logger.info(f"Tool: Saving plan for Team {team_id} to '{state_key}'.")
    console.print(f"[dim]Tool Call: Saving plan/initial argument for Team {team_id}...[/dim]")
    try:
        tool_context.state[state_key] = plan_or_initial_arg
        return {"status": "success", "message": f"Plan saved for Team {team_id}."}
    except Exception as e:
        logger.exception(f"Error saving plan for Team {team_id}: {e}")
        return {"status": "error", "message": str(e)}
save_plan_tool = FunctionTool(func=save_team_plan_func)

async def save_debate_argument_func(team_id: str, argument: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Saves the team's argument during the debate phase."""
    if team_id not in ('A', 'B'): return {"status": "error", "message": "Invalid team_id."}
    state_key = TEAM_A_LATEST_ARG_KEY if team_id == 'A' else TEAM_B_LATEST_ARG_KEY
    logger.info(f"Tool: Saving debate argument for Team {team_id} to '{state_key}'.")
    console.print(f"[dim]Tool Call: Saving debate argument for Team {team_id}...[/dim]")
    try:
        tool_context.state[state_key] = argument
        return {"status": "success", "message": f"Debate argument saved for Team {team_id}."}
    except Exception as e:
        logger.exception(f"Error saving debate argument for Team {team_id}: {e}")
        return {"status": "error", "message": str(e)}
save_debate_argument_tool = FunctionTool(func=save_debate_argument_func)

async def search_memory_wrapper(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Searches memory."""
    logger.info(f"Tool 'search_memory_wrapper' called with query: '{query}'")
    console.print(f"[dim]Tool Call: Searching memory with query: '{query}'...[/dim]")
    if not hasattr(tool_context, 'search_memory'):
         logger.error("ToolContext does not have search_memory method.")
         return {"status": "error", "message": "Memory search capability not available."}
    try:
        search_response: Optional[Any] = await tool_context.search_memory(query=query)
        if search_response and hasattr(search_response, 'results') and search_response.results:
            results_texts = []
            for entry in search_response.results:
                 if (hasattr(entry, 'retrieved_context') and entry.retrieved_context and
                     hasattr(entry.retrieved_context, 'content') and entry.retrieved_context.content and
                     hasattr(entry.retrieved_context.content, 'parts') and entry.retrieved_context.content.parts):
                     for part in entry.retrieved_context.content.parts:
                         if hasattr(part, 'text') and part.text: results_texts.append(part.text)
            MAX_CONTEXT_LENGTH = 500
            retrieved_text = "\n---\n".join(results_texts)
            if len(retrieved_text) > MAX_CONTEXT_LENGTH: retrieved_text = retrieved_text[:MAX_CONTEXT_LENGTH] + "..."
            logger.info(f"Memory search successful. Found {len(results_texts)} snippets.")
            console.print(f"[dim]Tool Response: Memory search found {len(results_texts)} snippets.[/dim]")
            return {"status": "success", "results_found": len(results_texts) > 0, "retrieved_context": retrieved_text if results_texts else "No specific text found."}
        else:
            logger.info("Memory search completed, but no results found.")
            console.print("[dim]Tool Response: Memory search found no results.[/dim]")
            return {"status": "success", "results_found": False, "retrieved_context": "No relevant information found in memory."}
    except Exception as e:
        error_msg = f"Error during memory search for query '{query}': {e}"
        logger.exception(error_msg)
        console.print(f"[red]Error during memory search: {e}[/red]")
        return {"status": "error", "message": error_msg}
memory_search_tool = FunctionTool(func=search_memory_wrapper)


# --- Agent Definitions ---

# 1. Topic Picker Agent
topic_picker_agent = LlmAgent(
    name="TopicPickerAgent", model=MODEL_NAME,
    instruction="You are the debate moderator. Pick a random, interesting, and debatable topic suitable for nuanced discussion. Output *only* the topic name.",
    description="Picks a debate topic.", output_key=DEBATE_TOPIC_KEY
)

# --- Phase 1: Internal Planning Agents ---

# MODIFIED: Enhanced instructions for deeper planning
def create_team_member_agent(team_id: str, member_name: str, step_num: int, total_steps: int) -> LlmAgent:
    """Creates an agent representing one step in the internal team planning."""
    output_key = f"team_{team_id}_planning_step_{step_num}_output"
    previous_step_key = f"team_{team_id}_planning_step_{step_num-1}_output" if step_num > 1 else None
    agent_name = f"Team{team_id}_Planner_{member_name}_Step{step_num}"

    instruction_context = f"""You are {member_name} of Team {team_id}, participating in an internal strategy session for an upcoming debate.
The debate topic is: {{{DEBATE_TOPIC_KEY}}}
This is step {step_num} of {total_steps} in your team's planning. Your goal is to develop a robust and nuanced strategy."""

    if previous_step_key:
        instruction_context += f"\nThe previous contribution from your teammate was: {{{previous_step_key}}}"
        instruction_context += "\nCritically evaluate this point. What are its strengths and weaknesses? How can it be refined or better supported? What counter-arguments might the opponent raise?"
    else:
        instruction_context += "\nYou are initiating the strategy discussion. Propose a strong core argument or strategic angle for the team."

    # Enhanced task instruction
    instruction_task = f"""
Your task: Contribute thoughtfully to your team's strategy. You can propose a core argument, suggest supporting evidence or examples, identify and preemptively counter potential opponent arguments, or critically refine/build upon the previous point. Aim for depth, clarity, and strategic thinking. Keep it concise (2-3 sentences).
Output *only* your contribution for this step."""

    return LlmAgent(
        name=agent_name, model=MODEL_NAME,
        instruction=instruction_context + instruction_task,
        description=f"Internal planning step {step_num} for Team {team_id} by {member_name}.",
        output_key=output_key
    )

# MODIFIED: Captain's instruction slightly refined for clarity
def create_team_captain_planner_agent(team_id: str, member_names: List[str], total_steps: int) -> LlmAgent:
    """Creates the final agent in the planning sequence to summarize and save the plan."""
    final_step_key = f"team_{team_id}_planning_step_{total_steps}_output"
    final_speaker_index = (total_steps - 1) % 2
    final_speaker_name = member_names[final_speaker_index]

    return LlmAgent(
        name=f"Team{team_id}_CaptainPlanner", model=MODEL_NAME,
        instruction=f"""You are the Captain of Team {team_id}. Your team members are {member_names[0]} and {member_names[1]}. You just finished an internal planning discussion.
The debate topic is: {{{DEBATE_TOPIC_KEY}}}
The final contribution in the discussion (from {final_speaker_name}) was: {{{final_step_key}}}

Your task: Synthesize the key strategic points and the core initial argument your team decided upon based on the entire discussion. Ensure it's a strong opening position.
Then, call the 'save_team_plan_func' tool with:
- team_id: "{team_id}"
- plan_or_initial_arg: [Your synthesized plan/initial argument text]

Output *only* a brief confirmation like 'Team {team_id} plan finalized and saved.' AFTER calling the tool.
""",
        description=f"Summarizes and saves Team {team_id}'s plan.",
        tools=[save_plan_tool]
    )

# create_team_planner_agent remains structurally the same, just uses the modified member/captain agents above.
def create_team_planner_agent(team_id: str, member_names: List[str], num_steps: int) -> SequentialAgent:
    """Creates the sequential agent for a team's internal planning phase."""
    planning_steps = []
    for i in range(1, num_steps + 1):
        speaker_index = (i - 1) % 2
        speaker_name = member_names[speaker_index]
        planning_steps.append(create_team_member_agent(team_id, speaker_name, i, num_steps))

    captain_agent = create_team_captain_planner_agent(team_id, member_names, num_steps)
    return SequentialAgent(
        name=f"Team{team_id}_Planner",
        sub_agents=planning_steps + [captain_agent],
        description=f"Orchestrates internal planning for Team {team_id} with {member_names[0]} and {member_names[1]}."
    )


# --- Phase 2: Inter-Team Debate Agents ---

# MODIFIED: Enhanced instructions for deeper debate/rebuttal
def create_debate_turn_agent(team_id: str, turn_num_overall: int) -> LlmAgent:
    """Creates an agent for one turn in the inter-team debate phase."""
    my_plan_key = TEAM_A_PLAN_KEY if team_id == 'A' else TEAM_B_PLAN_KEY
    my_latest_arg_key = TEAM_A_LATEST_ARG_KEY if team_id == 'A' else TEAM_B_LATEST_ARG_KEY
    opponent_latest_arg_key = TEAM_B_LATEST_ARG_KEY if team_id == 'A' else TEAM_A_LATEST_ARG_KEY

    if turn_num_overall == 1 and team_id == 'A':
        opponent_context = "The opponent (Team B) has not presented an argument yet. Present your team's opening argument based on your plan."
    else:
        opponent_context = f"The opponent's (Team {'B' if team_id == 'A' else 'A'}) most recent argument was: {{{opponent_latest_arg_key}}}"
        opponent_context += "\nAnalyze their argument: What is the core claim? What assumptions are made? Are there logical flaws or missing evidence?"

    return LlmAgent(
        name=f"Team{team_id}_DebateTurn_{turn_num_overall}", model=MODEL_NAME,
        instruction=f"""You are representing Team {team_id} during a formal debate. This is overall turn {turn_num_overall} / {MAX_DEBATE_TURNS}.
The debate topic is: {{{DEBATE_TOPIC_KEY}}}
Your team's initial strategy/plan was: {{{my_plan_key}}}
Your team's previous argument in this debate (if any) was: {{{my_latest_arg_key}}}
{opponent_context}

Your task: Present a compelling and well-reasoned argument for this turn.
{'Directly address and refute the opponent\'s last point, exposing weaknesses or flawed assumptions.' if turn_num_overall > 1 else 'Present your strong opening argument based on your plan.'}
Then, strongly advance your team's position, linking back to your core strategy. Consider the broader implications or nuances of the topic. Aim for persuasive language and logical depth. Keep it concise but impactful (2-4 sentences).
Use the 'search_memory_wrapper' tool if specific historical context from prior turns is needed.

After formulating your argument, call the 'save_debate_argument_func' tool with:
- team_id: "{team_id}"
- argument: [Your argument text for this turn]

Output *only* a brief confirmation like 'Team {team_id} argument for turn {turn_num_overall} presented and saved.' AFTER calling the tool.
""",
        description=f"Presents Team {team_id}'s argument for debate turn {turn_num_overall}.",
        tools=[save_debate_argument_tool, memory_search_tool]
    )

# --- Phase 3: Judging Agent (No change needed, but benefits from deeper arguments) ---
judge_agent = LlmAgent(
    name="JudgeAgent", model=MODEL_NAME,
    instruction=f"""You are the debate judge. The debate, consisting of internal planning and {MAX_DEBATE_TURNS} turns of argument/rebuttal, has concluded.
The topic was: {{{DEBATE_TOPIC_KEY}}}
Team A's initial plan was: {{{TEAM_A_PLAN_KEY}}}
Team B's initial plan was: {{{TEAM_B_PLAN_KEY}}}
Team A's final argument presented was: {{{TEAM_A_LATEST_ARG_KEY}}}
Team B's final argument presented was: {{{TEAM_B_LATEST_ARG_KEY}}}
Review the initial plans and the final arguments. Use the 'search_memory_wrapper' tool to recall specific arguments made during the debate turns if necessary (e.g., query='Team A argument turn 3').
Declare an overall winner for the debate and provide a brief (2-3 sentences) justification, considering consistency with the plan, responsiveness to rebuttals, the logical depth of arguments, and the strength of the final arguments.
Output *only* the winner declaration and justification.
""",
    description="Judges the entire debate based on plans and final arguments.",
    tools=[memory_search_tool],
    output_key=JUDGEMENT_KEY
)

# --- Overall Orchestrator Agent & Helper Functions (No changes needed) ---
def create_phase_update_agent(phase_name: str) -> LlmAgent:
    return LlmAgent(
        name=f"SetPhase_{phase_name}", model=MODEL_NAME,
        instruction=f"Update the state key '{CURRENT_PHASE_KEY}' to '{phase_name}'. Output only 'Phase set to {phase_name}'.",
        output_key=CURRENT_PHASE_KEY
    )

def print_planning_conversation(team_id: str, member_names: List[str], planning_log: Dict[int, str]):
    """Formats and prints the internal planning conversation."""
    console.print(Rule(f"Team {team_id} Internal Planning Summary ({member_names[0]} & {member_names[1]})", style="yellow"))
    if not planning_log:
        console.print(Padding("[i]No planning steps logged for this team.[/i]", (0, 2)))
        return

    for step in sorted(planning_log.keys()):
        speaker_index = (step - 1) % 2
        speaker_name = member_names[speaker_index]
        text = planning_log[step]
        console.print(Padding(f"[bold]{speaker_name}:[/bold] {text}", (0, 2))) # Indent lines
    console.print(Rule(style="yellow"))


# --- Main Simulation Logic ---
async def main():
    """Runs the complex, multi-phase debate simulation."""
    session_id = f"complex_debate_{random.randint(1000, 9999)}"
    console.print(Rule(f"Starting Complex Debate Simulation (Session: {session_id})", style="bold green"))

    # --- Assign Member Names ---
    if len(POTENTIAL_NAMES) < 4:
        print("Error: Need at least 4 potential names.")
        return
    assigned_names = random.sample(POTENTIAL_NAMES, 4)
    team_a_names = assigned_names[0:2]
    team_b_names = assigned_names[2:4]
    console.print(f"Team A Members: [bold cyan]{team_a_names[0]}[/] & [bold cyan]{team_a_names[1]}[/]")
    console.print(f"Team B Members: [bold cyan]{team_b_names[0]}[/] & [bold cyan]{team_b_names[1]}[/]")

    # --- Create Agents (Now passing names) ---
    team_a_planner = create_team_planner_agent('A', team_a_names, MAX_INTERNAL_PLANNING_STEPS)
    team_b_planner = create_team_planner_agent('B', team_b_names, MAX_INTERNAL_PLANNING_STEPS)
    planning_phase_agent = ParallelAgent(
        name="PlanningPhase", sub_agents=[team_a_planner, team_b_planner],
        description="Runs internal planning for both teams concurrently."
    )
    debate_phase_agents = []
    for i in range(1, MAX_DEBATE_TURNS + 1):
        current_team = 'A' if (i % 2 != 0) else 'B'
        debate_phase_agents.append(create_debate_turn_agent(current_team, i))
    main_orchestrator = SequentialAgent(
        name="ComplexDebateOrchestrator",
        sub_agents=[
            topic_picker_agent,
            create_phase_update_agent("Planning"),
            planning_phase_agent,
            create_phase_update_agent("Debate"), # Signal to print planning logs
            *debate_phase_agents,
            create_phase_update_agent("Judging"),
            judge_agent
        ],
        description="Orchestrates the complex debate with planning, debate, and judging phases."
    )

    # --- Services and Runner Setup ---
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    runner = Runner(
        agent=main_orchestrator, app_name=APP_NAME,
        session_service=session_service, memory_service=memory_service
    )

    # --- Initialize State ---
    initial_state = {
        DEBATE_TOPIC_KEY: "Not selected", TEAM_A_PLAN_KEY: "Not planned",
        TEAM_B_PLAN_KEY: "Not planned", TEAM_A_LATEST_ARG_KEY: "No arguments yet",
        TEAM_B_LATEST_ARG_KEY: "No arguments yet", JUDGEMENT_KEY: "Not judged",
        CURRENT_PHASE_KEY: "Initialization", DEBATE_TURN_NUMBER_KEY: 0
    }
    session = session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state=initial_state
    )
    logger.info(f"Session created with initial state: {session.state}")

    # --- Run Simulation ---
    trigger_text = "Begin the debate simulation."
    trigger = types.Content(parts=[types.Part(text=trigger_text)])
    console.print(f"\n[bold yellow]>>> Triggering Debate Sequence...[/bold yellow]")

    team_a_planning_log: Dict[int, str] = {}
    team_b_planning_log: Dict[int, str] = {}
    planning_logs_printed = False
    current_phase = "Initialization"

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=trigger
        ):
            author = event.author
            is_final = event.is_final_response()
            prefix = f"[cyan]{author}[/cyan]{' (Final)' if is_final else ''}:"

            # --- Capture Planning Phase Output ---
            planning_match = re.match(r"Team(A|B)_Planner_(.+)_Step(\d+)", author)
            if planning_match and is_final and event.content and event.content.parts:
                team_id = planning_match.group(1)
                step_num = int(planning_match.group(3))
                text = event.content.parts[0].text.strip()
                if team_id == 'A':
                    team_a_planning_log[step_num] = text
                else:
                    team_b_planning_log[step_num] = text

            # --- Phase Tracking and Printing Planning Logs ---
            # Use elif structure to avoid processing multiple conditions per event
            if author.startswith("SetPhase_"): # Check if it's a phase setting event first
                if is_final: # Only process final phase setting events
                    new_phase = author.split("_")[-1]
                    if new_phase == "Debate" and not planning_logs_printed:
                        # Handle transition TO Debate specifically
                        console.rule("End of Planning Phase", style="bold blue")
                        print_planning_conversation('A', team_a_names, team_a_planning_log)
                        print_planning_conversation('B', team_b_names, team_b_planning_log)
                        planning_logs_printed = True
                        console.rule(f"Entering Phase: Debate", style="bold blue")
                        current_phase = "Debate" # Assign the new phase
                        # Update state directly
                        current_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                        if current_session: current_session.state[CURRENT_PHASE_KEY] = current_phase
                    elif new_phase != current_phase: # Handle other phase changes (e.g., Judging)
                        console.rule(f"Entering Phase: {new_phase}", style="bold blue")
                        current_phase = new_phase # Assign the new phase
                        current_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                        if current_session: current_session.state[CURRENT_PHASE_KEY] = current_phase

            # --- Turn Number Tracking ---
            # Now check the current_phase AFTER it might have been updated above
            elif current_phase == "Debate" and author.startswith("Team") and "DebateTurn" in author and is_final:
                 try:
                     turn_str = author.split("DebateTurn_")[-1].split("_")[0]
                     turn_match = [int(s) for s in turn_str if s.isdigit()]
                     if turn_match:
                         overall_turn = turn_match[0]
                         current_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                         if current_session and current_session.state.get(DEBATE_TURN_NUMBER_KEY) != overall_turn:
                             current_session.state[DEBATE_TURN_NUMBER_KEY] = overall_turn
                             logger.info(f"Updated DEBATE_TURN_NUMBER_KEY to {overall_turn}")
                             console.print(f"[dim]State Update: Debate Turn set to {overall_turn}[/dim]")
                 except Exception as e:
                     logger.warning(f"Could not parse debate turn number from agent name {author}: {e}")

            # --- Console Event Logging (Print debate args) ---
            elif event.get_function_calls():
                call = event.get_function_calls()[0]
                console.print(f"{prefix} [magenta]Function Call[/magenta] -> {call.name}(...)")
                if call.name == "save_debate_argument_func":
                    try:
                        arg_text = call.args.get("argument", "[Argument text not found]")
                        team_id = call.args.get("team_id", "?")
                        # Use Padding for consistent indentation
                        console.print(Padding(f"[bold]Team {team_id} Arg:[/bold] {arg_text}", (0, 2)))
                    except Exception as e:
                        logger.warning(f"Could not extract/print argument from save_debate_argument_func call: {e}")
            elif event.get_function_responses():
                resp = event.get_function_responses()[0]
                response_str = str(resp.response)[:100] + ('...' if len(str(resp.response)) > 100 else '')
                console.print(f"{prefix} [magenta]Function Response[/magenta] -> {resp.name} = {response_str}")
            elif event.content and event.content.parts and event.content.parts[0].text:
                text_snippet = event.content.parts[0].text.strip()
                # Avoid printing phase/captain confirmations
                if not author.startswith("SetPhase_") and "CaptainPlanner" not in author:
                     if not planning_match: # Avoid printing planning steps again
                         console.print(f"{prefix} [green]Says[/green] -> '{text_snippet}'")
            elif event.actions and event.actions.state_delta:
                 logger.info(f"Event: {author} StateDelta: {event.actions.state_delta}")
            elif event.error_message:
                console.print(f"{prefix} [bold red]Error[/bold red] -> Code: {event.error_code}, Msg: {event.error_message}")

            # Log to file
            logger.info(f"Event: Author={author}, Final={is_final}, Content={event.content}, Actions={event.actions}, Error={event.error_message}")

        # --- Debate finished ---
        console.print(f"\n[bold yellow]<<< Debate Sequence Finished.[/bold yellow]")
        final_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        if not final_session:
            logger.error(f"Critical Error: Could not retrieve final session {session_id}.")
            console.print(f"[bold red]Critical Error: Could not retrieve final session state.[/bold red]")
            return

        session = final_session
        final_state = session.state

        # --- Display Final Summary ---
        console.print(Rule("Final Debate Summary", style="bold green"))
        summary_text = f"""
[bold]Topic:[/bold] {final_state.get(DEBATE_TOPIC_KEY, 'N/A')}
[bold]Team A Initial Plan:[/bold] {final_state.get(TEAM_A_PLAN_KEY, 'N/A')}
[bold]Team B Initial Plan:[/bold] {final_state.get(TEAM_B_PLAN_KEY, 'N/A')}
[bold]Team A Final Argument:[/bold] {final_state.get(TEAM_A_LATEST_ARG_KEY, 'N/A')}
[bold]Team B Final Argument:[/bold] {final_state.get(TEAM_B_LATEST_ARG_KEY, 'N/A')}
[bold]Final Judgement:[/bold] {final_state.get(JUDGEMENT_KEY, 'N/A')}
        """
        console.print(Panel(summary_text.strip(), title="Debate Outcome", border_style="green"))

        # --- Add completed debate session data to Memory ---
        logger.info(f"--- Adding Final Debate Session to Memory ---")
        console.print("[dim]Attempting to add final results to memory...[/dim]")
        method_to_call = None
        if hasattr(memory_service, 'add_session_to_memory'):
            method_to_call = getattr(memory_service, 'add_session_to_memory')
            logger.debug(f"Memory service object ID before await: {id(memory_service)}") # Add ID check

        if method_to_call and asyncio.iscoroutinefunction(method_to_call):
            try:
                await method_to_call(final_session)
                logger.info("Session added to memory successfully.")
                console.print("[dim]Memory updated successfully.[/dim]")
            except Exception as mem_e:
                logger.exception(f"Error during memory_service.add_session_to_memory call: {mem_e}")
                console.print(f"[red]Error adding session to memory: {mem_e}[/red]")
        else:
            logger.error(f"Cannot call 'add_session_to_memory'. Method is None or not an async function. Skipping memory add.")
            console.print("[red]Error: Cannot add session to memory (method invalid).[/red]")

        # --- REMOVED Memory Contents Inspection Block ---


    except Exception as e:
        logger.exception(f"An error occurred during the debate sequence: {e}")
        console.print(f"[bold red]An error occurred during the debate: {e}[/bold red]")
        console.print_exception(show_locals=False)

    console.print(Rule("Debate Simulation Finished", style="bold green"))
    logger.info(f"\n{'='*20} Debate Simulation Finished {'='*20}")


# --- Main Execution Block ---
if __name__ == "__main__":
    if 'GOOGLE_API_KEY' not in os.environ:
        console.print("[bold red]ERROR: Please set the GOOGLE_API_KEY environment variable.[/bold red]")
        logger.critical("GOOGLE_API_KEY environment variable not set. Exiting.")
        sys.exit(1)
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        logger.info("Google Generative AI configured successfully.")
    except ImportError:
         console.print("[bold red]ERROR: google.generativeai library not found.[/bold red]")
         logger.critical("google.generativeai library not found.")
         sys.exit(1)
    except Exception as e:
         console.print(f"[bold red]ERROR: Failed to configure Google Generative AI: {e}[/bold red]")
         logger.critical(f"Failed to configure Google Generative AI: {e}")
         sys.exit(1)

    try:
        # Add initial check for memory service method right after creation
        temp_memory_service = InMemoryMemoryService()
        logger.debug(f"INITIAL CHECK - Type of memory_service: {type(temp_memory_service)}")
        logger.debug(f"INITIAL CHECK - Memory service object ID: {id(temp_memory_service)}")
        initial_method = None
        if hasattr(temp_memory_service, 'add_session_to_memory'):
            initial_method = getattr(temp_memory_service, 'add_session_to_memory')
            logger.debug(f"INITIAL CHECK - Has 'add_session_to_memory' attr: True")
            logger.debug(f"INITIAL CHECK - Type of method attribute: {type(initial_method)}")
            logger.debug(f"INITIAL CHECK - Is method callable: {callable(initial_method)}")
            is_coroutine = asyncio.iscoroutinefunction(initial_method)
            logger.debug(f"INITIAL CHECK - Is method a coroutine function: {is_coroutine}")
            logger.debug(f"INITIAL CHECK - Method object itself: {initial_method}")
        else:
             logger.error("INITIAL CHECK - CRITICAL: memory_service object DOES NOT HAVE 'add_session_to_memory' attribute upon creation!")
        del temp_memory_service # Clean up temporary object

        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
        logger.warning("Simulation interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in the main execution: {e}")
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False)
    finally:
        logging.shutdown()
        console.print("Application finished.")
