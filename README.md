# TheSimulation

Exploring Simulation Theory through Large Language Models

## Project Overview

**Status:** Under Development - Core agent logic implemented.

TheSimulation is an experiment exploring simulation theory utilizing Large Language Models (LLMs). This project investigates how LLMs, specifically through frameworks like the Google AI Developer Kit (ADK), can be used to create autonomous agents ("Simulacra") that perceive, reflect, and act within a simulated environment.

The project aims to use LLMs to generate and analyze diverse simulated scenarios by endowing agents with personas, goals, and a structured thinking process. Potential applications include:

- Ancestral simulations
- Contemporary social and cultural simulations
- Fictional world simulations
- Future scenario projections

## Practical Applications

This research has potential practical applications across various fields:

1.  **Consumer Research and Product Development:** Simulating consumer behavior, testing product concepts.
2.  **Political Science:** Modeling political systems, simulating policy impacts.
3.  **Environmental Studies:** Projecting climate scenarios, modeling ecosystem responses.
4.  **Urban Planning:** Simulating city development, modeling traffic patterns.
5.  **Educational Tools:** Developing immersive historical simulations, creating interactive learning environments.
6.  **Psychological Research:** Simulating social interactions, exploring cognitive processes.
7.  **Crisis Management:** Simulating disaster scenarios, modeling response strategies.
8.  **Entertainment Industry:** Developing rich fictional worlds, simulating audience reactions.
9.  **Ethical Decision Making:** Creating complex moral dilemmas for analysis.

## Framework: Language as Compressed Reality & Agent Architecture

### Language as Compressed Reality

The conceptual basis is that language functions as a compressed representation of reality, developed through consensus. LLMs leverage this compression to understand and generate complex information, making them suitable for simulating aspects of reality. (See original README section for more detail on "Invariant Representations" like the concept of "Dog").

### Agent Architecture (Simulacra V3)

The core simulation entities ("Simulacra") are implemented using the Google AI Developer Kit (ADK). Each Simulacrum is structured as a `SequentialAgent` that follows a specific thinking process:

1.  **Observe:** An `LlmAgent` perceives the current world state, its location, personal status (mood, physical condition), and social context from the simulation's shared state (`Session`).
2.  **Reflect:** Another `LlmAgent` takes the observation and performs an internal monologue. It considers its persona (traits, aspirations), current goal, status, and the observation to evaluate its situation, potentially considering alternative actions or goal adjustments.
3.  **Decide Intent:** A final `LlmAgent` processes the reflection and observation to determine the Simulacrum's next action. It outputs a structured JSON intent (e.g., `move`, `talk`, `use`, `wait`, `think`, `update_goal`) based on its prioritized needs and the plausibility within the simulation context.

This sequence uses LLMs (like Google's Gemini models) at each step, guided by specific instructions and drawing data from a shared `Session` state managed by the ADK.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Access to Google AI Studio and a Google API Key.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd TheSimulation
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Likely dependencies include: google-adk, google-generativeai, python-dotenv
    ```
4.  **Configure API Key:**
    - Create a `.env` file in the project root.
    - Add your Google API Key to the `.env` file:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      # Optional: Specify a Gemini model
      # MODEL_GEMINI_PRO="gemini-2.0-pro"
      ```
5.  **World Setup**
    - first run setup_simulation.py
    - python main3.py \* current working version

### Running the Agent Test

The `simulacra_v3.py` script includes a test function (`_test_agent`) to demonstrate the agent's thinking process in isolation.

```bash
python src/agents/simulacra_v3.py

This will:

Load the API key from .env.
Create a mock persona and session state.
Instantiate the Simulacra_V3 sequential agent.
Run the agent through one Observe-Reflect-Decide cycle using the ADK Runner.
Print the intermediate outputs (Observation, Reflection) and the final Intent JSON.

**Key Changes Made:**

1.  **Status Update:** Changed "NON FUNCTIONAL" to "Under Development - Core agent logic implemented."
2.  **Project Overview:** Added mention of Google ADK and the Simulacra concept.
3.  **Framework:** Added a subsection explaining the "Agent Architecture (Simulacra V3)" based on the Observe-Reflect-Decide sequence using `SequentialAgent` and `LlmAgent` from `google-adk`.
4.  **Project Structure:** Added a basic directory structure reflecting the current code (`src/agents/simulacra_v3.py`).
5.  **Getting Started:** Added concrete steps for setup (cloning, venv, dependencies, API key configuration via `.env`) and instructions on how to run the test function in `simulacra_v3.py`.
6.  **Current Status and Future Directions:** Updated the current status to reflect the implemented V3 agent and outlined more specific next steps.
7.  **Placeholders:** Kept placeholders for sections like Contributing, Ethical Considerations, etc., but noted they need content.


## Project Insights & Future Directions

*(The following is a reflection on the project's status and potential based on the architecture developed in `src/simulation_async.py`)*

### Breathing Life into Code: Simulating Reality with LLMs in "TheSimulation"

What if we could simulate not just physics, but societies, cultures, or even individual lives? The "Simulation Hypothesis" has long been a topic of philosophical debate, but Large Language Models (LLMs) are giving us new tools to explore these ideas computationally. Our project, aptly named **TheSimulation**, dives headfirst into this exploration, using Google's Generative AI and the Agent Development Kit (ADK) to build a world populated by autonomous agents – the Simulacra.

### Current Architecture: A Glimpse Under the Hood

Based on the code structure (primarily in `src/simulation_async.py`), TheSimulation currently operates with a few key components orchestrated using Python's `asyncio` and the Google ADK:

1.  **The State:** A central Python dictionary holds the entire simulation state – world details, object states, location descriptions, and the current status of all active Simulacra. This state is loaded from and saved to JSON files, allowing simulations to be paused and resumed.
2.  **The Time Manager:** An asynchronous task advances simulation time, processes scheduled events (like action completions), and updates the simulation state accordingly.
3.  **The Event Bus:** A simple `asyncio.Queue` acts as a communication channel, primarily for agents to declare their intended actions.
4.  **The World Engine (LLM Agent):** This ADK `LlmAgent` listens for intents on the event bus. It uses an LLM (like Gemini) to interpret the agent's intended action within the context of the current world state (location, object properties, rules). It determines if the action is valid, calculates its duration, figures out the consequences (state changes), and generates a descriptive narrative of the outcome. We've worked on refining its prompt to ensure it focuses on results and uses sensory details.
5.  **The Simulacra/Boltzmann Brain (LLM Agents):** Each active agent is also an ADK `LlmAgent`. When idle, it:
    *   **Observes:** Gathers context from the simulation state (location, recent events, nearby objects/agents, its own status, and persona).
    *   **Reflects & Decides:** Feeds this context into its LLM, guided by a detailed prompt. The LLM generates an internal monologue (following a structured thinking process: Recall/React -> Analyze Goal -> Identify Options -> Prioritize/Choose) and decides on the next action (like `move`, `use`, `look_around`, `talk`, `wait`). This decision is formatted as a JSON "intent".
    *   **Declares Intent:** Puts the generated intent onto the event bus for the World Engine to process.
6.  **ADK Integration:** The `Runner`, `SessionService`, and the `MemoryService` from the ADK are used to manage the interaction flow with the LLM agents and handle features like tool calling.

### Observing the Simulation: What the Logs Might Tell Us

Examine the output both from console and the log file, which you will find some interesting and weird interactions which points to the need for further refinement. For whatever reason the world_engine seems to always narrate extremely horror movie esque settings regardless of the initial conditions...
"You wake up at home, a friend and cozy place"

To the narration agent: "The room is bleak without any features, there are no doors or windows..." or other strange and at times humours outputs.

![Alt text](output1.png)

*   **Simulacra Monologues:** Rich text showing the agent's internal reasoning – reacting to observations ("The door is locked, drat!"), considering its persona ("As Eleanor Vance, a brewery guide, maybe I should look for local history books?"), weighing options, and finally stating its chosen action.
*   **Simulacra Intents:** Clean JSON objects like `{"action_type": "use", "target_id": "door_main", "details": "try the handle"}` or `{"action_type": "look_around", "target_id": null, "details": ""}`.
*   **World Engine Resolutions:** JSON outputs from the World Engine detailing the action's validity, duration, any state changes (`results`), and the crucial narrative (e.g., `"narrative": "Eleanor Vance tries the handle of the main door, but it's firmly locked."`).
*   **Narrative Log:** A growing list showing the chronological story unfolding based on the World Engine's narratives: `[T10.5] Eleanor Vance looks around.`, `[T13.5] The room is dusty, containing only a table and a locked chest.`, `[T15.0] Eleanor Vance tries the handle of the chest, but it's firmly locked.`

### Future Work

While the core loop is functional, several exciting areas need development:

1.  **Longer-Term Memory:**
    *   **Current State:** We've just integrated ADK's `InMemoryMemoryService` to store the initial persona. Agents can use the `load_memory` tool to recall this static background. The main `memory_log` is still just a list in the state file, prone to growing large and lacking efficient search.
    *   **Next Steps:** Persistent, searchable memory. Replacing `InMemoryMemoryService` with `VertexAiRagMemoryService` (leveraging Vertex AI Vector Search). This would allow agents to semantically search *all* their past observations, actions, and reflections ("What did I learn about locked doors yesterday?", "Who did I talk to in the library?").

2.  **Multi-Agent Interaction:**
    *   **Current State:** The basic `talk` action exists. An agent can declare an intent to talk to another agent in the same location, and the World Engine bypasses the LLM to directly update the target's `last_observation`.
    *   **Next Steps:** This is very rudimentary. True multi-agent dialogue requires agents to process incoming messages, reflect on them, and formulate replies within their turn cycle. This might involve dedicated "conversation manager" logic or more sophisticated prompting for the Simulacra agents to handle dialogue turns naturally. Beyond simple Q&A, modeling social dynamics, relationship building, and group interactions is a long-term goal.

3.  **Real-World Sync:**
    *   **Current State:** The simulation runs on its own accelerated or decelerated clock (`SIMULATION_SPEED_FACTOR`).
    *   **Next Steps:**
        *   For certain applications (e.g., simulating contemporary events), linking the simulation clock to real-world external real-time events to influence the simulation. Prior implementation included this code base which is basic weather sync and google news searches to set the stage on what the world state is.  The hierachy is World, Regional, and Local Weather/News context.  Again this will use the memory service and a timed sync mechanism.
        * For non-real (ie Sci-Fi/Fantazy) worlds - split on world history generation is needed and many of the tasks will be offloaded either through on demand generation or based on similar structure life summaries are generated for the simulacras.
        * Real time interaction with simulacra - Mechanism to talk to the simulated individual - Either via Text/IM/Chat (in real word) to crystal ball / telepathy / disembodied voice... to interact with individual.
        * Ability to inject events - ie "You suddently meet your long lost high school best friend in the street" , "All of a sudden you are teleported to the moon." Or more practical - "You are shopping at a supermarket, you are faced with two different toothbrushes... which one do you choose and why?"
        * Detailed locality

4.  **External Tool Integration:**
    *   **Current State:** Agents have the `load_memory` tool.
    *   **Next Steps:** This is where things get *really* interesting. The ADK's tool framework is perfect for expanding agent capabilities. Imagine agents that can:
        *   **Search the Web:** Give them a tool to query Google Search for real-time information relevant to their goals.
        *   **Generate Images:** Allow an agent to "visualize" something by calling an image generation API (like Imagen).
        *   **Interact with Social Media:** Create a tool for an agent to post updates or read feeds on platforms like Bluesky (using its API).
        *   **Send/Receive Email:** Simulate communication with the "outside world."
        *   **Access Other APIs:** Connect to weather services, stock market data, translation tools, etc.
    Each tool requires defining the function, providing it to the `LlmAgent`, and instructing the agent on when and how to use it.
```
