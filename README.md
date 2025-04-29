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

Remember to create a `requirements.txt` file listing the necessary Python packages (like `google-adk`, `google-generativeai`, `python-dotenv`).
```
