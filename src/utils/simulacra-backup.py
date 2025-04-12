# src/simulacra.py
import json
import logging
import os
import time # Added for RAG timing if needed
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel

# --- Existing Imports ---
from src.llm_service import LLMService
from src.prompt_manager import PromptManager
from src.utils.llm_utils import generate_and_validate_llm_response
from src.models import EmotionAnalysisResponse, ActionDecisionResponse, DayResponse
# --- End Existing Imports ---

# --- RAG Imports ---
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- End RAG Imports ---


logger = logging.getLogger(__name__)

# --- Default Persona Structure (Helper) ---
DEFAULT_PERSONA = {
    "name": "Default Persona", "age": 30, "occupation": "Worker",
    "personality_traits": ["Adaptable", "Observant"], "goals": ["Get through the day"],
    "current_state": {"physical": "Normal", "emotional": "Neutral", "mental": "Aware"},
    "memory": {"short_term": [], "long_term": []}
}
# --- End Default Persona ---

# --- RAG Constants ---
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"
CHROMA_DB_PATH = os.path.join("db", "chroma_simulacra") # Store DB in db/chroma_simulacra
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
RAG_QUERY_RESULTS = 3 # Number of chunks to retrieve
# --- End RAG Constants ---


class Simulacra:
    """Represents a simulated human with personality, goals, and behaviors."""

    def __init__(self,
                 life_summary_path: Optional[str], # Path for INITIAL persona
                 delta_state_path: str = "simulacra_deltas.json", # Path for RUNTIME state
                 console: Optional[Console] = None,
                 new_simulacra: bool = False):
        """
        Initialize the Simulacra.
        Loads runtime state from delta_state_path if it exists.
        Otherwise, initializes persona from life_summary_path.
        Falls back to default if needed. Includes RAG setup.

        Args:
            life_summary_path: Path to the life summary JSON file (for initial persona).
            delta_state_path: Path to save/load runtime persona and history.
            console: Rich console for output.
            new_simulacra: If True, ignore delta file and re-index RAG based on life summary.
        """
        self.console = console or Console()
        self.state_path = delta_state_path
        self.life_summary_path = life_summary_path
        self.persona: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.llm_service = LLMService() # Assumes LLMService setup handles API keys internally or globally

        # --- RAG Setup Attributes ---
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.memory_collection: Optional[chromadb.Collection] = None
        self.embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
        self.collection_name: Optional[str] = None
        # --- End RAG Setup Attributes ---

        loaded_from_delta = False

        # --- Handle --new Flag ---
        if new_simulacra:
            logger.info("`new_simulacra` flag set. Will initialize from life summary and force re-indexing.")
            self.console.print("[yellow]--new flag set: Initializing from life summary, ignoring delta file.[/yellow]")
            # Don't load default here yet, proceed to life summary load below
        else:
            # --- Attempt 1: Load from Delta State File ---
            if os.path.exists(self.state_path):
                 logger.info(f"Attempting to load runtime state from: {self.state_path}")
                 try:
                     with open(self.state_path, 'r', encoding='utf-8') as file:
                         delta_data = json.load(file)
                     if isinstance(delta_data.get("persona"), dict) and isinstance(delta_data.get("history"), list):
                         self.persona = delta_data["persona"]
                         self.history = delta_data["history"]
                         required_keys = ["name", "age", "occupation", "personality_traits", "current_state", "memory"]
                         if not all(key in self.persona for key in required_keys):
                            logger.warning(f"Loaded persona from delta file '{self.state_path}' missing core keys. Falling back.")
                            self.persona = {}
                            self.history = []
                         else:
                             logger.info(f"Successfully loaded runtime state for '{self.persona.get('name', 'Unknown')}' from {self.state_path}")
                             self.console.print(f"[green]Loaded runtime state for {self.persona.get('name', 'Unknown')} from {self.state_path}[/green]")
                             loaded_from_delta = True
                     else:
                         logger.warning(f"Invalid structure in delta file '{self.state_path}'. Falling back.")
                 except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                     logger.error(f"Failed to load or parse delta state file '{self.state_path}': {e}. Falling back.", exc_info=True)
                     self.console.print(f"[red]Error loading delta state file '{self.state_path}'. Trying life summary.[/red]")
            else:
                 logger.info(f"Delta state file '{self.state_path}' not found. Trying life summary.")
                 self.console.print(f"[yellow]Delta state file '{self.state_path}' not found. Trying life summary.[/yellow]")

        # --- Attempt 2: Load Initial Persona from Life Summary (if Delta failed OR --new flag) ---
        if not loaded_from_delta: # This condition is met if delta load failed OR if --new was used
            self.history = [] # Reset history if loading initial persona
            if not self.life_summary_path or not os.path.exists(self.life_summary_path):
                logger.warning(f"Life summary file not found: '{self.life_summary_path}'. Using default persona.")
                self.console.print(f"[yellow]Life summary file '{self.life_summary_path}' not found. Using default persona.[/yellow]")
                self.persona = DEFAULT_PERSONA.copy()
            else:
                 logger.info(f"Loading initial persona from life summary: {self.life_summary_path}")
                 try:
                     with open(self.life_summary_path, 'r', encoding='utf-8') as file:
                         life_data = json.load(file)
                     persona_details = life_data.get("persona_details") # Using your structure

                     if isinstance(persona_details, dict):
                         self.persona = {key.lower(): value for key, value in persona_details.items()}
                         self.persona.setdefault("current_state", {"physical": "Normal", "emotional": "Neutral", "mental": "Aware"})
                         self.persona.setdefault("memory", {"short_term": [], "long_term": []})

                         # Your logic to add initial short-term memory from yearly summary
                         latest_year_key = str(life_data.get("birth_year", 0) + life_data.get("age", 0))
                         latest_year_summary = life_data.get("yearly_summaries", {}).get(latest_year_key)
                         if latest_year_summary and isinstance(latest_year_summary, list):
                             full_summary_text = latest_year_summary[0] # Get the full text
                             self.persona["memory"]["short_term"].append(f"Context from age {life_data.get('age', '?')}: {full_summary_text}")

                         required_keys = ["name", "age", "occupation", "personality_traits", "current_state", "memory"]
                         if not all(key in self.persona for key in required_keys):
                             logger.warning(f"Loaded persona details from '{self.life_summary_path}' missing core keys. Reverting to default.")
                             self.console.print(f"[yellow]Invalid persona structure in '{self.life_summary_path}'. Using default.[/yellow]")
                             self.persona = DEFAULT_PERSONA.copy()
                         else:
                             logger.info(f"Successfully loaded initial persona '{self.persona.get('name', 'Unknown')}' from {self.life_summary_path}")
                             self.console.print(f"[green]Loaded initial persona: {self.persona.get('name', 'Unknown')} from {self.life_summary_path}[/green]")
                     else:
                         logger.error(f"Missing or invalid 'persona_details' in '{self.life_summary_path}'. Using default persona.")
                         self.console.print(f"[red]Missing 'persona_details' in '{self.life_summary_path}'. Using default.[/red]")
                         self.persona = DEFAULT_PERSONA.copy()

                 except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                      logger.error(f"Failed to load/parse life summary '{self.life_summary_path}': {e}. Using default.", exc_info=True)
                      self.console.print(f"[red]Error loading life summary '{self.life_summary_path}'. Using default.[/red]")
                      self.persona = DEFAULT_PERSONA.copy()

        # --- Final Fallback (If somehow persona is still empty) ---
        if not self.persona:
             logger.critical("Failed to load persona from any source. Using default.")
             self.console.print("[bold red]CRITICAL: Failed to load persona from any source. Using default.[/bold red]")
             self.persona = DEFAULT_PERSONA.copy()
             self.history = []
        # --- End Loading Logic ---

        # --- Setup Vector DB *after* persona is confirmed loaded ---
        if self.persona and self.persona.get('name'):
            # Pass new_simulacra flag to potentially force re-indexing
            self._setup_vector_db(new_simulacra)
        else:
            logger.error("Could not setup vector DB because persona name is missing after initialization.")
            self.console.print("[bold red]Error: Could not initialize RAG memory system - persona invalid after load.[/bold red]")

# src/simulacra.py

    # --- RAG Method: Setup Vector DB ---
    def _setup_vector_db(self, force_reindex: bool):
        """Initializes ChromaDB client, collection, and indexes the life summary if needed or forced.
           Uses .get(limit=1) to check for existing items, avoiding .count()."""
        persona_name = self.persona.get('name', 'unknown_persona')
        logger.info(f"Setting up vector DB for persona: {persona_name} (Force Re-index: {force_reindex})")

        try:
            # --- Gemini config, Chroma init, collection name setup ---
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key: raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            persona_name_safe = persona_name.lower().replace(' ', '_').replace('.', '')
            self.collection_name = f"life_summary_{persona_name_safe}"
            logger.info(f"Using ChromaDB collection name: {self.collection_name}")
            # --- End initial setup ---

            # --- Handle Collection Deletion if forcing re-index ---
            force_reindex = True
            if force_reindex:
                try:
                    logger.warning(f"Force re-index requested. Deleting collection '{self.collection_name}'.")
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted collection '{self.collection_name}'.")
                except Exception:
                     logger.info(f"Collection '{self.collection_name}' did not exist or couldn't be deleted (this is ok).")
            # --- End Deletion ---

            # --- Get or Create Collection ---
            self.memory_collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' ready.")
            logger.info(f"DEBUG: Type of self.memory_collection = {type(self.memory_collection)}") # Keep this for now
            # --- End Get/Create ---


            # --- Indexing Check (Using only .get()) ---
            collection_is_empty = True # Assume empty by default
            if self.memory_collection is not None:
                try:
                    # Try to get just the IDs of one item. If it returns IDs, it's not empty.
                    results = self.memory_collection.get(limit=1, include=[]) # include=[] is efficient
                    # Check if 'ids' key exists and if the list is not empty
                    if results and results.get('ids'):
                        collection_is_empty = False
                        logger.info(f"Collection '{self.collection_name}' check via .get() indicates it is NOT empty.")
                    else:
                        logger.info(f"Collection '{self.collection_name}' check via .get() indicates it IS empty.")
                except Exception as get_e:
                         logger.error(f"Error checking collection content via .get(limit=1): {get_e}", exc_info=True)
                         # If the check fails, safer to assume we might need indexing,
                         # but log the error. We'll still rely on force_reindex flag mainly.
                         # Setting collection_is_empty = True might be too aggressive if .get() fails intermittently.
                         # Let's proceed assuming it *might* have content if .get() failed,
                         # unless force_reindex is true.
                         collection_is_empty = False # Assume not empty if check fails, rely on force_reindex
                         logger.warning("Could not determine collection emptiness via .get(), assuming potentially non-empty.")

            needs_indexing = force_reindex or collection_is_empty
            # --- End Indexing Check ---


            if needs_indexing:
                if not collection_is_empty: # This means force_reindex was true
                     logger.info(f"Forcing re-indexing for potentially non-empty collection '{self.collection_name}'.")
                else:
                     logger.info(f"Collection '{self.collection_name}' is empty. Indexing life summary...")

                self.console.print(f"[cyan]🧠 Indexing background memory for {persona_name}...[/cyan]")
                # If forcing re-index on a collection where .get() failed, we still need to clear it first
                if force_reindex and not collection_is_empty:
                     try:
                          ids_to_delete = self.memory_collection.get(include=[])['ids']
                          if ids_to_delete:
                              logger.warning(f"Clearing existing items from {self.collection_name} due to force_reindex.")
                              self.memory_collection.delete(ids=ids_to_delete)
                     except Exception as del_err:
                          logger.error(f"Failed to clear collection during force_reindex after .get() check failed: {del_err}")
                          # Proceeding with add anyway, might lead to duplicate IDs if delete failed

                self._index_life_summary(self.life_summary_path) # Call indexer
            else:
                # If we reach here, force_reindex is False and collection_is_empty is False
                logger.info(f"Collection '{self.collection_name}' is not empty and force_reindex is False. Skipping indexing.")
                self.console.print(f"[green]🧠 Background memory for {persona_name} already indexed.[/green]")

        except Exception as e:
            logger.error(f"Failed to setup vector database for {persona_name}: {e}", exc_info=True)
            self.console.print(f"[bold red]❌ Error setting up RAG vector store: {e}[/bold red]")
            self.memory_collection = None # Disable RAG

    # --- RAG Method: Index Life Summary (Ensure this is uncommented) ---
    def _index_life_summary(self, life_summary_path: Optional[str]):
         """(Now called from setup) Chunks, embeds, and indexes the text content."""
         logger.info("--- STARTING _index_life_summary ---")
         if not self.memory_collection:
              logger.error("Cannot index: memory collection is None.")
              self.console.print("[red]Error: Cannot index, collection object invalid.[/red]")
              return
         if not life_summary_path or not os.path.exists(life_summary_path):
               logger.warning(f"Life summary file not found: {life_summary_path}. Cannot index.")
               self.console.print(f"[yellow]⚠️ Life summary file '{life_summary_path}' not found.[/yellow]")
               return

         try:
              # --- Load Data ---
              logger.info("Loading life summary JSON...")
              with open(life_summary_path, 'r', encoding='utf-8') as f: life_summary_data = json.load(f)
              logger.info("JSON loaded.")

              # --- Extract Text (NEEDS CUSTOMIZATION) ---
              logger.info("Extracting text to index...")
              text_to_index = ""
              # Add your specific logic here based on your JSON structure (Examples provided)
              if isinstance(life_summary_data.get("life_summary"), str): text_to_index = life_summary_data["life_summary"]
              elif isinstance(life_summary_data.get("persona_details"), dict) and isinstance(life_summary_data["persona_details"].get("background"), str): text_to_index = life_summary_data["persona_details"]["background"]
              elif isinstance(life_summary_data.get("yearly_summaries"), dict):
                   all_summaries = []
                   try: years = sorted(life_summary_data["yearly_summaries"].keys(), key=int)
                   except ValueError: years = life_summary_data["yearly_summaries"].keys()
                   for year in years:
                       summary_list = life_summary_data["yearly_summaries"][year]
                       if isinstance(summary_list, list) and summary_list and isinstance(summary_list[0], str): all_summaries.append(f"--- Age {year} --- \n{summary_list[0].strip()}")
                   if all_summaries: text_to_index = "\n\n".join(all_summaries)

              if not text_to_index or not isinstance(text_to_index, str) or not text_to_index.strip():
                  logger.error(f"No suitable text found or extracted text empty in {life_summary_path}.")
                  self.console.print("[yellow]⚠️ No text extracted for indexing.[/yellow]")
                  return
              logger.info(f"Text extracted successfully (length: {len(text_to_index)}).")
              # --- End Text Extraction ---

              # --- Chunking ---
              logger.info("Chunking text...")
              text_splitter = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
              chunks = text_splitter.split_text(text_to_index)
              if not chunks: logger.warning("Text splitting resulted in zero chunks."); return
              logger.info(f"Split into {len(chunks)} chunks.")
              # --- End Chunking ---

              # --- Embedding ---
              logger.info(f"Generating embeddings for {len(chunks)} chunks via Gemini...")
              embeddings_response = genai.embed_content(model=self.embedding_model_name, content=chunks, task_type="RETRIEVAL_DOCUMENT")
              embeddings = embeddings_response['embedding']
              if len(embeddings) != len(chunks): raise RuntimeError(f"Embedding count mismatch: Got {len(embeddings)} for {len(chunks)}.")
              logger.info("Embeddings generated successfully.")
              # --- End Embedding ---

              # --- Indexing ---
              ids = [f"{self.collection_name}_chunk_{i}" for i in range(len(chunks))]
              logger.info(f"Adding/updating {len(chunks)} items in Chroma collection '{self.collection_name}'...")
              # Using add which might error if IDs exist and collection wasn't cleared - adjust if needed
              self.memory_collection.add(embeddings=embeddings, documents=chunks, ids=ids)
              logger.info(f"Successfully added/updated items in collection.")
              self.console.print(f"[green]✅ Successfully indexed background memory ({len(chunks)} chunks).[/green]")
              # --- End Indexing ---

         except Exception as e:
              logger.error(f"--- FAILED during _index_life_summary: {e} ---", exc_info=True)
              self.console.print(f"[bold red]❌ Error during indexing: {e}[/bold red]")
         finally:
              logger.info("--- FINISHED _index_life_summary ---")

    # --- Existing Methods (_reflect_on_situation, _analyze_emotions) ---
    async def _reflect_on_situation(self, observations: str, immediate_environment: Dict) -> str:
        """Reflect on the current situation based on observations using LLM (via utility)."""
        prompt = PromptManager.reflect_on_situation_prompt(
            observations=observations,
            immediate_environment=immediate_environment,
            persona_state=self.persona
        )
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service, prompt=prompt, response_model=DayResponse, # Use DayResponse for {reflect:...}
            operation_description="Simulacra Reflection", system_instruction="Reflect deeply on the situation provided."
        )
        if response_dict and "reflect" in response_dict:
            return response_dict['reflect']
        else:
            logger.warning(f"Failed to get valid reflection. Response: {response_dict}")
            return "I need a moment to process what's happening." # Fallback reflection

    async def _analyze_emotions(self, situation_reflection: str, current_emotional_state: str) -> Dict:
        """Analyze emotions based on reflection and current state using LLM."""
        prompt = PromptManager.analyze_emotions_prompt(situation_reflection, current_emotional_state)
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service, prompt=prompt, response_model=EmotionAnalysisResponse,
            operation_description="Simulacra Emotion Analysis", system_instruction="Analyze the emotional response to the situation."
        )
        if response_dict and "error" not in response_dict:
            return response_dict
        else:
            logger.warning(f"Failed to get valid emotion analysis. Response: {response_dict}")
            # Fallback emotion analysis
            return {
                "primary_emotion": "Neutral", "intensity": "Medium",
                "secondary_emotion": None, "emotional_update": current_emotional_state
            }

    # --- MODIFIED: _decide_action now accepts retrieved_context and persona_state ---
    async def _decide_action(self,
                             reflection: str,
                             emotional_analysis: Dict,
                             immediate_environment: Dict,
                             retrieved_context: str) -> Dict: # Parameter added
        """Decides the next action based on reflection, emotions, goals, environment, and retrieved background."""
        goals = self.persona.get("goals", []) # Extract goals from persona

        # --- Call Updated PromptManager method ---
        prompt = PromptManager.decide_action_prompt(
            reflection=reflection,
            emotional_analysis=emotional_analysis,
            goals=goals,
            immediate_environment=immediate_environment,
            persona_state=self.persona,         # <<< Pass self.persona
            retrieved_background=retrieved_context # <<< Pass retrieved context
        )
        # --- End Update ---

        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
                prompt=prompt,
            response_model=ActionDecisionResponse,
            operation_description="Simulacra Action Decision",
            system_instruction="Decide the next logical action based on the provided context, including relevant background."
        )
        if response_dict and "error" not in response_dict:
            return response_dict
        else:
            logger.warning(f"Failed to get valid action decision. Response: {response_dict}")
            # Fallback action
            return {
                "thought_process": "Internal processing error. Defaulting to observation.",
                "action": "Observe surroundings again.",
                "action_details": None
            }

    # --- Existing save_state (Keep as is) ---
    def save_state(self):
        """Saves the Simulacra's current runtime state (persona + history) to the delta state file."""
        if not self.state_path:
            logger.warning("No delta state path provided, cannot save Simulacra state.")
            return
        simulacra_runtime_state = {"persona": self.persona, "history": self.history}
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as file:
                json.dump(simulacra_runtime_state, file, indent=2, default=str)
            logger.info(f"Simulacra runtime state saved to delta file: {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving Simulacra delta state to {self.state_path}: {e}", exc_info=True)
            self.console.print(f"[red]Error saving Simulacra delta state to {self.state_path}: {e}[/red]")

    # --- MODIFIED: process_perception integrates RAG retrieval ---
    async def process_perception(self, world_update: Dict[str, Any]) -> Dict[str, Any]:
        """Process perceptions, retrieve background, decide action, update state."""
        world_state = world_update.get("world_state", {})
        immediate_environment = world_update.get("immediate_environment", {})
        observations = world_update.get("observations", [])
        consequences = world_update.get("consequences", [])

        # Log current state before processing
        self.console.print(Panel(self.get_simulacra_summary(), title="[bold cyan]Simulacra State BEFORE Perception[/bold cyan]", border_style="cyan", expand=True))

        # Add perception to history
        perception_entry = {
            "timestamp": world_state.get("current_time", "unknown"), "date": world_state.get("current_date", "unknown"),
            "type": "perception", "data": {"observations": observations, "consequences": consequences}
        }
        self.history.append(perception_entry)

        # Prepare observations string for reflection
        observations_str_list = [str(obs) for obs in observations]
        consequences_str_list = [str(cons) for cons in consequences]
        observations_str = f"Previous Consequences:\n{json.dumps(consequences_str_list, indent=2)}\n\nNew Observations:\n{json.dumps(observations_str_list, indent=2)}"
        self.console.print(f"\n[bold yellow]Processing Perception:[/bold yellow]", observations_str)

        # --- Cognitive Loop Step 1 & 2 (Existing) ---
        self.console.print("\n[bold green]1. Reflecting on situation...[/bold green]")
        reflection_text = await self._reflect_on_situation(observations_str, immediate_environment)
        self.console.print(f"[italic green] -> Reflection:[/italic green] {reflection_text}\n")

        self.console.print("[bold blue]2. Analyzing emotions...[/bold blue]")
        current_emotional_state = self.persona.get("current_state", {}).get("emotional", "Neutral")
        emotional_analysis = await self._analyze_emotions(reflection_text, current_emotional_state)
        self.console.print(f"[italic blue] -> Emotional analysis:[/italic blue] {json.dumps(emotional_analysis, indent=2)}\n")

        # Update persona's emotional state
        if "emotional_update" in emotional_analysis:
             previous_emotional = self.persona.get("current_state",{}).get("emotional","?")
             if "current_state" not in self.persona: self.persona["current_state"] = {}
             self.persona["current_state"]["emotional"] = emotional_analysis["emotional_update"]
             logger.info(f"Emotional state updated: {previous_emotional} -> {self.persona['current_state']['emotional']}")

        # --- Cognitive Loop Step 2.5: RAG Retrieval ---
        self.console.print("[bold yellow]2.5 Retrieving relevant background...[/bold yellow]")
        retrieved_context = "[Background retrieval skipped or failed]"
        if self.memory_collection:
            try:
                # Formulate Query
                basic_persona_info = f"""
                Name: {self.persona.get('name', 'Unknown')}
                Age: {self.persona.get('age', 'Unknown')}
                Occupation: {self.persona.get('occupation', 'Unknown')}
                Location: {self.persona.get('current_location', 'Unknown')}
                Personality: {', '.join(self.persona.get('personality_traits', ['Unknown']))}
                """

                # 2. Relationship Info (Summarized)
                relationships = self.persona.get('initial_relationships', {})
                relationship_summary_parts = []
                for rel_type, rel_list in relationships.items():
                    if isinstance(rel_list, list):
                        names = [f"{p.get('name', 'Unknown')} ({p.get('relationship', 'Unknown')})" for p in rel_list[:2]] # Limit display
                        if names:
                            relationship_summary_parts.append(f"{rel_type.capitalize()}: {', '.join(names)}" + ('...' if len(rel_list) > 2 else ''))
                relationship_summary = "\n".join(relationship_summary_parts) if relationship_summary_parts else "No relationship details available."


                # 3. Recent Short-Term Memory (Last 3 entries)
                short_term_memories = self.persona.get('memory', {}).get('short_term', [])
                recent_memory_limit = 3
                recent_memories_str = ""
                if short_term_memories:
                    recent_entries = [f"- {mem}" for mem in short_term_memories[-recent_memory_limit:]]
                    recent_memories_str = "\n".join(recent_entries)
                else:
                    recent_memories_str = "No recent memories recorded."

                # 4. Current Situation (Reflection, Emotion, Observations)
                current_situation = f"""
                Reflection: {reflection_text}
                Feeling: {self.persona['current_state']['emotional']}
                Observations: {observations_str}
                """

                # 5. Combine all parts into the final query text
                query_text = f"""
                --- Persona ---
                {basic_persona_info}

                --- Relationships ---
                {relationship_summary}

                --- Recent Events ---
                {recent_memories_str}

                --- Current Situation ---
                {current_situation}
                """

                # Limit query size reasonably
                query_text = query_text[:1500] # Adjust limit if needed

                logger.debug(f"RAG Query (Truncated): {query_text[:300]}...") # Log more for debugging

                # Embed Query (Code remains the same)
                query_embedding_response = genai.embed_content(model=self.embedding_model_name, content=query_text, task_type="RETRIEVAL_QUERY")
                query_embedding = query_embedding_response['embedding']

                # Query Chroma (Code remains the same - consider increasing RAG_QUERY_RESULTS)
                results = self.memory_collection.query(query_embeddings=[query_embedding], n_results=RAG_QUERY_RESULTS, include=['documents']) # Consider adding 'distances'
                retrieved_docs = results.get('documents', [[]])[0]

                # Process results (Code remains the same)
                if retrieved_docs:
                     retrieved_context = "\n\n".join(retrieved_docs)
                     logger.info(f"Retrieved {len(retrieved_docs)} relevant background chunks.")
                     self.console.print(f"[italic yellow] -> Retrieved Background Context ({len(retrieved_docs)} chunks).[/italic yellow]\n")
                     logger.debug(f"Retrieved Context (Truncated): {retrieved_context[:200]}...")
                else:
                     retrieved_context = "[No specific background seemed relevant]"
                     logger.info("No relevant background chunks found.")
                     self.console.print(f"[italic yellow] -> No specific background context found.[/italic yellow]\n")

            except Exception as e:
                logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
                self.console.print(f"[yellow]⚠️ Warning: Failed to retrieve background context: {e}[/yellow]\n")
                retrieved_context = "[Error retrieving background context]"
        else:
            logger.warning("Skipping RAG retrieval as memory collection is not available.")
            self.console.print("[yellow]⚠️ RAG memory system inactive. Skipping background retrieval.[/yellow]\n")
            retrieved_context = "[Background retrieval system inactive]"
        # --- End RAG Retrieval ---

        # --- Cognitive Loop Step 3 (Modified Call) ---
        self.console.print("[bold magenta]3. Deciding on action (with background)...[/bold magenta]")
        action_decision = await self._decide_action(
            reflection=reflection_text,
            emotional_analysis=emotional_analysis,
            immediate_environment=immediate_environment,
            retrieved_context=retrieved_context # <<< Pass retrieved context
        )
        self.console.print(f"[italic magenta] -> Action decision:[/italic magenta] {json.dumps(action_decision, indent=2)}\n")
        # --- End Cognitive Loop ---

        # Prepare response for WorldEngine
        simulacra_response = {
            "thought_process": action_decision.get("thought_process", "..."),
            "action": action_decision.get("action", "Observe surroundings."),
            "action_details": action_decision.get("action_details"),
            "updated_state": self.persona.get("current_state", {}).copy()
        }

        # Update Short-Term Memory
        try:
            if 'memory' not in self.persona: self.persona['memory'] = {}
            if 'short_term' not in self.persona['memory']: self.persona['memory']['short_term'] = []
            new_memory_entry = f"Action taken: {simulacra_response['action']}" + (f" (Details: {simulacra_response['action_details']})" if simulacra_response.get('action_details') else "")
            self.persona['memory']['short_term'].append(new_memory_entry)
            max_short_term_memories = 15 # Keep this configurable maybe?
            self.persona['memory']['short_term'] = self.persona['memory']['short_term'][-max_short_term_memories:]
            logger.info(f"Added to short-term memory: '{new_memory_entry}'")
        except Exception as mem_err:
            logger.error(f"Failed to update short-term memory: {mem_err}", exc_info=True)

        # Add decided action to history
        action_entry = {
            "timestamp": world_state.get("current_time", "unknown"), "date": world_state.get("current_date", "unknown"),
            "type": "action", "data": simulacra_response
        }
        self.history.append(action_entry)

        # Log final state
        summary_text = self.get_simulacra_summary()
        self.console.print(Panel(summary_text, title="[bold magenta]Simulacra State AFTER Decision[/bold magenta]", border_style="magenta", expand=True))

        # Save state
        self.save_state() # Saves to delta file

        return simulacra_response

    # --- Existing get_simulacra_summary (Keep as is) ---
    def get_simulacra_summary(self) -> str:
        """Generate a concise summary of the simulacra's current state."""
        name = self.persona.get("name", "Unknown")
        age = self.persona.get("age", "?")
        occupation = self.persona.get("occupation", "?")
        physical = self.persona.get("current_state", {}).get("physical", "?")
        emotional = self.persona.get("current_state", {}).get("emotional", "?")
        mental = self.persona.get("current_state", {}).get("mental", "?")

        goals = self.persona.get("goals", [])
        goals_str = ", ".join(goals[:2]) + ('...' if len(goals) > 2 else '') if goals else "[italic]None[/italic]"

        short_term = self.persona.get("memory", {}).get("short_term", [])
        memory_limit = 5 # Show last 5 memories

        recent_memories_list = []
        if short_term:
            # Iterate in reverse to get the latest first, then reverse again for display order
            for mem in reversed(short_term[-memory_limit:]):
                mem_str = str(mem)
                recent_memories_list.append(f"'{mem_str}'")
            recent_memories = "\n  ".join(reversed(recent_memories_list)) # Display oldest of the last 5 first
        else:
            recent_memories = "[italic]None[/italic]"

        summary = f"[b]{name}[/b] ({age}, {occupation})\n"
        summary += f" State: [green]{physical}[/green] | [yellow]{emotional}[/yellow] | [magenta]{mental}[/magenta]\n"
        summary += f" Goals: [cyan]{goals_str}[/cyan]\n"
        summary += f" Recent Memory (last {memory_limit}):\n  [orange3]{recent_memories}[/orange3]" # Clarified memory display
        return summary
