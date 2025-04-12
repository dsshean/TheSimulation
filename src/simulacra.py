# src/simulacra.py
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import re
from rich.console import Console
from rich.panel import Panel

# --- Existing Imports ---
from src.llm_service import LLMService
from src.prompt_manager import PromptManager
from src.utils.llm_utils import generate_and_validate_llm_response
from src.models import EmotionAnalysisResponse, ActionDecisionResponse, DayResponse, ActionDetails # Added ActionDetails
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
# --- End Default Persona ---\

# --- RAG Constants ---
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"
CHROMA_DB_PATH = os.path.join("db", "chroma_simulacra")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
RAG_QUERY_RESULTS = 3
# --- End RAG Constants ---\


class Simulacra:
    """Represents a simulated human with personality, goals, and behaviors."""

    # <<< Keep __init__, _setup_vector_db, _index_life_summary as is >>>
    def __init__(self,
                 life_summary_path: Optional[str], # Path for INITIAL persona
                 delta_state_path: str = "simulacra_deltas.json", # Path for RUNTIME state
                 console: Optional[Console] = None,
                 new_simulacra: bool = False):
        """Initialize the Simulacra."""
        self.console = console or Console()
        self.state_path = delta_state_path
        self.life_summary_path = life_summary_path
        self.persona: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.llm_service = LLMService()

        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.memory_collection: Optional[chromadb.Collection] = None
        self.embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
        self.collection_name: Optional[str] = None

        loaded_from_delta = False
        if new_simulacra:
            logger.info("`new_simulacra` flag set. Will initialize from life summary and force re-indexing.")
            self.console.print("[yellow]--new flag set: Initializing from life summary, ignoring delta file.[/yellow]")
        else:
            if os.path.exists(self.state_path):
                 logger.info(f"Attempting to load runtime state from: {self.state_path}")
                 try:
                     with open(self.state_path, 'r', encoding='utf-8') as file: delta_data = json.load(file)
                     if isinstance(delta_data.get("persona"), dict) and isinstance(delta_data.get("history"), list):
                         self.persona = delta_data["persona"]
                         self.history = delta_data["history"]
                         required_keys = ["name", "age", "occupation", "personality_traits", "current_state", "memory"]
                         if not all(key in self.persona for key in required_keys):
                            logger.warning(f"Loaded persona from delta file '{self.state_path}' missing core keys. Falling back.")
                            self.persona = {}; self.history = []
                         else:
                             logger.info(f"Successfully loaded runtime state for '{self.persona.get('name', 'Unknown')}' from {self.state_path}")
                             self.console.print(f"[green]Loaded runtime state for {self.persona.get('name', 'Unknown')} from {self.state_path}[/green]")
                             loaded_from_delta = True
                     else: logger.warning(f"Invalid structure in delta file '{self.state_path}'. Falling back.")
                 except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                     logger.error(f"Failed to load or parse delta state file '{self.state_path}': {e}. Falling back.", exc_info=True)
                     self.console.print(f"[red]Error loading delta state file '{self.state_path}'. Trying life summary.[/red]")
            else:
                 logger.info(f"Delta state file '{self.state_path}' not found. Trying life summary.")
                 self.console.print(f"[yellow]Delta state file '{self.state_path}' not found. Trying life summary.[/yellow]")

        if not loaded_from_delta:
            self.history = []
            if not self.life_summary_path or not os.path.exists(self.life_summary_path):
                logger.warning(f"Life summary file not found: '{self.life_summary_path}'. Using default persona.")
                self.console.print(f"[yellow]Life summary file '{self.life_summary_path}' not found. Using default persona.[/yellow]")
                self.persona = DEFAULT_PERSONA.copy()
            else:
                 logger.info(f"Loading initial persona from life summary: {self.life_summary_path}")
                 try:
                     with open(self.life_summary_path, 'r', encoding='utf-8') as file: life_data = json.load(file)
                     persona_details = life_data.get("persona_details")
                     if isinstance(persona_details, dict):
                         # Convert keys to lowercase for consistency if needed
                         self.persona = {key.lower().replace('_', ''): value for key, value in persona_details.items()}
                         # Ensure core structures exist
                         self.persona.setdefault("current_state", {"physical": "Normal", "emotional": "Neutral", "mental": "Aware"})
                         self.persona.setdefault("memory", {"short_term": [], "long_term": []})
                         self.persona.setdefault("goals", []) # Ensure goals list exists

                         # Add context from latest year summary to short-term memory
                         birth_year = life_data.get("birth_year")
                         age = self.persona.get("age")
                         if isinstance(birth_year, int) and isinstance(age, int):
                             latest_year_key = str(birth_year + age)
                             yearly_summaries = life_data.get("yearly_summaries", {})
                             if isinstance(yearly_summaries, dict):
                                latest_year_summary_data = yearly_summaries.get(latest_year_key)
                                if isinstance(latest_year_summary_data, list) and latest_year_summary_data:
                                     # Assume the first item in the list is the summary text
                                     full_summary_text = str(latest_year_summary_data[0])
                                     self.persona["memory"]["short_term"].append(f"Context from age {age}: {full_summary_text}")
                                elif isinstance(latest_year_summary_data, str): # Handle if it's just a string
                                     self.persona["memory"]["short_term"].append(f"Context from age {age}: {latest_year_summary_data}")

                         required_keys = ["name", "age", "occupation", "personalitytraits", "currentstate", "memory", "goals"] # Use lowercase keys now
                         if not all(key in self.persona for key in required_keys):
                             logger.warning(f"Loaded persona details from '{self.life_summary_path}' missing core keys after processing. Reverting to default.")
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

        if not self.persona:
             logger.critical("Failed to load persona from any source. Using default.")
             self.console.print("[bold red]CRITICAL: Failed to load persona from any source. Using default.[/bold red]")
             self.persona = DEFAULT_PERSONA.copy(); self.history = []

        # --- Setup Vector DB ---
        if self.persona and self.persona.get('name'): self._setup_vector_db(new_simulacra)
        else: logger.error("Could not setup vector DB because persona name is missing after initialization."); self.console.print("[bold red]Error: Could not initialize RAG memory system - persona invalid after load.[/bold red]")

    def _setup_vector_db(self, force_reindex: bool):
        """Initializes ChromaDB client, collection, and indexes the life summary if needed or forced."""
        persona_name = self.persona.get('name', 'unknown_persona')
        logger.info(f"Setting up vector DB for persona: {persona_name} (Force Re-index: {force_reindex})")
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key: raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            persona_name_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', persona_name.lower()) # Sanitize further
            persona_name_safe = re.sub(r'_+', '_', persona_name_safe).strip('_') # Remove multiple/leading/trailing underscores
            if not persona_name_safe: persona_name_safe = "default_persona" # Fallback if name becomes empty
            self.collection_name = f"life_summary_{persona_name_safe}"
            # Validate collection name length and format
            if not (3 <= len(self.collection_name) <= 63): raise ValueError(f"Invalid collection name length: {self.collection_name}")
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$", self.collection_name):
                 if not re.match(r"^[a-zA-Z0-9]{1,2}$", self.collection_name): # Allow short names too
                     raise ValueError(f"Invalid collection name format: {self.collection_name}")
            if ".." in self.collection_name: raise ValueError("Collection name cannot contain ..")

            logger.info(f"Using ChromaDB collection name: {self.collection_name}")

            collection_exists = False
            try:
                 # Try getting the collection to check existence efficiently
                 self.memory_collection = self.chroma_client.get_collection(name=self.collection_name)
                 collection_exists = True
                 logger.info(f"Collection '{self.collection_name}' already exists.")
            except ValueError: # Chroma raises ValueError if collection doesn't exist
                 logger.info(f"Collection '{self.collection_name}' does not exist. Will create.")
                 collection_exists = False
            except Exception as e:
                 logger.error(f"Unexpected error checking for collection '{self.collection_name}': {e}", exc_info=True)
                 raise # Re-raise unexpected errors

            # Delete if forcing re-index and it exists
            if force_reindex and collection_exists:
                try:
                    logger.warning(f"Force re-index requested. Deleting collection '{self.collection_name}'.")
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted collection '{self.collection_name}'.")
                    collection_exists = False # Mark as non-existent now
                except Exception as e:
                    logger.error(f"Failed to delete collection '{self.collection_name}' for re-indexing: {e}", exc_info=True)
                    # Decide if we should proceed without re-indexing or raise error
                    # raise RuntimeError(f"Failed to delete collection for re-index: {e}") from e

            # Create collection if it doesn't exist (or was just deleted)
            if not collection_exists:
                self.memory_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"} # Specify cosine distance
                )
                logger.info(f"Created new collection '{self.collection_name}'.")

            # Check if collection is empty before indexing
            needs_indexing = False
            if self.memory_collection:
                 try:
                     results = self.memory_collection.get(limit=1) # Efficient check for emptiness
                     if not results or not results.get('ids'):
                          logger.info(f"Collection '{self.collection_name}' check via .get() indicates it IS empty.")
                          needs_indexing = True
                     else: logger.info(f"Collection '{self.collection_name}' check via .get() indicates it is NOT empty.")
                 except Exception as e:
                     logger.error(f"Error checking collection count/emptiness for '{self.collection_name}': {e}", exc_info=True)
                     needs_indexing = True # Assume indexing needed if check fails

            # Index if needed
            if needs_indexing:
                 if self.life_summary_path and os.path.exists(self.life_summary_path):
                      logger.info(f"Collection '{self.collection_name}' is empty. Indexing life summary...")
                      self._index_life_summary(self.life_summary_path)
                 else:
                      logger.warning(f"Collection '{self.collection_name}' is empty, but no life summary path provided/found ('{self.life_summary_path}') to index.")
                      self.console.print(f"[yellow]Warning: RAG Memory for {persona_name} is empty and cannot be populated without a life summary.[/yellow]")

            logger.info(f"Collection '{self.collection_name}' ready.")
            logger.info(f"DEBUG: Type of self.memory_collection = {type(self.memory_collection)}")


        except ValueError as ve: # Catch ChromaDB/validation specific errors
            logger.critical(f"Failed to setup vector DB for '{persona_name}': {ve}", exc_info=True)
            self.console.print(f"[bold red]CRITICAL ERROR setting up RAG memory: {ve}[/bold red]")
            self.memory_collection = None # Ensure collection is None on error
        except Exception as e:
            logger.critical(f"Unexpected error setting up vector DB for '{persona_name}': {e}", exc_info=True)
            self.console.print(f"[bold red]CRITICAL UNEXPECTED ERROR setting up RAG memory: {e}[/bold red]")
            self.memory_collection = None # Ensure collection is None on error

    def _index_life_summary(self, life_summary_path: Optional[str]):
        """Chunks and indexes the text content of the life summary JSON."""
        if not self.memory_collection:
            logger.error("Cannot index life summary: Memory collection not initialized.")
            return
        if not life_summary_path or not os.path.exists(life_summary_path):
            logger.error(f"Cannot index life summary: File path invalid or file not found ('{life_summary_path}').")
            return

        logger.info("--- STARTING _index_life_summary ---")
        try:
            # 1. Load JSON
            logger.info("Loading life summary JSON...")
            with open(life_summary_path, 'r', encoding='utf-8') as f:
                life_data = json.load(f)
            logger.info("JSON loaded.")

            # 2. Extract Text (Combine relevant fields - adapt as needed)
            logger.info("Extracting text to index...")
            texts_to_index = []
            if isinstance(life_data.get("persona_details"), dict):
                # Add structured persona details as text
                pd = life_data["persona_details"]
                texts_to_index.append(f"Name: {pd.get('Name', '?')}. Age: {pd.get('Age', '?')}. Occupation: {pd.get('Occupation', '?')}.")
                texts_to_index.append(f"Born in: {pd.get('Birthplace', '?')}. Education: {pd.get('Education', 'N/A')}.")
                texts_to_index.append(f"Personality Traits: {', '.join(pd.get('Personality_Traits', []))}.")

            # Add yearly summaries (assuming they are structured correctly)
            yearly_summaries = life_data.get("yearly_summaries", {})
            if isinstance(yearly_summaries, dict):
                 for year, summary_data in sorted(yearly_summaries.items()):
                      # Assuming summary_data is a list where the first item is the text summary
                      if isinstance(summary_data, list) and summary_data:
                           summary_text = str(summary_data[0]) # Ensure string
                           texts_to_index.append(f"Year {year}: {summary_text}")
                      elif isinstance(summary_data, str): # Handle if it's just a string
                           texts_to_index.append(f"Year {year}: {summary_data}")


            full_text = "\n\n".join(texts_to_index)
            if not full_text.strip():
                 logger.warning("No text extracted from life summary for indexing.")
                 print("--- FINISHED _index_life_summary (No text) ---")
                 return
            logger.info(f"Text extracted successfully (length: {len(full_text)}).")


            # 3. Chunk Text
            logger.info("Chunking text...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_text(full_text)
            logger.info(f"Split into {len(chunks)} chunks.")
            if not chunks:
                 logger.warning("Text splitting resulted in zero chunks.")
                 print("--- FINISHED _index_life_summary (No chunks) ---")
                 return

            # 4. Generate Embeddings (Batch if possible, handle errors)
            logger.info(f"Generating embeddings for {len(chunks)} chunks via Gemini...")
            embeddings = []
            try:
                # Batch embedding generation
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=chunks,
                    task_type="retrieval_document" # Important for document retrieval tasks
                )
                embeddings = result['embedding']
                if len(embeddings) != len(chunks):
                     raise ValueError(f"Mismatch between number of chunks ({len(chunks)}) and embeddings received ({len(embeddings)}).")
                logger.info("Embeddings generated successfully.")
            except Exception as emb_err:
                 logger.error(f"Error generating embeddings: {emb_err}", exc_info=True)
                 # Optional: Fallback to individual embedding generation or abort
                 raise RuntimeError(f"Embedding generation failed: {emb_err}") from emb_err


            # 5. Add/Update in Chroma
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": os.path.basename(life_summary_path)} for _ in chunks] # Basic metadata

            logger.info(f"Adding/updating {len(chunks)} items in Chroma collection '{self.collection_name}'...")
            try:
                self.memory_collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas
                )
                logger.info("Successfully added/updated items in collection.")
            except Exception as upsert_err:
                logger.error(f"Error upserting data into Chroma: {upsert_err}", exc_info=True)
                raise RuntimeError(f"ChromaDB upsert failed: {upsert_err}") from upsert_err

            logger.info("--- FINISHED _index_life_summary ---")

        except Exception as e:
            logger.error(f"Error during life summary indexing: {e}", exc_info=True)
            # Optionally re-raise or handle specific exceptions differently
            self.console.print(f"[bold red]Error indexing life summary: {e}[/bold red]")


    # --- RAG Method: Query Memories ---
    def _query_memories(self, query_text: str) -> str:
        """Queries the vector DB for relevant memories/context."""
        start_time = time.time()
        if not self.memory_collection:
            logger.warning("Cannot query memories: Memory collection not available.")
            return "[Memory retrieval failed: Collection not initialized]"

        try:
            # 1. Generate Query Embedding
            query_embedding = genai.embed_content(
                model=self.embedding_model_name,
                content=query_text,
                task_type="retrieval_query" # Use appropriate task type for querying
            )['embedding']

            # 2. Query Chroma
            results = self.memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=RAG_QUERY_RESULTS,
                include=['documents'] # Only need the document text
            )

            # 3. Format Results
            retrieved_docs = results.get('documents', [[]])[0] # Get the list of documents for the first query
            if retrieved_docs:
                 context = "\n---\n".join(retrieved_docs)
                 logger.info(f"Retrieved {len(retrieved_docs)} relevant background chunks.")
                 end_time = time.time()
                 logger.debug(f"RAG Query Time: {end_time - start_time:.4f} seconds")
                 return context
            else:
                 logger.info("No relevant background chunks found for query.")
                 return "[No relevant background context found]"

        except Exception as e:
            logger.error(f"Error querying memories: {e}", exc_info=True)
            return f"[Memory retrieval error: {e}]"


    # --- Core Agent Loop Methods ---

    async def _reflect_on_situation(self, observations: str, immediate_environment: Dict) -> str:
        """Generates internal reflection based on observations and environment."""
        # Use generate_and_validate_llm_response for consistency
        prompt = PromptManager.reflect_on_situation_prompt(observations, immediate_environment, self.persona)
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=DayResponse, # Expecting {'reflect': '...'}
            system_instruction="Reflect deeply on the situation. Respond ONLY with the DayResponse JSON.",
            operation_description="Simulacra Reflection"
        )
        reflection = "Could not reflect properly."
        if response_dict and "error" not in response_dict and "reflect" in response_dict:
            reflection = response_dict["reflect"]
        elif response_dict and "error" in response_dict:
             reflection = f"[Reflection Error: {response_dict['error']}]"
        logger.debug(f"Reflection generated: {reflection[:100]}...")
        return reflection

    async def _analyze_emotions(self, situation_reflection: str, current_emotional_state: str) -> Dict:
        """Analyzes emotional response to the situation and reflection."""
        prompt = PromptManager.analyze_emotions_prompt(situation_reflection, current_emotional_state)
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=EmotionAnalysisResponse,
            system_instruction="Analyze emotions based on situation. Respond ONLY with EmotionAnalysisResponse JSON.",
            operation_description="Simulacra Emotion Analysis"
        )
        if response_dict and "error" not in response_dict:
            logger.info(f"Emotional state updated: {current_emotional_state} -> {response_dict.get('emotional_update', '???')}")
            return response_dict # Return the validated dictionary
        else:
            logger.error(f"Emotion analysis failed: {response_dict}")
            # Return a placeholder or previous state if error
            return {"primary_emotion": "Neutral", "intensity": "Medium", "secondary_emotion": None, "emotional_update": current_emotional_state + " (Analysis Failed)"}

    async def _decide_action(self,
                             reflection: str,
                             emotional_analysis: Dict,
                             immediate_environment: Dict,
                             retrieved_context: str,
                             step_duration_minutes: int,
                             last_action_taken: Optional[str],
                             world_state: Dict) -> Dict: # <<< ADDED world_state parameter >>>
        """Decides the next action based on internal state, environment, context, last action, and world state."""
        goals = self.persona.get("goals", [])
        prompt = PromptManager.decide_action_prompt(
            reflection=reflection,
            emotional_analysis=emotional_analysis,
            goals=goals,
            immediate_environment=immediate_environment,
            persona_state=self.persona,
            retrieved_background=retrieved_context,
            step_duration_minutes=step_duration_minutes,
            last_action_taken=last_action_taken,
            world_state=world_state # <<< PASS world_state here >>>
        )

        # ... rest of the method (LLM call, validation, logging) remains the same ...
        # Validate against ActionDecisionResponse
        response_dict = await generate_and_validate_llm_response(
            llm_service=self.llm_service,
            prompt=prompt,
            response_model=ActionDecisionResponse,
            system_instruction=f"Decide the next plausible action for the {step_duration_minutes} min step, considering the last action and REALITY CONSTRAINTS. Respond ONLY with ActionDecisionResponse JSON.", # <<< Updated instruction
            operation_description="Simulacra Action Decision"
        )

        if response_dict and "error" not in response_dict:
            # ... (validation/handling of action_details remains the same) ...
            if 'action_details' in response_dict and not isinstance(response_dict['action_details'], (dict, type(None))):
                 logger.warning(f"LLM returned invalid type for action_details ({type(response_dict['action_details'])}). Setting to None.")
                 response_dict['action_details'] = None
            # Note: Pydantic validation happens within generate_and_validate_llm_response now

            logger.info(f"Action decided: {response_dict.get('action', '???')} Details: {response_dict.get('action_details')}")
            return response_dict
        else:
            logger.error(f"Action decision failed: {response_dict}")
            return {"thought_process": "Failed to decide action, waiting.", "action": "wait", "action_details": None}

    def save_state(self):
        """Saves the current persona and history to the delta state file."""
        state_data = {
            "persona": self.persona,
            "history": self.history
        }
        try:
            with open(self.state_path, 'w', encoding='utf-8') as file:
                json.dump(state_data, file, indent=2, default=str) # Use default=str for non-serializable
            logger.info(f"Simulacra runtime state saved to delta file: {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving Simulacra state to {self.state_path}: {e}", exc_info=True)
            self.console.print(f"[red]Error saving Simulacra state: {e}[/red]")


    # --- Main Method Called by Simulation Loop ---

    async def process_perception(self, world_update: Dict[str, Any]) -> None:
        """
        Processes incoming world state and observations, updating internal state.
        (This method primarily updates memory/state based on world feedback).
        """
        if not isinstance(self.persona, dict): self.persona = {} # Ensure dict
        if 'current_state' not in self.persona: self.persona['current_state'] = {}
        if 'memory' not in self.persona: self.persona['memory'] = {'short_term': [], 'long_term': []}
        if 'short_term' not in self.persona['memory']: self.persona['memory']['short_term'] = []

        # --- Extract data from world_update ---
        consequences = world_update.get("consequences", [])
        observations = world_update.get("observations", []) # Observations should include dialogue dicts
        narrative_update = world_update.get("narrative_update")
        # world_state = world_update.get("world_state", {}) # Not directly used here?
        # immediate_environment = world_update.get("immediate_environment", {}) # Not directly used here?

        # --- Log and Update History ---
        timestamp = f"{world_update.get('world_state', {}).get('current_date','?')}_{world_update.get('world_state', {}).get('current_time','?')}"
        perception_log = {"timestamp": timestamp, "type": "perception", "data": {"consequences": consequences, "observations": observations, "narrative": narrative_update}}
        self.history.append(perception_log)

        # --- Update Short-Term Memory ---
        # Add a concise summary of the perception to short-term memory
        memory_entry_parts = []
        if narrative_update and narrative_update.strip() and "[Narrative" not in narrative_update:
             memory_entry_parts.append(narrative_update) # Add narrative first if available
        else: # Fallback if no narrative
             if consequences: memory_entry_parts.append(f"Consequences: {'; '.join(consequences)}")
             obs_strs = []
             for obs in observations:
                 if isinstance(obs, dict) and obs.get('type') == 'dialogue':
                     obs_strs.append(f"Dialogue: {obs.get('from','?')} said '{obs.get('utterance','...')}' to {obs.get('to','?')}.")
                 elif isinstance(obs, str):
                     obs_strs.append(obs)
             if obs_strs: memory_entry_parts.append(f"Observed: {'; '.join(obs_strs)}")

        if memory_entry_parts:
             memory_entry = " ".join(memory_entry_parts)
             # Limit memory entry length if needed
             MAX_STM_ENTRY_LEN = 500
             if len(memory_entry) > MAX_STM_ENTRY_LEN:
                 memory_entry = memory_entry[:MAX_STM_ENTRY_LEN] + "..."

             self.persona["memory"]["short_term"].append(memory_entry)
             # Keep short-term memory concise (e.g., last 10 entries)
             MAX_STM_LEN = 10
             self.persona["memory"]["short_term"] = self.persona["memory"]["short_term"][-MAX_STM_LEN:]
             logger.debug(f"Added to short-term memory: '{memory_entry[:80]}...'")

        # --- Potentially update other internal states based on perception ---
        # Example: Inferring physical state from consequences/observations
        # This is complex and might require another LLM call or rule-based system.
        # For now, we rely on the reflection/emotion analysis steps before action.

        # --- Save the updated state ---
        self.save_state() # Save after processing perception

    async def decide_next_action(self,
                                 world_update: Dict[str, Any],
                                 step_duration_minutes: int,
                                 last_action_taken: Optional[str]) -> Dict[str, Any]:
        """
        Simulacra's main turn-taking method.
        Reflects, analyzes emotions, queries RAG, decides action. Accepts last action context.
        """
        # ... (persona structure checks remain the same) ...
        if not isinstance(self.persona, dict): self.persona = DEFAULT_PERSONA.copy()
        if 'current_state' not in self.persona: self.persona['current_state'] = {}
        if 'memory' not in self.persona: self.persona['memory'] = {'short_term': [], 'long_term': []}

        # 1. Extract relevant info from world update (remains same)
        world_state = world_update.get("world_state", {}) # <<< Get world_state
        immediate_environment = world_update.get("immediate_environment", {})
        perception_summary = "; ".join(world_update.get("consequences", [])) + "; " + "; ".join(map(str, world_update.get("observations", [])))
        narrative = world_update.get("narrative_update", "")
        situation_context = narrative if narrative and "[Narrative" not in narrative else perception_summary

        # 2. Reflect on the current situation (remains same)
        reflection = await self._reflect_on_situation(situation_context, immediate_environment)
        # ... log reflection ...

        # 3. Analyze Emotions (remains same)
        current_emotional_state = self.persona.get("current_state", {}).get("emotional", "Neutral")
        emotion_analysis = await self._analyze_emotions(reflection, current_emotional_state)
        # ... log emotion_analysis ...
        self.persona["current_state"]["emotional"] = emotion_analysis.get("emotional_update", current_emotional_state)

        # 4. Query RAG based on reflection and current state (remains same)
        query_text = f"Reflection: {reflection}\nFeeling: {self.persona['current_state']['emotional']}\nGoals: {self.persona.get('goals',[])}\nLocation: {immediate_environment.get('current_location_name', '?')}"
        retrieved_context = self._query_memories(query_text)
        # ... log retrieved_context ...

        # 5. Decide Action (passing RAG context, duration, last action, and world_state)
        # <<< MODIFIED: Pass world_state down to _decide_action >>>
        action_decision = await self._decide_action(
            reflection=reflection,
            emotional_analysis=emotion_analysis,
            immediate_environment=immediate_environment,
            retrieved_context=retrieved_context,
            step_duration_minutes=step_duration_minutes,
            last_action_taken=last_action_taken,
            world_state=world_state # <<< Pass world_state here
        )
        thought_process = action_decision.get("thought_process", "[Thought process not generated]")
        # ... log thought_process ...

        # 6. Update History & State (remains same, but logs the decided action)
        timestamp = f"{world_state.get('current_date','?')}_{world_state.get('current_time','?')}"
        action_log = {"timestamp": timestamp, "type": "action_taken", "data": action_decision}
        self.history.append(action_log)
        # ... format action for memory entry ...
        mem_action_verb = action_decision.get('action', '?')
        mem_action_details = action_decision.get('action_details')
        mem_details_str = ""
        # ... formatting details_str ...
        short_term_memory_entry = f"Action planned: {mem_action_verb}{mem_details_str}" # Changed wording slightly

        self.persona["memory"]["short_term"].append(short_term_memory_entry)
        MAX_STM_LEN = 10
        self.persona["memory"]["short_term"] = self.persona["memory"]["short_term"][-MAX_STM_LEN:]
        logger.info(f"Added to short-term memory: '{short_term_memory_entry}'")

        self.save_state() # Save state after deciding action

        # 7. Format and return the action for the World Engine (remains same)
        return {
            "agent_name": self.persona.get("name", "Simulacra"),
            "action": action_decision.get("action", "wait"),
            "action_details": action_decision.get("action_details"),
            "reflection": reflection,
            "thought_process": thought_process
        }

    def get_simulacra_summary(self) -> str:
        """Generate a concise summary string for the Simulacra."""
        # <<< Keep the existing 'get_simulacra_summary' method logic from your full file >>>
        if not self.persona: return "[bold red]Simulacra NA.[/bold red]"
        p = self.persona; cs = p.get('current_state', {})
        name = f"[b]{p.get('name', '?')} ({p.get('age', '?')})[/b]"
        occupation = f"[i]{p.get('occupation', '?')}[/i]"
        emotion = f"[e]{cs.get('emotional', '?')}[/e]"
        phys = f"[p]{cs.get('physical', '?')}[/p]"; ment = f"[m]{cs.get('mental', '?')}[/m]"
        last_action = "None"
        action_history = [h for h in reversed(self.history) if h.get('type') == 'action_taken']
        if action_history:
            last_action_data = action_history[0].get('data', {})
            act_verb = last_action_data.get('action', '?')
            act_details = last_action_data.get('action_details')
            last_action = f"{act_verb}"
            if isinstance(act_details, dict): # Format details if dict
                details_str = ", ".join(f"{k}={v}" for k, v in act_details.items() if v is not None)
                if details_str: last_action += f"({details_str})"
            elif isinstance(act_details, str): last_action += f"({act_details})" # Fallback for old string format
        return f"{name} {occupation}. {emotion} {phys} {ment}. Last: {last_action}"
