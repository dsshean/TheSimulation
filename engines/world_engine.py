from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.vector_stores import QdrantVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from qdrant_client import QdrantClient
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

@dataclass
class WorldState:
    timestamp: datetime
    locations: Dict[str, Dict]  # Location name -> state
    events: List[Dict]  # Historical events
    current_conditions: Dict  # Weather, time of day, etc.
    npcs: Dict[str, Dict]  # NPC name -> state

@dataclass
class Character:
    name: str
    location: str
    stats: Dict[str, any]
    inventory: List[Dict]
    status: Dict[str, any]
    history: List[Dict]

class WorldEngine:
    def __init__(
        self,
        openai_api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        world_collection: str = "world_state",
        character_collection: str = "character_state",
        model_name: str = "gpt-4",
        persist_dir: str = "./world_data"
    ):
        """
        Initialize the World Engine with separate RAG systems for world and character state.
        """
        self.logger = logging.getLogger(__name__)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM and embedding model
        self.llm = OpenAI(temperature=0.7, model=model_name, api_key=openai_api_key)
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize vector stores for world and character
        self.world_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=world_collection
        )
        self.character_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=character_collection
        )
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        
        # Initialize current state
        self.current_world_state = None
        self.current_character = None
        
        # Load initial states if they exist
        self._load_current_states()

    def initialize_world(self, initial_world_state: WorldState):
        """Initialize or reset the world state."""
        self.current_world_state = initial_world_state
        self._save_world_state()
        self._update_world_rag()

    def initialize_character(self, character: Character):
        """Initialize or reset the character state."""
        self.current_character = character
        self._save_character_state()
        self._update_character_rag()

    def progress_time(self, hours: int = 1):
        """
        Progress the world state by specified number of hours.
        Updates both world and character states based on interactions.
        """
        if not self.current_world_state or not self.current_character:
            raise ValueError("World and character must be initialized first")

        # Generate world update prompt
        world_prompt = self._create_world_progression_prompt(hours)
        
        # Get world update from LLM
        world_update = self.llm.complete(world_prompt)
        world_changes = self._parse_world_update(world_update.text)
        
        # Generate character update prompt based on world changes
        character_prompt = self._create_character_progression_prompt(hours, world_changes)
        
        # Get character update from LLM
        character_update = self.llm.complete(character_prompt)
        character_changes = self._parse_character_update(character_update.text)
        
        # Apply updates
        self._apply_world_changes(world_changes)
        self._apply_character_changes(character_changes)
        
        # Update timestamp
        self.current_world_state.timestamp += timedelta(hours=hours)
        
        # Save and update RAG systems
        self._save_world_state()
        self._save_character_state()
        self._update_world_rag()
        self._update_character_rag()
        
        return {
            "world_changes": world_changes,
            "character_changes": character_changes,
            "new_timestamp": self.current_world_state.timestamp
        }

    def _create_world_progression_prompt(self, hours: int) -> str:
        """Create prompt for world state progression."""
        return f"""
        Progress the world state by {hours} hours. Consider:
        - Current timestamp: {self.current_world_state.timestamp}
        - Current locations: {json.dumps(self.current_world_state.locations, indent=2)}
        - Recent events: {json.dumps(self.current_world_state.events[-5:], indent=2)}
        - Current conditions: {json.dumps(self.current_world_state.current_conditions, indent=2)}
        - Active NPCs: {json.dumps(self.current_world_state.npcs, indent=2)}
        - Character's current location: {self.current_character.location}
        - Character's recent actions: {json.dumps(self.current_character.history[-3:], indent=2)}

        Describe the changes to the world state in the following format:
        1. Location changes
        2. New events
        3. Updated conditions
        4. NPC updates
        """

    def _create_character_progression_prompt(self, hours: int, world_changes: Dict) -> str:
        """Create prompt for character state progression."""
        return f"""
        Update the character state based on {hours} hours of world progression. Consider:
        - Character current state: {json.dumps(self.current_character.__dict__, indent=2)}
        - World changes: {json.dumps(world_changes, indent=2)}
        - Location conditions: {json.dumps(self.current_world_state.locations[self.current_character.location], indent=2)}

        Describe the changes to the character in the following format:
        1. Status updates
        2. Inventory changes
        3. New history entries
        4. Stat changes
        """

    def _parse_world_update(self, update_text: str) -> Dict:
        """Parse LLM response for world updates."""
        # In a real implementation, you'd want more robust parsing
        # This is a simplified example
        changes = {
            "location_changes": {},
            "new_events": [],
            "condition_updates": {},
            "npc_updates": {}
        }
        
        # Parse the LLM response and populate the changes dict
        # You'd want to implement proper parsing logic here
        
        return changes

    def _parse_character_update(self, update_text: str) -> Dict:
        """Parse LLM response for character updates."""
        # Similar to world update parsing
        changes = {
            "status_updates": {},
            "inventory_changes": [],
            "new_history": [],
            "stat_changes": {}
        }
        
        return changes

    def _apply_world_changes(self, changes: Dict):
        """Apply world state changes."""
        # Update locations
        for loc, updates in changes["location_changes"].items():
            self.current_world_state.locations[loc].update(updates)
        
        # Add new events
        self.current_world_state.events.extend(changes["new_events"])
        
        # Update conditions
        self.current_world_state.current_conditions.update(changes["condition_updates"])
        
        # Update NPCs
        for npc, updates in changes["npc_updates"].items():
            if npc in self.current_world_state.npcs:
                self.current_world_state.npcs[npc].update(updates)
            else:
                self.current_world_state.npcs[npc] = updates

    def _apply_character_changes(self, changes: Dict):
        """Apply character state changes."""
        # Update status
        self.current_character.status.update(changes["status_updates"])
        
        # Update inventory
        for item in changes["inventory_changes"]:
            if item.get("action") == "add":
                self.current_character.inventory.append(item["item"])
            elif item.get("action") == "remove":
                # Find and remove item
                pass
        
        # Add new history entries
        self.current_character.history.extend(changes["new_history"])
        
        # Update stats
        self.current_character.stats.update(changes["stat_changes"])

    def _save_world_state(self):
        """Save current world state to file."""
        with open(self.persist_dir / "world_state.json", "w") as f:
            json.dump({
                "timestamp": self.current_world_state.timestamp.isoformat(),
                "locations": self.current_world_state.locations,
                "events": self.current_world_state.events,
                "current_conditions": self.current_world_state.current_conditions,
                "npcs": self.current_world_state.npcs
            }, f, indent=2)

    def _save_character_state(self):
        """Save current character state to file."""
        with open(self.persist_dir / "character_state.json", "w") as f:
            json.dump(self.current_character.__dict__, f, indent=2)

    def _update_world_rag(self):
        """Update world RAG system with current state."""
        # Convert world state to documents
        documents = [
            Document(text=json.dumps(location_data), metadata={"location": location_name})
            for location_name, location_data in self.current_world_state.locations.items()
        ]
        documents.extend([
            Document(text=json.dumps(event), metadata={"type": "event"})
            for event in self.current_world_state.events[-10:]  # Last 10 events
        ])
        
        # Update vector store
        VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context,
            vector_store=self.world_store
        )

    def _update_character_rag(self):
        """Update character RAG system with current state."""
        documents = [
            Document(
                text=json.dumps(self.current_character.__dict__),
                metadata={"type": "character_state"}
            ),
            Document(
                text=json.dumps(self.current_character.history[-5:]),
                metadata={"type": "character_history"}
            )
        ]
        
        VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context,
            vector_store=self.character_store
        )

    def _load_current_states(self):
        """Load saved states if they exist."""
        try:
            if (self.persist_dir / "world_state.json").exists():
                with open(self.persist_dir / "world_state.json", "r") as f:
                    data = json.load(f)
                    self.current_world_state = WorldState(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        locations=data["locations"],
                        events=data["events"],
                        current_conditions=data["current_conditions"],
                        npcs=data["npcs"]
                    )
            
            if (self.persist_dir / "character_state.json").exists():
                with open(self.persist_dir / "character_state.json", "r") as f:
                    data = json.load(f)
                    self.current_character = Character(**data)
        except Exception as e:
            self.logger.error(f"Error loading saved states: {e}")