from dataclasses import dataclass, field
from typing import List, Dict, Optional
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.vector_stores import QdrantVectorStore
from llama_index.llms import OpenAI
import json
from datetime import datetime
import logging
from pathlib import Path
import random

@dataclass
class CharacterPersona:
    name: str
    background: str
    goals: List[str]
    personality_traits: List[str]
    relationships: Dict[str, str]  # NPC name -> relationship type
    preferences: Dict[str, List[str]]  # Category -> list of preferences
    daily_routine: Dict[str, str]  # Hour -> typical activity
    behavioral_triggers: Dict[str, str]  # Situation -> likely reaction

@dataclass
class CharacterAction:
    action_type: str
    target: str
    location: str
    description: str
    motivation: str
    timestamp: datetime
    consequences: List[str] = field(default_factory=list)

class CharacterAgent:
    def __init__(
        self,
        persona: CharacterPersona,
        openai_api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        memory_collection: str = "character_memory",
        model_name: str = "gpt-4",
        persist_dir: str = "./character_data"
    ):
        """
        Initialize the Character Agent with a specific persona.
        """
        self.logger = logging.getLogger(__name__)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.persona = persona
        self.current_state = {
            "location": "",
            "status": "idle",
            "current_goal": None,
            "energy": 100,
            "mood": "neutral",
            "inventory": [],
            "recent_actions": []
        }
        
        # Initialize LLM
        self.llm = OpenAI(temperature=0.7, model=model_name, api_key=openai_api_key)
        
        # Initialize memory vector store
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.memory_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=memory_collection
        )
        
        # Load previous state if exists
        self._load_state()

    def decide_action(self, world_state: Dict, current_hour: int) -> CharacterAction:
        """
        Decide on the next action based on persona, current state, and world state.
        """
        # Create decision prompt
        decision_prompt = self._create_decision_prompt(world_state, current_hour)
        
        # Get action decision from LLM
        response = self.llm.complete(decision_prompt)
        
        # Parse action from response
        action = self._parse_action_response(response.text)
        
        # Update character state based on action
        self._update_state_from_action(action)
        
        # Save current state
        self._save_state()
        
        return action

    def _create_decision_prompt(self, world_state: Dict, current_hour: int) -> str:
        """
        Create a prompt for action decision based on all available context.
        """
        return f"""
        As {self.persona.name}, with the following characteristics:
        - Background: {self.persona.background}
        - Current Goals: {', '.join(self.persona.goals)}
        - Personality: {', '.join(self.persona.personality_traits)}
        - Typical activity at this hour: {self.persona.daily_routine.get(str(current_hour), 'No specific routine')}

        Current State:
        - Location: {self.current_state['location']}
        - Status: {self.current_state['status']}
        - Energy Level: {self.current_state['energy']}
        - Mood: {self.current_state['mood']}
        - Current Goal: {self.current_state['current_goal']}
        - Recent Actions: {json.dumps(self.current_state['recent_actions'][-3:], indent=2)}

        World State:
        - Current Location Details: {json.dumps(world_state['locations'].get(self.current_state['location'], {}), indent=2)}
        - NPCs Present: {self._get_npcs_at_location(world_state, self.current_state['location'])}
        - Current Conditions: {json.dumps(world_state['current_conditions'], indent=2)}
        - Recent Events: {json.dumps(world_state['events'][-3:], indent=2)}

        Decide on the next action for the next hour. Consider:
        1. Current goals and priorities
        2. Energy level and mood
        3. Available opportunities in the environment
        4. Relationships with present NPCs
        5. Current time and daily routine
        6. Recent events and their impact
        7. Personality traits and typical behaviors

        Provide the action in the following format:
        Action Type: [type of action]
        Target: [target of action, if any]
        Location: [where the action takes place]
        Description: [detailed description of the action]
        Motivation: [reason for taking this action]
        Consequences: [expected outcomes]
        """

    def _parse_action_response(self, response_text: str) -> CharacterAction:
        """
        Parse LLM response into a structured action.
        """
        # Parse the response text to extract action components
        # This is a simplified example - you'd want more robust parsing
        lines = response_text.strip().split('\n')
        action_dict = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                action_dict[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # Create action object
        action = CharacterAction(
            action_type=action_dict.get('action_type', 'idle'),
            target=action_dict.get('target', 'none'),
            location=action_dict.get('location', self.current_state['location']),
            description=action_dict.get('description', ''),
            motivation=action_dict.get('motivation', ''),
            timestamp=datetime.now(),
            consequences=[c.strip() for c in action_dict.get('consequences', '').split(';')]
        )
        
        return action

    def _update_state_from_action(self, action: CharacterAction):
        """
        Update character state based on the chosen action.
        """
        # Update location if changed
        if action.location != self.current_state['location']:
            self.current_state['location'] = action.location
        
        # Update energy level based on action type
        energy_cost = {
            'walk': 5,
            'run': 10,
            'fight': 20,
            'work': 15,
            'rest': -10,  # Recover energy
            'eat': -8,
            'talk': 2,
            'idle': 1
        }.get(action.action_type.lower(), 5)
        
        self.current_state['energy'] = max(0, min(100, self.current_state['energy'] - energy_cost))
        
        # Update status
        self.current_state['status'] = action.action_type
        
        # Add to recent actions
        self.current_state['recent_actions'].append({
            'action_type': action.action_type,
            'description': action.description,
            'timestamp': action.timestamp.isoformat()
        })
        
        # Keep only last 10 actions
        self.current_state['recent_actions'] = self.current_state['recent_actions'][-10:]
        
        # Update mood based on action success and energy level
        self._update_mood(action)

    def _update_mood(self, action: CharacterAction):
        """Update character mood based on action and state."""
        # Simple mood update logic based on energy and action
        if self.current_state['energy'] < 20:
            self.current_state['mood'] = 'exhausted'
        elif self.current_state['energy'] < 50:
            self.current_state['mood'] = 'tired'
        elif 'rest' in action.action_type.lower() or 'eat' in action.action_type.lower():
            self.current_state['mood'] = 'refreshed'
        else:
            # Random mood based on personality traits
            possible_moods = {
                'cheerful': 3 if 'optimistic' in self.persona.personality_traits else 1,
                'focused': 3 if 'determined' in self.persona.personality_traits else 1,
                'curious': 3 if 'inquisitive' in self.persona.personality_traits else 1,
                'neutral': 1
            }
            moods = []
            for mood, weight in possible_moods.items():
                moods.extend([mood] * weight)
            self.current_state['mood'] = random.choice(moods)

    def _get_npcs_at_location(self, world_state: Dict, location: str) -> List[str]:
        """Get list of NPCs at the current location."""
        npcs_present = []
        for npc_name, npc_data in world_state['npcs'].items():
            if npc_data.get('location') == location:
                npcs_present.append(npc_name)
        return npcs_present

    def _save_state(self):
        """Save current character state to file."""
        state_data = {
            'persona': self.persona.__dict__,
            'current_state': self.current_state
        }
        with open(self.persist_dir / f"{self.persona.name}_state.json", 'w') as f:
            json.dump(state_data, f, indent=2)

    def _load_state(self):
        """Load character state from file if it exists."""
        state_file = self.persist_dir / f"{self.persona.name}_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # Load persona data
                    self.persona = CharacterPersona(**data['persona'])
                    # Load current state
                    self.current_state = data['current_state']
            except Exception as e:
                self.logger.error(f"Error loading character state: {e}")

# Example usage
# def main():
#     # Create character persona
#     persona = CharacterPersona(
#         name="Adventurer Alex",
#         background="Wandering hero seeking fortune and glory",
#         goals=[
#             "Find legendary artifacts",
#             "Help those in need",
#             "Build reputation"
#         ],
#         personality_traits=[
#             "brave",
#             "curious",
#             "helpful",
#             "determined"
#         ],
#         relationships={
#             "merchant": "friendly",
#             "guard": "respectful",
#             "innkeeper": "regular customer"
#         },
#         preferences={
#             "activities": ["exploring", "combat training", "helping others"],
#             "locations": ["tavern", "marketplace", "wilderness"],
#             "items": ["weapons", "maps", "magical artifacts"]
#         },
#         daily_routine={
#             "8": "training",
#             "12": "exploring",
#             "18": "socializing",
#             "22": "resting"
#         },
#         behavioral_triggers={
#             "danger": "protect others",
#             "injustice": "intervene",
#             "mystery": "investigate"
#         }
#     )
    
#     # Initialize character agent
#     agent = CharacterAgent(
#         persona=persona,
#         openai_api_key="your_openai_api_key"
#     )
    
#     # Example world state
#     world_state = {
#         "locations": {
#             "town_square": {
#                 "description": "Busy central plaza",
#                 "current_activity": "market day",
#                 "opportunities": ["trade", "information gathering", "helping merchants"]
#             }
#         },
#         "events": [
#             {"type": "market_day", "location": "town_square", "description": "Busy trading day"}
#         ],
#         "current_conditions": {
#             "time_of_day": "morning",
#             "weather": "sunny",
#             "crowd_level": "high"
#         },
#         "npcs": {
#             "merchant": {"location": "town_square", "status": "selling"},
#             "guard": {"location": "town_square", "status": "patrolling"}
#         }
#     }
    
#     # Decide on action
#     action = agent.decide_action(world_state, current_hour=9)
#     print(f"Character Action:\n{json.dumps(action.__dict__, indent=2)}")

# if __name__ == "__main__":
#     main()