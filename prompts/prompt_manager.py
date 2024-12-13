from typing import Optional
from .models import SimulationState

class PromptManager:
    @staticmethod
    def generate_simulation_prompt(current_state: Optional[SimulationState] = None) -> str:
        base_prompt = """You are tasked with generating a detailed personal history of a simulated human being.
        Your response must conform to the following structure and constraints:
        
        1. Identity and Basic Information
        - Generate complete demographic, physical, and psychological attributes
        - Ensure all numerical values are within specified ranges
        - Maintain temporal consistency in the life history
        
        2. Relationships and Social Network
        - Create realistic relationship networks with proper strength metrics
        - Ensure reciprocal relationships are consistent
        - Model relationship evolution over time
        
        3. Life History
        - Divide the life story into three temporal segments
        - Provide appropriate detail granularity for each period
        - Maintain causal consistency in life events
        
        4. System Constraints
        - All metrics must be within specified ranges (0-100 or -100 to +100)
        - Timestamps must be in ISO format
        - State transitions must have valid triggers and durations
        """
        
        if current_state:
            # Add context from current state if available
            base_prompt += f"\nCurrent state context:\n{current_state.json(indent=2)}"
            
        return base_prompt

    @staticmethod
    def validate_response(response: dict) -> SimulationState:
        """Validate and parse response into SimulationState model"""
        return SimulationState.parse_obj(response)