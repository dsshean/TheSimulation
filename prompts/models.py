from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from enum import Enum
from datetime import datetime

# Enums for constrained choices
class NodeType(str, Enum):
    consciousness = "consciousness"
    trait = "trait"
    skill = "skill"
    state = "state"
    memory = "memory"

class RelationType(str, Enum):
    family = "family"
    friend = "friend"
    professional = "professional"
    community = "community"

class StatsTrend(str, Enum):
    A = "A"
    B = "B"
    C = "C"

# Identity Models
class PhysicalAttributes(BaseModel):
    biometric: str = Field(..., description="Height, weight, DNA markers")
    health: str = Field(..., description="Current conditions, historical records")
    capabilities: str = Field(..., description="Strength, endurance, skills")
    appearance: str = Field(..., description="Features, changes over time")

class PsychologicalAttributes(BaseModel):
    personality: str = Field(..., description="Big Five model scores")
    cognitive: str = Field(..., description="Decision patterns, behavior models")
    emotional: str = Field(..., description="Current state, historical patterns")
    consciousness: str = Field(..., description="Awareness states, thought patterns")

class DemographicAttributes(BaseModel):
    age: int = Field(..., description="Current age in years")
    gender: str = Field(..., description="Gender identity")
    ethnicity: str = Field(..., description="Cultural background")
    nationality: str = Field(..., description="Citizenship status")

class IdentityNode(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    type: NodeType
    attributes: Dict[str, Dict[str, str]]
    state: str = Field(..., description="Current node state")
    timestamp: datetime = Field(..., description="Last update time")

# Relationship Models
class RelationshipAttributes(BaseModel):
    connection_type: str = Field(..., description="Specific relationship category")
    strength: int = Field(..., ge=0, le=100, description="Integer 0-100")
    duration: str = Field(..., description="Time period in ISO format")
    quality: int = Field(..., ge=-100, le=100, description="Integer -100 to +100")
    influence_level: int = Field(..., ge=0, le=100, description="Integer 0-100")
    interaction_frequency: str = Field(..., description="Average interactions per time unit")

class RelationshipNode(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    type: RelationType
    attributes: RelationshipAttributes

# Network Metrics
class NetworkStatistics(BaseModel):
    diameter: int = Field(..., description="Integer representing maximum path length")
    average_path_length: float = Field(..., description="Float representing average distance")
    clustering_coefficient: float = Field(..., ge=0, le=1, description="Float 0-1")
    modularity: float = Field(..., description="Float representing community structure")

class ValidationResults(BaseModel):
    structural: float = Field(..., ge=0, le=1, description="Float 0-1 indicating graph integrity")
    temporal: float = Field(..., ge=0, le=1, description="Float 0-1 indicating time consistency")
    psychological: float = Field(..., ge=0, le=1, description="Float 0-1 indicating behavioral realism")

# Time-based Models
class AgeSummary(BaseModel):
    first_third: str = Field(..., description="Yearly summary of first third of life")
    second_third: str = Field(..., description="Monthly summary of second third of life")
    final_third: Dict[str, str] = Field(..., description="Daily and hourly breakdown of final third")

# World State Models
class WorldState(BaseModel):
    time_tick: str = Field(..., description="Current time tick")
    world_delta: str = Field(..., description="Changes in world state")
    narration_delta: str = Field(..., description="Narrative changes")
    self_consistency_check: str = Field(..., description="Consistency validation")
    subject_thought_process: str = Field(..., description="Subject's thoughts")
    actions: str = Field(..., description="Current actions")
    stats_trend: StatsTrend = Field(..., description="Statistical trend")
    state_description: Literal["FLAT", "SIDEWAYS", "RANGE-BOUND", "CONSOLIDATING"] = Field(
        ..., description="Market state description"
    )

# Main Simulation Model
class SimulationState(BaseModel):
    identity_nodes: List[IdentityNode]
    relationship_nodes: List[RelationshipNode]
    network_metrics: NetworkStatistics
    validation: ValidationResults
    world_state: WorldState
    age_summary: AgeSummary

    class Config:
        json_schema_extra = {
            "example": {
                "identity_nodes": [{
                    "id": "ID001",
                    "type": "consciousness",
                    "attributes": {
                        "physical": {
                            "biometric": "Height: 175cm, Weight: 70kg",
                            "health": "Excellent condition, no chronic issues"
                        }
                    },
                    "state": "active",
                    "timestamp": "2024-03-25T12:00:00Z"
                }]
                # Add more examples as needed
            }
        }