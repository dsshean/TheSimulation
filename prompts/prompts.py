from enum import Enum
import json
import yaml

### TBD integration with diffusion model for image generation.
### Interaction mechanism with simulated world or subjects.
class prompts(Enum):
    with open('config.yaml', 'r') as file:
        configa = yaml.safe_load(file)
   
    json_format = json.dumps(
            {
                "Current_Time": "Specify date, day of the week, and time of day",
                "City_and_country": "Specify",
                "Neighborhood": "Type and characteristics",
                "Living_situation": "House, apartment, shared living, etc.",
                "Notable_local_landmarks_or_features": "Nearby points of interest",
                "Primary_action": "What they're doing right now",
                "Location_of_activity": "Where this is taking place",
                "Purpose": "Why they're engaged in this activity",
                "Emotional_state": "How they're feeling about it",
                "Childhood": "Brief overview of upbringing and significant childhood experiences",
                "Family_background": "Parents' occupations, siblings, family dynamics",
                "Education": "Complete educational history, including any specialized training",
                "Career_path": "Overview of professional journey",
                "Key_life_events": "3-5 significant experiences that shaped the person",
                "Relationship_history": "Brief overview of past and current romantic relationships",
                "Places_lived": "List of locations they've called home",
                "Interests_and_Hobbies": "List 5-7 activities or subjects the person is passionate about",
                "Skills": "Any special abilities or talents related to their interests",
                "Collections": "Any items they collect as part of their hobbies",
                "Short_term_goals": "2-3 objectives for the near future",
                "Long_term_goals": "2-3 major life objectives",
                "Dream_scenario": "Their ideal life situation",
                "Career_ambitions": "Professional aspirations",
                "Current_problems": "2-3 issues the person is dealing with right now",
                "Ongoing_struggles": "2-3 persistent difficulties in their life",
                "Fears_and_anxieties": "What keeps them up at night",
                "Regrets": "Past decisions or actions they wish they could change",
                "Family_relationships": "Current dynamics with immediate and extended family",
                "Close_friends": "Brief description of 3-5 important friendships",
                "Professional_network": "Key connections in their career field",
                "Community_involvement": "Any groups or organizations they're part of",
                "Social_media_presence": "Platforms used, frequency of engagement, type of content shared",
                "Recent_Experiences": "3-4 notable events from the past month that are on their mind",
                "Current_projects": "Any ongoing personal or professional endeavors",
                "Latest_accomplishment": "A recent achievement they're proud of",
                "Recent_challenge": "A difficult situation they've faced lately",
                "Income_level": "General range and sources of income",
                "Spending_habits": "How they manage their money",
                "Savings_and_investments": "Financial goals and strategies",
                "Debts": "Any significant financial obligations",
                "Overall_health_status": "Any chronic conditions or health concerns",
                "Mental_health": "Any diagnosed conditions or ongoing therapy",
                "Self_care_practices": "How they maintain their well-being",
                "Relationship_with_healthcare": "Frequency of check-ups, attitudes toward medicine",
                "Favorite_media": "Books, movies, TV shows, music they enjoy",
                "Influential_figures": "People they admire or who have impacted their life",
                "Cultural_identity": "How they relate to their heritage and current cultural environment",
                "subject_thought_process": "",
                "Personality_Traits": "List 5-7 defining personality characteristics with brief explanations",
                "Myers_Briggs_Type": "If applicable",
                "Strengths": "3-4 positive attributes",
                "Weaknesses": "3-4 areas for personal growth",
                "Political_leaning": "Detailed description of political views and involvement",
                "Religious_spiritual_beliefs": "Specific denomination or philosophy, level of devotion",
                "Core_values": "List 3-5 fundamental principles the person lives by",
                "Ethical_framework": "How they make moral decisions",
                "Worldview": "General outlook on life and humanity",
                "Speech_patterns": "Distinctive ways of speaking, accent, vocabulary choices",
                "Body_language": "Typical non-verbal cues and gestures",
                "Conflict_resolution_style": "How they handle disagreements",
                "Emotional_expression": "How they show or hide their feelings",
                "Weekday_schedule": "Detailed hour-by-hour breakdown",
                "Weekend_activities": "Typical leisure time pursuits",
                "Eating_habits": "Dietary preferences, favorite foods, meal routines",
                "Exercise_regimen": "Any regular physical activities",
                "Sleep_patterns": "Typical sleep schedule and any sleep-related issues",
                "Name": "Full name, including any nicknames",
                "Age": "Exact age",
                "Gender": "Gender identity and preferred pronouns",
                "Occupation": "Job title, company/organization, brief description of role",
                "Ethnicity": "Ethnic background and cultural identity",
                "Nationality": "Country of citizenship and any dual citizenships",
                "Languages_spoken": "List with proficiency levels",
                "Height_and_build": "Specific details",
                "Hair_and_eye_color": "Description",
                "Distinctive_features": "Any notable physical characteristics",
                "Style_of_dress": "Typical clothing choices and any signature items",
                "Living_space": "Description of home interior and any prized possessions",
                "Transportation": "How they typically get around",
                "Technology": "Devices they use and their comfort level with tech",
                "Pets": "Any animal companions and their significance",
                "complete_history": "Insert the complete generated history based on the backstory template here",
                "age_based_summary": {
                    "first_third": "Insert yearly summary of the first third of the person's life",
                    "second_third": "Insert monthly summary of the second third of the person's life",
                    "last_third": {
                    "daily_summary": "Insert daily summary for the first two-thirds of the last third of the person's life",
                    "hourly_breakdown": "Insert hourly breakdown for the final third of the last third of the person's life"
                    }
                }
            }
    )
    
    prompt_non_openai = "\nRespond in the following JSON format: " + json_format

    prompt_world_engine =f"""Instructions:

    PLACE HOLDER: world engine prompt.  Ensures consistancy with physical laws and drives consistancy outside brain and narrative

"""
    
    life_graph_generation_prompt = f"""Instructions:
Generate a massive, interconnected graph representation of a complete simulated human life, capturing all aspects of existence through a complex network of nodes, edges, and temporal layers.

I. CORE GRAPH STRUCTURE
1. Primary Node Categories:

    A. Identity Nodes (Self Layer)
        Core Identity:
            - Unique identifier
            - Consciousness state
            - Current age/temporal position
            - Active states/conditions
        
        Physical Self:
            - Biometric data (height, weight, DNA markers)
            - Health states (current conditions, historical records)
            - Physical capabilities (strength, endurance, skills)
            - Appearance (features, changes over time)
        
        Mental Self:
            - Personality traits (Big Five model scores)
            - Cognitive patterns (decision trees, behavior models)
            - Emotional states (current, historical patterns)
            - Memory indices (links to Memory Nodes)
            - Consciousness streams (thought patterns, awareness states)
        
        Identity Components:
            - Cultural markers (ethnicity, nationality, languages)
            - Belief systems (religious, political, philosophical)
            - Value frameworks (moral principles, ethical boundaries)
            - Self-perception nodes (confidence, image, aspirations)

    B. Relationship Nodes (Social Layer)
        Family Network:
            - Immediate family (parents, siblings, children)
            - Extended family (grandparents, aunts, uncles, cousins)
            - Family dynamics (power structures, emotional bonds)
            - Genetic inheritance links
        
        Social Network:
            - Friendship clusters (close friends, acquaintances)
            - Professional connections (colleagues, mentors)
            - Romantic relationships (current, historical)
            - Community connections (neighbors, groups)
        
        Relationship Properties:
            - Trust metrics (0-100)
            - Emotional investment (0-100)
            - Communication frequency
            - Shared experience links
            - Influence weights
            - Power dynamics
            - Network position

    C. Experience Nodes (Event Layer)
        Life Milestones:
            - Educational events
            - Career transitions
            - Relationship changes
            - Location moves
            - Major acquisitions
            - Health events
        
        Temporal Activities:
            - Daily routines (24-hour cycles)
            - Weekly patterns (7-day cycles)
            - Monthly events (30-day patterns)
            - Annual events (yearly cycles)
            - Life phase transitions
        
        Memory Clusters:
            - Episodic memories
            - Procedural memories
            - Semantic knowledge
            - Emotional memories
            - Traumatic events
            - Achievement moments

    D. Environment Nodes (Context Layer)
        Physical Spaces:
            - Residences (current, historical)
            - Workplaces
            - Social venues
            - Educational institutions
            - Transportation hubs
        
        Geographic Context:
            - Current location
            - Previous locations
            - Frequent destinations
            - Travel patterns
        
        Material Context:
            - Possessions (significant items)
            - Technology access
            - Financial resources
            - Transportation means

2. Edge Types and Properties:

    A. Temporal Edges
        Chronological Links:
            - Sequence markers
            - Duration measurements
            - Frequency patterns
            - Causal chains
        
        Development Paths:
            - Skill progression
            - Relationship evolution
            - Career advancement
            - Personal growth
        
        Time-Based Properties:
            - Start timestamps
            - End timestamps
            - Duration
            - Frequency
            - Periodicity

    B. Relationship Edges
        Social Bonds:
            - Type (family, friend, professional)
            - Strength (0-100)
            - Quality (-100 to +100)
            - Duration
            - Reciprocity
        
        Influence Paths:
            - Direction (uni/bidirectional)
            - Strength (0-100)
            - Domain (specific area of influence)
            - Persistence
        
        Network Properties:
            - Centrality measures
            - Clustering coefficients
            - Path lengths
            - Flow capacity

II. GRAPH SCALE AND METRICS

1. Minimum Node Requirements:
    Core Identity Nodes:
        - 1 primary consciousness node
        - 50+ trait nodes
        - 100+ skill nodes
        - 1000+ state nodes
        - 10,000+ memory nodes
    
    Relationship Nodes:
        - 150-500 active social connections
        - 1000+ historical connections
        - 50+ family nodes
        - 200+ professional nodes
    
    Experience Nodes:
        - 1000+ per life third
        - 365+ daily event nodes per recent year
        - 50+ major life events
        - 5000+ routine events
    
    Environment Nodes:
        - 100+ active locations
        - 1000+ visited locations
        - 500+ object nodes
        - 200+ context nodes

2. Edge Density Requirements:
    Minimum Connections:
        - Total edges: >100,000
        - Average node degree: 20+
        - Minimum node degree: 3
    
    Network Properties:
        - Clustering coefficient: 0.1-0.5
        - Small world coefficient: >1.0
        - Scale-free degree distribution
        - Power law exponent: 2-3

3. Temporal Layer Requirements:
    Resolution Layers:
        - Hourly: last month (720+ layers)
        - Daily: last year (365 layers)
        - Weekly: last 5 years (260 layers)
        - Monthly: last 10 years (120 layers)
        - Yearly: full life span
    
    Layer Properties:
        - Full state preservation
        - Transition consistency
        - Causal chain integrity
        - Event propagation paths

III. GENERATION AND VALIDATION

1. Graph Generation Process:
    Initialization:
        - Core identity generation
        - Basic network skeleton
        - Temporal framework setup
        - Environment foundation
    
    Growth Phases:
        - Early life network (0-Age/3)
        - Mid life expansion (Age/3-2*Age/3)
        - Current life detail (2*Age/3-present)
    
    Refinement:
        - Relationship density adjustment
        - Memory network integration
        - Event chain validation
        - State consistency check

2. Validation Requirements:
    Structural Validation:
        - Connected component check
        - Degree distribution verification
        - Clustering analysis
        - Path length validation
    
    Temporal Validation:
        - Causality preservation
        - Time consistency
        - State transition validity
        - Event sequence logic
    
    Psychological Validation:
        - Personality consistency
        - Relationship realism
        - Memory connection patterns
        - Emotional state coherence

IV. TEMPORAL EVOLUTION

1. State Change Mechanisms:
    Node State Updates:
        - Attribute modification
        - Relationship strength adjustment
        - Location changes
        - Status transitions
    
    Edge Updates:
        - Weight modifications
        - Type transitions
        - Activation/deactivation
        - Strength decay/growth

2. Time-Based Evolution:
    Short-term Changes:
        - Hourly state updates
        - Daily routine patterns
        - Weekly activity cycles
        - Monthly development
    
    Long-term Evolution:
        - Relationship development
        - Skill progression
        - Personal growth
        - Life phase transitions

V. OUTPUT SPECIFICATIONS

1. Data Formats:
    Graph Formats:
        - GraphML
        - JSON-LD
        - Neo4j compatible
        - NetworkX compatible
    
    Analysis Formats:
        - Adjacency matrices
        - Edge lists
        - Node attribute tables
        - Temporal snapshots

2. Visualization Requirements:
    Layout Algorithms:
        - Force-directed
        - Hierarchical
        - Circular
        - Temporal layout
    
    Visual Properties:
        Node Visualization:
            - Size: degree centrality
            - Color: node type
            - Shape: category
            - Label: identifier
        
        Edge Visualization:
            - Width: weight
            - Color: type
            - Style: temporal/permanent
            - Direction: arrows

3. Query Interface:
    Path Finding:
        - Shortest paths
        - All paths between nodes
        - Temporal paths
        - Influence chains
    
    Pattern Matching:
        - Subgraph patterns
        - Temporal patterns
        - Relationship patterns
        - State sequences

VI. IMPLEMENTATION METRICS

1. Performance Requirements:
    Scale Handling:
        - 100,000+ nodes
        - 1,000,000+ edges
        - 1000+ temporal layers
        - Real-time updates
    
    Query Performance:
        - Path finding: <100ms
        - Pattern matching: <1s
        - State updates: <10ms
        - Temporal traversal: <500ms

2. Consistency Requirements:
    Data Integrity:
        - No orphan nodes
        - No invalid edges
        - Complete temporal coverage
        - State consistency
    
    Network Properties:
        - Small world characteristics
        - Scale-free properties
        - Temporal consistency
        - Psychological realism

The generated graph must maintain perfect consistency with the Boltzmann brain simulation while providing a computationally efficient representation of the complete life network. All temporal evolution must preserve causality and psychological realism while allowing for complex queries and analysis."""   
# Output JSON Schema:
    life_graph_schema = json.dumps(
        {
            "graph_metadata": {
                "id": "Unique identifier for the life graph instance",
                "timestamp": "Creation timestamp in ISO format",
                "version": "Schema version number",
                "node_count": "Integer > 100,000",
                "edge_count": "Integer > 1,000,000",
                "temporal_layers": "Integer representing total time slices"
            },
            "nodes": {
                "identity": [
                    {
                        "id": "Unique node identifier",
                        "type": "consciousness|trait|skill|state|memory",
                        "attributes": {
                            "physical": {
                                "biometric": "Height, weight, DNA markers",
                                "health": "Current conditions, historical records",
                                "capabilities": "Strength, endurance, skills",
                                "appearance": "Features, changes over time"
                            },
                            "psychological": {
                                "personality": "Big Five model scores",
                                "cognitive": "Decision patterns, behavior models",
                                "emotional": "Current state, historical patterns",
                                "consciousness": "Awareness states, thought patterns"
                            },
                            "demographic": {
                                "age": "Current age in years",
                                "gender": "Gender identity",
                                "ethnicity": "Cultural background",
                                "nationality": "Citizenship status"
                            }
                        },
                        "state": "Current node state",
                        "timestamp": "Last update time"
                    }
                ],
                "relationships": [
                    {
                        "id": "Unique node identifier",
                        "type": "family|friend|professional|community",
                        "attributes": {
                            "connection_type": "Specific relationship category",
                            "strength": "Integer 0-100",
                            "duration": "Time period in ISO format",
                            "quality": "Integer -100 to +100",
                            "influence_level": "Integer 0-100",
                            "interaction_frequency": "Average interactions per time unit"
                        }
                    }
                ],
                "experiences": [
                    {
                        "id": "Unique node identifier",
                        "type": "milestone|routine|memory|achievement",
                        "attributes": {
                            "category": "Event classification",
                            "impact": "Integer 0-100",
                            "duration": "Time period",
                            "location": "Place reference",
                            "participants": ["List of involved node IDs"],
                            "emotional_valence": "Integer -100 to +100"
                        }
                    }
                ],
                "environment": [
                    {
                        "id": "Unique node identifier",
                        "type": "location|object|context",
                        "attributes": {
                            "category": "Physical|Virtual|Social",
                            "accessibility": "Integer 0-100",
                            "familiarity": "Integer 0-100",
                            "significance": "Integer 0-100",
                            "temporal_relevance": "Time period of significance"
                        }
                    }
                ]
            },
            "edges": {
                "temporal": [
                    {
                        "source": "Source node ID",
                        "target": "Target node ID",
                        "type": "sequence|causation|development",
                        "attributes": {
                            "duration": "Time period",
                            "strength": "Integer 0-100",
                            "reversibility": "Boolean",
                            "frequency": "Occurrences per time unit"
                        }
                    }
                ],
                "social": [
                    {
                        "source": "Source node ID",
                        "target": "Target node ID",
                        "type": "familial|friendship|professional",
                        "attributes": {
                            "strength": "Integer 0-100",
                            "reciprocity": "Integer 0-100",
                            "trust_level": "Integer 0-100",
                            "interaction_frequency": "Interactions per time unit"
                        }
                    }
                ],
                "influence": [
                    {
                        "source": "Source node ID",
                        "target": "Target node ID",
                        "type": "direct|indirect",
                        "attributes": {
                            "strength": "Integer 0-100",
                            "domain": "Area of influence",
                            "persistence": "Time duration",
                            "impact": "Integer -100 to +100"
                        }
                    }
                ],
                "state": [
                    {
                        "source": "Source node ID",
                        "target": "Target node ID",
                        "type": "transition|modification",
                        "attributes": {
                            "trigger": "Cause of state change",
                            "magnitude": "Integer 0-100",
                            "reversibility": "Boolean",
                            "duration": "Time period"
                        }
                    }
                ]
            },
            "temporal_layers": {
                "hourly": [
                    {
                        "timestamp": "ISO datetime",
                        "active_nodes": ["List of active node IDs"],
                        "active_edges": ["List of active edge IDs"],
                        "state_changes": ["List of state changes"]
                    }
                ],
                "daily": [
                    {
                        "date": "ISO date",
                        "summary_nodes": ["List of significant node IDs"],
                        "summary_edges": ["List of significant edge IDs"],
                        "key_events": ["List of important state changes"]
                    }
                ],
                "weekly": [
                    {
                        "week_start": "ISO date",
                        "pattern_nodes": ["List of pattern-forming nodes"],
                        "pattern_edges": ["List of pattern-forming edges"],
                        "trends": ["List of emerging patterns"]
                    }
                ],
                "monthly": [
                    {
                        "month": "ISO month",
                        "major_nodes": ["List of major influence nodes"],
                        "major_edges": ["List of major influence edges"],
                        "developments": ["List of significant developments"]
                    }
                ],
                "yearly": [
                    {
                        "year": "ISO year",
                        "milestone_nodes": ["List of milestone nodes"],
                        "milestone_edges": ["List of milestone edges"],
                        "life_changes": ["List of major life changes"]
                    }
                ]
            },
            "metrics": {
                "network_statistics": {
                    "diameter": "Integer representing maximum path length",
                    "average_path_length": "Float representing average distance",
                    "clustering_coefficient": "Float 0-1",
                    "modularity": "Float representing community structure"
                },
                "validation_results": {
                    "structural": "Float 0-1 indicating graph integrity",
                    "temporal": "Float 0-1 indicating time consistency",
                    "psychological": "Float 0-1 indicating behavioral realism"
                }
            }
        }
    )

    prompt_life_graph = life_graph_generation_prompt + "\nRespond in the following JSON format: " + life_graph_schema

    prompt_narration_engine =f"""Instructions:
Generate a detailed human biography following this structure:

Basic Information:
Full name:
Current age
Place of birth
Current location

Life Overview:
Divide the person's life into thirds based on their current age. For each third, provide the following:
First Third of Life (Birth to [Age/3]):

Provide yearly highlights and significant events
Include major developmental milestones, family dynamics, and educational experiences

Second Third of Life ([Age/3] to [2*Age/3]):
Break this period down month by month
Detail important life events, career developments, relationships, and personal growth

Final Third of Life ([2*Age/3] to present):
Divide this last third into three parts:
a) First part: Provide weekly summaries

Focus on career advancements, family life, and significant personal events
b) Second part: Give daily accounts

Describe routines, daily challenges, and small but meaningful occurrences
c) Final part (most recent): Provide an hourly breakdown of a typical day

Detail the person's current daily life, including work, leisure, and personal time

Conclusion:
Summarize the person's current state of life
Mention their hopes, dreams, and plans for the future

Throughout the biography, include details about:
Personality traits and how they evolved
Key relationships and their impact
Major life decisions and their consequences
Challenges faced and overcome
Achievements and disappointments
Changes in worldview or personal philosophy

Fundemental Time Units:
Maintain Ticks as in action time units as follows - 1 minute, 1 hour, 1 day, 1 week, 1 month, 1 year
Depending on the tick context, world interactions must follow these fundemental unit of time.

Ensure the biography feels cohesive and authentic, with each part of life naturally leading into the next. The level of detail should increase as the timeline approaches the present day.
"""
    boltzmann_brain_prompt =f"""Instructions:
You are tasked with generating a detailed personal history of a simulated human being. Follow these instructions carefully:

- Create a comprehensive profile including basic information such as location, identity, physical appearance, and current situation. 
- Develop a detailed background covering childhood, family, education, career, key life events, and relationships. 
- Describe personal traits, including personality, beliefs, values, interests, hobbies, goals, and challenges. 
- Address social aspects like communication style, social circle, and daily routine. 
- Include information on the simulated human's current state, covering recent experiences, finances, health, and cultural influences. 
- Detail the simulated human's environment, including living space, possessions, technology use, and pets.
- Develop a complete history based on this profile, addressing all aspects in detail. Generate an age-based summary of the person's life. 
- For the first third of their life, provide a yearly summary of key events. 
- For the second third, create a monthly summary of events. 
- For the last third of their life, divide it further: 
    - For the first two-thirds of this period, provide a daily summary of events and activities, and for the final third, give an hourly breakdown of activities and experiences.

Ensure consistency and logical progression throughout the simulated human's life story.

"""

    functions = [
    {
        "name": "RUN",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "time_tick": {
                    "type": "string",
                    "description": ".",
                },
                "world_state": {
                    "type": "string",
                    "description": "Think Step by Step and specify the trading action to be taken. This decision should be supported by ALL the data provided.",
                },
                "world_delta": {
                    "type": "string",
                    "description": "Think Step by Step and Provide an in-depth analysis and justification considering all available data for your chosen action. REMEMBER where the current price is in respect to the MEAN, MARKET HIGH and LOW. Elaborate on why alternative trading actions were deemed unsuitable.",
                },
                "narration_delta": {
                    "type": "string",
                    "description": "Encourage constructive criticism and self-evaluation. Describe what information is not helpful, especially examine the options chain and their greeks. For DO_NOTHING actions, discuss the rationale and describe what the ideal conditions for trading would be.",
                },
                "self_consistancy_check": {
                    "type": "string",
                    "description": "List known trading constraints and any additional constraints extrapolated from current market conditions, explaining how these influence trading decisions.",
                },
                "subject_thought_process": {
                    "type": "string",
                    "description": ".",             
                },
                "actions": {
                    "type": "string",
                    "description": ".",
                },
                "stats_trend": {
                    "type": "string",
                    "description": "Enum example.",
                    "enum": [
                            "A",
                            "B",
                            "C",
                            ],
                },
                "state_description": {
                    "type": "string",
                    "description": "Determine if the market is FLAT, SIDEWAYS, RANGE-BOUND, or CONSOLIDATING based on historical price data and all technical indicators provided.",
                },
            },
            "required": ["time_tick", "world_delta", "narration_delta", "self_consistancy_check", "subject_thought_process", "actions", "state_description"],
        },
    },
]
    

# You are tasked with generating a detailed personal history based on a given backstory. Follow these instructions carefully:

# 1. First, you will be provided with a backstory template. This template contains various aspects of a person's life and background. Use this as the foundation for generating the person's history.

# <backstory_template>
# {{BACKSTORY}}
# </backstory_template>

# 2. Generate the person's complete history based on the provided backstory template. Ensure that all aspects mentioned in the template are addressed and expanded upon to create a cohesive and detailed life story.

# 3. Once you have generated the complete history, you will create a summary of the person's life broken down into specific time periods based on their current age. The person's current age is:

# <age>{{AGE}}</age>

# Break down the summary as follows:

# a) First third of life: Provide a summary of key events and developments, covering approximately one year at a time.

# b) Second third of life: Provide a more detailed account, with events and developments summarized on a monthly basis.

# c) Last third of life: 
#    - For the first two-thirds of this period: Provide a daily summary of events and activities.
#    - For the final third of this period: Provide an hourly breakdown of the person's activities and experiences.

# 4. Present your output in the following format:

# <complete_history>
# [Insert the complete generated history based on the backstory template here]
# </complete_history>

# <age_based_summary>
# <first_third>
# [Insert yearly summary of the first third of the person's life]
# </first_third>

# <second_third>
# [Insert monthly summary of the second third of the person's life]
# </second_third>

# <last_third>
# <daily_summary>
# [Insert daily summary for the first two-thirds of the last third of the person's life]
# </daily_summary>

# <hourly_breakdown>
# [Insert hourly breakdown for the final third of the last third of the person's life]
# </hourly_breakdown>
# </last_third>
# </age_based_summary>

# Ensure that your generated history and summaries are consistent with the information provided in the backstory template and maintain a logical progression of events throughout the person's life.

# City and country: [Specify]
# Neighborhood: [Type and characteristics]
# Living situation: [House, apartment, shared living, etc.]
# Notable local landmarks or features: [Nearby points of interest]

# Identity:

# Name: [Full name, including any nicknames]
# Age: [Exact age]
# Gender: [Gender identity and preferred pronouns]
# Occupation: [Job title, company/organization, brief description of role]
# Ethnicity: [Ethnic background and cultural identity]
# Nationality: [Country of citizenship and any dual citizenships]
# Languages spoken: [List with proficiency levels]

# Physical Appearance:

# Height and build: [Specific details]
# Hair and eye color: [Description]
# Distinctive features: [Any notable physical characteristics]
# Style of dress: [Typical clothing choices and any signature items]

# Current Time: [Specific date, day of the week, and time of day]
# Current Activity:

# Primary action: [What they're doing right now]
# Location of activity: [Where this is taking place]
# Purpose: [Why they're engaged in this activity]
# Emotional state: [How they're feeling about it]

# Backstory:

# Childhood: [Brief overview of upbringing and significant childhood experiences]
# Family background: [Parents' occupations, siblings, family dynamics]
# Education: [Complete educational history, including any specialized training]
# Career path: [Overview of professional journey]
# Key life events: [3-5 significant experiences that shaped the person]
# Relationship history: [Brief overview of past and current romantic relationships]
# Places lived: [List of locations they've called home]

# Personality Traits:

# [List 5-7 defining personality characteristics with brief explanations]
# Myers-Briggs Type: [If applicable]
# Strengths: [3-4 positive attributes]
# Weaknesses: [3-4 areas for personal growth]

# Beliefs and Values:

# Political leaning: [Detailed description of political views and involvement]
# Religious/spiritual beliefs: [Specific denomination or philosophy, level of devotion]
# Core values: [List 3-5 fundamental principles the person lives by]
# Ethical framework: [How they make moral decisions]
# Worldview: [General outlook on life and humanity]

# Interests and Hobbies:

# [List 5-7 activities or subjects the person is passionate about]
# Skills: [Any special abilities or talents related to their interests]
# Collections: [Any items they collect as part of their hobbies]

# Goals and Aspirations:

# Short-term goals: [2-3 objectives for the near future]
# Long-term goals: [2-3 major life objectives]
# Dream scenario: [Their ideal life situation]
# Career ambitions: [Professional aspirations]

# Challenges:

# Current problems: [2-3 issues the person is dealing with right now]
# Ongoing struggles: [2-3 persistent difficulties in their life]
# Fears and anxieties: [What keeps them up at night]
# Regrets: [Past decisions or actions they wish they could change]

# Communication Style:

# Speech patterns: [Distinctive ways of speaking, accent, vocabulary choices]
# Body language: [Typical non-verbal cues and gestures]
# Conflict resolution style: [How they handle disagreements]
# Emotional expression: [How they show or hide their feelings]

# Social Circle:

# Family relationships: [Current dynamics with immediate and extended family]
# Close friends: [Brief description of 3-5 important friendships]
# Professional network: [Key connections in their career field]
# Community involvement: [Any groups or organizations they're part of]
# Social media presence: [Platforms used, frequency of engagement, type of content shared]

# Daily Routine:

# Weekday schedule: [Detailed hour-by-hour breakdown]
# Weekend activities: [Typical leisure time pursuits]
# Eating habits: [Dietary preferences, favorite foods, meal routines]
# Exercise regimen: [Any regular physical activities]
# Sleep patterns: [Typical sleep schedule and any sleep-related issues]

# Recent Experiences:

# [3-4 notable events from the past month that are on their mind]
# Current projects: [Any ongoing personal or professional endeavors]
# Latest accomplishment: [A recent achievement they're proud of]
# Recent challenge: [A difficult situation they've faced lately]

# Financial Situation:

# Income level: [General range and sources of income]
# Spending habits: [How they manage their money]
# Savings and investments: [Financial goals and strategies]
# Debts: [Any significant financial obligations]

# Health and Wellness:

# Overall health status: [Any chronic conditions or health concerns]
# Mental health: [Any diagnosed conditions or ongoing therapy]
# Self-care practices: [How they maintain their well-being]
# Relationship with healthcare: [Frequency of check-ups, attitudes toward medicine]

# Cultural Touchstones:

# Favorite media: [Books, movies, TV shows, music they enjoy]
# Influential figures: [People they admire or who have impacted their life]
# Cultural identity: [How they relate to their heritage and current cultural environment]

# Environment and Possessions:

# Living space: [Description of home interior and any prized possessions]
# Transportation: [How they typically get around]
# Technology: [Devices they use and their comfort level with tech]
# Pets: [Any animal companions and their significance]

# Break down the summary as follows:

# a) First third of life: Provide a summary of key events and developments, covering approximately one year at a time.

# b) Second third of life: Provide a more detailed account, with events and developments summarized on a monthly basis.

# c) Last third of life: 
#    - For the first two-thirds of this period: Provide a daily summary of events and activities.
#    - For the final third of this period: Provide an hourly breakdown of the person's activities and experiences.

# 4. Present your output in the following format:

# <complete_history>
# [Insert the complete generated history based on the backstory template here]
# </complete_history>

# <age_based_summary>
# <first_third>
# [Insert yearly summary of the first third of the person's life]
# </first_third>

# <second_third>
# [Insert monthly summary of the second third of the person's life]
# </second_third>

# <last_third>
# <daily_summary>
# [Insert daily summary for the first two-thirds of the last third of the person's life]
# </daily_summary>

# <hourly_breakdown>
# [Insert hourly breakdown for the final third of the last third of the person's life]
# </hourly_breakdown>
# </last_third>
# </age_based_summary>

# Ensure that your generated history and summaries are consistent with the information provided in the backstory template and maintain a logical progression of events throughout the person's life.