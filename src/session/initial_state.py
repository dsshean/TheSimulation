# src/session/initial_state.py

# Define DEFAULT initial state
# This might be overridden in main.py if generation occurs
default_initial_sim_state = {
    "world_time": "Day 1, 09:00",
    "simulacra_location": "Home",
    "simulacra_goal": "Explore the nearby Market square.", # Default goal
    "simulacra_status": {
        "Name": "Default Simulacra", # Default Name
        "Age": 25, # Default Age
        "Occupation": "Adventurer", # Default Occupation
        "Personality_Traits": ["Curious", "Brave"], # Default Traits
        "inventory": ["map", "coin pouch (10 coins)"],
        "mood": "neutral"
        },
    "npc_locations": {"Librarian": "Library", "Merchant Bob": "Market"},
    "npc_states": {
        "Librarian": {"persona": "knowledgeable but slightly aloof", "activity": "shelving books"},
        "Merchant Bob": {"persona": "boisterous and eager to trade", "activity": "arranging apples"}
    },
    "location_details": {
        "Home": "Your small but comfortable room. Sunlight streams through the window.",
        "Market": "A lively square filled with stalls selling food, crafts, and trinkets. The smell of baked bread hangs in the air. Merchant Bob is here.",
        "Library": "A quiet, imposing building made of stone, smelling faintly of old paper. The Librarian is inside."
    },
    "last_narration": "The simulation begins.",
    "last_simulacra_action": None,
    "last_world_engine_update": None,
    "last_npc_interaction": None,
    "last_state_summary": None,
    "last_setting_details": None,
    "last_goal_update": None,
    "last_simulacra_status_check": None,
    "generated_background": None, # Optional key to store full generated data
}