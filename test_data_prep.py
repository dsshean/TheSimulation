#!/usr/bin/env python3
"""
Test the visualization data preparation function independently
"""

import json
import time
from typing import Dict, Any

def _prepare_visualization_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare simulation state data for web visualization"""
    
    # Extract simulacra data
    simulacra_data = {}
    simulacra_profiles = state.get('simulacra_profiles', {})
    for agent_id, agent_data in simulacra_profiles.items():
        simulacra_data[agent_id] = {
            'id': agent_id,
            'name': agent_data.get('persona_details', {}).get('Name', agent_id),
            'current_location': agent_data.get('current_location'),
            'status': agent_data.get('status', 'unknown'),
            'current_action': agent_data.get('current_action_description', 'No current action'),
            'last_observation': agent_data.get('last_observation', 'No recent observations'),
            'goal': agent_data.get('goal', 'No current goal')[:100] + '...' if len(agent_data.get('goal', '')) > 100 else agent_data.get('goal', 'No current goal'),
            'age': agent_data.get('persona_details', {}).get('Age'),
            'occupation': agent_data.get('persona_details', {}).get('Occupation')
        }
    
    # Extract location data
    locations_data = {}
    location_details = state.get('current_world_state', {}).get('location_details', {})
    for loc_id, loc_data in location_details.items():
        # Count objects and NPCs
        ephemeral_objects = loc_data.get('ephemeral_objects', [])
        ephemeral_npcs = loc_data.get('ephemeral_npcs', [])
        connected_locations = loc_data.get('connected_locations', [])
        
        locations_data[loc_id] = {
            'id': loc_id,
            'name': loc_data.get('name', loc_id),
            'description': loc_data.get('description', 'No description'),
            'object_count': len(ephemeral_objects),
            'npc_count': len(ephemeral_npcs),
            'connections': [conn.get('to_location_id_hint') for conn in connected_locations if isinstance(conn, dict)],
            'objects': [{'id': obj.get('id'), 'name': obj.get('name'), 'description': obj.get('description')} 
                       for obj in ephemeral_objects if isinstance(obj, dict)][:10]  # Limit for performance
        }
    
    # Extract global objects
    objects_data = {}
    global_objects = state.get('objects', {})
    
    # Handle both dictionary and list formats
    if isinstance(global_objects, dict):
        for obj_id, obj_data in global_objects.items():
            objects_data[obj_id] = {
                'id': obj_id,
                'properties': obj_data.get('properties', {}),
                'name': obj_data.get('name', obj_id)
            }
    elif isinstance(global_objects, list):
        for obj_data in global_objects:
            if isinstance(obj_data, dict) and 'id' in obj_data:
                obj_id = obj_data['id']
                objects_data[obj_id] = {
                    'id': obj_id,
                    'properties': obj_data.get('properties', {}),
                    'name': obj_data.get('name', obj_id)
                }
    
    # Extract narrative timeline (last 10 entries)
    narrative_log = state.get('narrative_log', [])
    recent_narrative = narrative_log[-10:] if len(narrative_log) > 10 else narrative_log
    
    return {
        'type': 'simulation_state',
        'timestamp': time.time(),
        'world_time': state.get('world_time', 0),
        'simulacra': simulacra_data,
        'locations': locations_data, 
        'objects': objects_data,
        'recent_narrative': recent_narrative,
        'world_feeds': state.get('world_feeds', {}),
        'active_simulacra_count': len(simulacra_data),
        'total_locations': len(locations_data),
        'total_objects': len(objects_data)
    }

def main():
    print("Testing visualization data preparation...")
    
    # Load the actual state file
    try:
        with open('data/states/simulation_state_cce7f369-f7dc-4e11-8136-5a142fa5d787.json', 'r') as f:
            state = json.load(f)
        print("✓ Loaded simulation state successfully")
    except Exception as e:
        print(f"✗ Failed to load state: {e}")
        return
    
    # Test data preparation
    try:
        viz_data = _prepare_visualization_data(state)
        print("✓ Data preparation successful")
        
        print(f"\nPrepared visualization data:")
        print(f"  Type: {viz_data.get('type')}")
        print(f"  World Time: {viz_data.get('world_time'):.1f}s")
        print(f"  Simulacra Count: {len(viz_data.get('simulacra', {}))}")
        print(f"  Locations Count: {len(viz_data.get('locations', {}))}")
        print(f"  Objects Count: {len(viz_data.get('objects', {}))}")
        print(f"  Recent Narrative: {len(viz_data.get('recent_narrative', []))} entries")
        
        print(f"\nSimulacra Details:")
        for agent_id, agent in viz_data.get('simulacra', {}).items():
            print(f"  • {agent.get('name')} ({agent_id})")
            print(f"    Status: {agent.get('status')}")
            print(f"    Location: {agent.get('current_location')}")
            print(f"    Action: {agent.get('current_action')}")
            
        print(f"\nLocation Details:")
        for loc_id, location in viz_data.get('locations', {}).items():
            print(f"  • {location.get('name')} ({loc_id})")
            print(f"    Objects: {location.get('object_count')}, NPCs: {location.get('npc_count')}")
            print(f"    Connections: {location.get('connections')}")
            
        # Test JSON serialization
        json_str = json.dumps(viz_data)
        print(f"\n✓ JSON serialization successful ({len(json_str)} characters)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed! The visualization data preparation is working correctly.")
    else:
        print("\n❌ Tests failed! Check the errors above.")