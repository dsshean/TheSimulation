#!/usr/bin/env python3
"""
Test script to verify that the visualizer ports are working correctly.
This tests the WebSocket connection without needing the full simulation.
"""

import asyncio
import json
import websockets
import time

async def test_visualization_websocket():
    """Test the visualization WebSocket server on port 8766"""
    print("Testing Visualization WebSocket Server...")
    
    # Sample simulation data
    test_data = {
        'type': 'simulation_state',
        'timestamp': time.time(),
        'world_time': 123.45,
        'simulacra': {
            'test_agent': {
                'id': 'test_agent',
                'name': 'Test Agent',
                'current_location': 'test_room',
                'status': 'idle',
                'current_action': 'Testing the system',
                'last_observation': 'Everything looks good!',
                'goal': 'Test the visualization',
                'age': 30,
                'occupation': 'Test Engineer'
            }
        },
        'locations': {
            'test_room': {
                'id': 'test_room',
                'name': 'Test Room',
                'description': 'A room for testing',
                'object_count': 3,
                'npc_count': 0,
                'connections': ['other_room'],
                'objects': [
                    {'id': 'test_obj1', 'name': 'Test Object', 'description': 'A test object'}
                ]
            },
            'other_room': {
                'id': 'other_room',
                'name': 'Other Room',
                'description': 'Another test room',
                'object_count': 1,
                'npc_count': 0,
                'connections': ['test_room'],
                'objects': []
            }
        },
        'objects': {},
        'recent_narrative': [
            '[T123] The test agent is working properly in the test environment.',
            '[T122] Everything initialized successfully.',
            '[T121] System test commenced.'
        ],
        'world_feeds': {
            'weather': {
                'condition': 'clear',
                'temperature_celsius': 22
            }
        },
        'active_simulacra_count': 1,
        'total_locations': 2,
        'total_objects': 1
    }
    
    connected_clients = set()
    
    async def handle_client(websocket, path):
        """Handle test client connections"""
        connected_clients.add(websocket)
        print(f"✓ Test client connected from {websocket.remote_address}")
        
        try:
            # Send initial test data
            await websocket.send(json.dumps(test_data))
            print("✓ Sent initial test data to client")
            
            # Keep connection alive for a few seconds
            for i in range(3):
                await asyncio.sleep(1)
                test_data['world_time'] += 1
                await websocket.send(json.dumps(test_data))
                print(f"✓ Sent update {i+1}/3")
            
        except websockets.exceptions.ConnectionClosed:
            print("✓ Test client disconnected")
        finally:
            connected_clients.discard(websocket)
    
    try:
        print("Starting test WebSocket server on port 8766...")
        server = await websockets.serve(handle_client, "localhost", 8766)
        print("✓ WebSocket server started successfully")
        
        # Keep server running for a few seconds
        print("Server running... (will stop automatically)")
        await asyncio.sleep(5)
        
        server.close()
        await server.wait_closed()
        print("✓ Server stopped successfully")
        return True
        
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print("✗ ERROR: Port 8766 is already in use!")
            print("  This means either:")
            print("  1. The simulation is already running (good!)")
            print("  2. Another process is using this port")
            return False
        else:
            print(f"✗ ERROR: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    print("TheSimulation Visualizer Port Test")
    print("=" * 40)
    print()
    
    try:
        result = asyncio.run(test_visualization_websocket())
        
        print()
        print("=" * 40)
        if result:
            print("✓ ALL TESTS PASSED!")
            print()
            print("The visualization system should work correctly.")
            print("To use it:")
            print("1. Start the simulation: python main_async.py")
            print("2. Start the visualizer: python start_visualizer.py")
            print("3. Open browser at: http://localhost:8080")
        else:
            print("✗ TESTS FAILED!")
            print()
            print("Port 8766 is not available. Check for conflicts.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")

if __name__ == "__main__":
    main()