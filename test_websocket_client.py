#!/usr/bin/env python3
"""
Simple WebSocket client test to verify visualization data is flowing correctly
"""

import asyncio
import json
import websockets
import time

async def test_websocket_connection():
    uri = "ws://localhost:8766"
    
    try:
        print("Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Listen for messages for a few seconds
            message_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 10 and message_count < 5:
                try:
                    # Wait for a message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    message_count += 1
                    
                    # Parse and display the message
                    try:
                        data = json.loads(message)
                        print(f"\n--- Message {message_count} ---")
                        print(f"Type: {data.get('type', 'unknown')}")
                        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
                        print(f"World Time: {data.get('world_time', 'N/A')}")
                        
                        simulacra = data.get('simulacra', {})
                        locations = data.get('locations', {})
                        narrative = data.get('recent_narrative', [])
                        
                        print(f"Simulacra: {len(simulacra)} agents")
                        for agent_id, agent in simulacra.items():
                            print(f"  • {agent.get('name', agent_id)}: {agent.get('status', 'unknown')} at {agent.get('current_location', 'unknown')}")
                        
                        print(f"Locations: {len(locations)} total")
                        for loc_id, location in list(locations.items())[:3]:  # Show first 3
                            print(f"  • {location.get('name', loc_id)}: {location.get('object_count', 0)} objects")
                        
                        print(f"Recent Narrative: {len(narrative)} entries")
                        if narrative:
                            print(f"  Latest: {narrative[-1][:100]}...")
                            
                    except json.JSONDecodeError:
                        print(f"Non-JSON message: {message[:100]}...")
                        
                except asyncio.TimeoutError:
                    print(".", end="", flush=True)  # Show we're waiting
                    
            if message_count == 0:
                print("\n❌ No messages received from WebSocket server")
                return False
            else:
                print(f"\n✅ Successfully received {message_count} messages from WebSocket server")
                return True
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing WebSocket Visualization Connection")
    print("=" * 50)
    
    result = asyncio.run(test_websocket_connection())
    
    print("\n" + "=" * 50)
    if result:
        print("🎉 WebSocket test PASSED! Visualization data is flowing correctly.")
    else:
        print("💥 WebSocket test FAILED! Check the simulation and WebSocket server.")