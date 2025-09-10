#!/usr/bin/env python3
"""
Monitor Redis channels to see what data is being published.
"""

import asyncio
import json
import redis.asyncio as redis

async def monitor_redis():
    print("Connecting to Redis...")
    try:
        client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        
        # Test connection
        await client.ping()
        print("Connected to Redis successfully")
        
        # Subscribe to all simulation channels
        pubsub = client.pubsub()
        await pubsub.subscribe("simulation:state", "simulation:commands", "simulation:events")
        print("Subscribed to simulation channels")
        print("Listening for messages... (Ctrl+C to stop)")
        print("-" * 50)
        
        while True:
            try:
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    print(f"[{channel}] Received message:")
                    
                    # Try to parse as JSON for better formatting
                    try:
                        parsed_data = json.loads(data)
                        if channel == "simulation:state":
                            # Show key fields only for state updates
                            print(f"  World Time: {parsed_data.get('world_time', 'N/A')}")
                            print(f"  Active Simulacra: {len(parsed_data.get('active_simulacra_ids', []))}")
                            print(f"  World UUID: {parsed_data.get('world_instance_uuid', 'N/A')}")
                            if parsed_data.get('recent_events'):
                                print(f"  Recent Events: {len(parsed_data['recent_events'])}")
                        else:
                            print(f"  Data: {json.dumps(parsed_data, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"  Raw Data: {data}")
                    
                    print("-" * 50)
                    
            except asyncio.TimeoutError:
                continue  # Normal timeout, keep listening
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'client' in locals():
            await client.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(monitor_redis())
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")