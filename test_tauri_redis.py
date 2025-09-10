#!/usr/bin/env python3
"""
Test script to publish sample data to Redis to verify Tauri UI can receive it.
"""

import asyncio
import json
import redis.asyncio as redis
import time

async def publish_test_data():
    print("Publishing test simulation data to Redis...")
    
    try:
        client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        await client.ping()
        print("Connected to Redis")
        
        # Create sample simulation state data
        test_state = {
            "world_time": 342.5,
            "world_instance_uuid": "test-ui-connection",
            "active_simulacra_ids": ["test_agent"],
            "simulacra_profiles": {
                "test_agent": {
                    "persona_details": {
                        "Name": "Test Agent",
                        "Occupation": "UI Tester"
                    },
                    "current_location": "Test_Location",
                    "status": "testing",
                    "last_observation": "Testing UI connection"
                }
            },
            "current_world_state": {
                "location_details": {
                    "Test_Location": {
                        "name": "Test Location",
                        "description": "A test location for UI verification"
                    }
                }
            },
            "world_feeds": {
                "weather": {
                    "condition": "sunny",
                    "temperature_celsius": 25
                }
            },
            "narrative_log": [
                "[T342.5] This is a test narrative entry to verify UI display.",
                "[T340.0] Testing the narrative log display functionality.",
                "[T335.0] Checking if the Tauri UI can receive Redis data."
            ],
            "recent_events": [
                {
                    "event_type": "test_event",
                    "agent_type": "test",
                    "agent_id": "test_agent",
                    "sim_time_s": 342.5,
                    "data": {
                        "content": "This is a test event for UI verification"
                    }
                }
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        }
        
        # Publish to Redis
        message = json.dumps(test_state)
        await client.publish("simulation:state", message)
        print("Published test state to simulation:state channel")
        
        # Keep publishing every 2 seconds for 30 seconds
        for i in range(15):
            test_state["world_time"] += 2.0
            test_state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            message = json.dumps(test_state)
            await client.publish("simulation:state", message)
            print(f"Published update #{i+1} (world_time: {test_state['world_time']})")
            await asyncio.sleep(2)
            
        await client.aclose()
        print("Test completed - check Tauri UI for data")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(publish_test_data())