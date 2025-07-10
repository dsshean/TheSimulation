#!/usr/bin/env python3
"""
Quick test script to verify Redis integration is working properly.
"""

import asyncio
import json
import sys
import time
from src.redis_client import redis_client
from src.redis_commands import create_command_handlers

async def test_redis_integration():
    print("🧪 Testing Redis Integration for TheSimulation")
    print("=" * 50)
    
    # Test Redis connection
    print("1. Testing Redis connection...")
    connected = await redis_client.connect()
    if not connected:
        print("❌ Failed to connect to Redis")
        return False
    print("✅ Connected to Redis successfully")
    
    # Test state publishing
    print("\n2. Testing state publishing...")
    redis_client.set_state_reference(
        {"test": "data", "world_time": 123.45}, 
        lambda: 123.45
    )
    await redis_client.publish_state_update()
    print("✅ Published test state update")
    
    # Test event publishing
    print("\n3. Testing event publishing...")
    await redis_client.publish_event({
        "event_type": "test_event",
        "data": "test_data"
    })
    print("✅ Published test event")
    
    # Test command handlers
    print("\n4. Testing command handlers...")
    handlers = create_command_handlers(
        {"simulacra_profiles": {"test_agent": {"persona_details": {"Name": "TestAgent"}}}},
        None,  # No narration queue for test
        lambda: 123.45
    )
    
    # Test inject_narrative handler
    result = await handlers["inject_narrative"]({"text": "Test narrative"})
    if result["success"]:
        print("✅ inject_narrative handler works")
    else:
        print(f"❌ inject_narrative failed: {result['message']}")
    
    # Cleanup
    await redis_client.disconnect()
    print("\n✅ All Redis integration tests passed!")
    print("\n🚀 Ready to run the full simulation with Redis!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_redis_integration())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)