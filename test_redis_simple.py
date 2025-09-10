#!/usr/bin/env python3
"""
Simple Redis connection test with ASCII output only.
"""

import asyncio
import redis.asyncio as redis

async def test_simple_redis():
    print("Testing Redis connection...")
    try:
        client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        result = await client.ping()
        if result:
            print("SUCCESS: Redis connection works!")
            await client.close()
            return True
        else:
            print("FAILED: Redis ping returned False")
            return False
    except Exception as e:
        print(f"ERROR: Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_redis())
    if success:
        print("\nRedis is ready for TheSimulation!")
    else:
        print("\nRedis is NOT working. Please check Redis server.")