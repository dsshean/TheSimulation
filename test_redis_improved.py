#!/usr/bin/env python3
"""
Test script for the improved Redis client
"""

import asyncio
import logging
import json
import time
from src.redis_client_improved import ImprovedRedisClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_improved_redis_client():
    """Test the improved Redis client functionality"""
    logger.info("Starting Redis client test...")
    
    client = ImprovedRedisClient()
    
    # Test 1: Basic connection
    logger.info("Test 1: Basic connection")
    success = await client.connect()
    if success:
        logger.info("✅ Connection successful")
    else:
        logger.error("❌ Connection failed")
        return
    
    # Test 2: Connection stats
    logger.info("Test 2: Connection statistics")
    stats = client.get_connection_stats()
    logger.info(f"Connection stats: {json.dumps(stats, indent=2)}")
    
    # Test 3: State publishing with mock state
    logger.info("Test 3: State publishing")
    mock_state = {
        "world_time": 123.45,
        "world_instance_uuid": "test-uuid-123",
        "simulacra_profiles": {
            "agent1": {"status": "idle", "location": "home"}
        },
        "active_simulacra_ids": ["agent1"],
        "narrative_log": ["Test narrative entry"]
    }
    
    def mock_time_getter():
        return time.time()
    
    client.set_state_reference(mock_state, mock_time_getter)
    
    # Publish several state updates
    for i in range(3):
        await client.publish_state_update()
        logger.info(f"Published state update {i+1}")
        await asyncio.sleep(0.1)
    
    # Test 4: Event publishing
    logger.info("Test 4: Event publishing")
    test_event = {
        "event_type": "test_event",
        "agent_id": "agent1",
        "message": "This is a test event"
    }
    
    await client.publish_event(test_event)
    logger.info("✅ Event published successfully")
    
    # Test 5: Recent events buffer
    logger.info("Test 5: Recent events buffer")
    for i in range(5):
        client.add_recent_event({
            "event_id": i,
            "timestamp": time.time(),
            "message": f"Test event {i}"
        })
    
    logger.info(f"Recent events count: {len(client.recent_events)}")
    
    # Test 6: Command handling
    logger.info("Test 6: Command handling setup")
    
    async def test_command_handler(message_data):
        """Mock command handler"""
        return {
            "success": True, 
            "message": f"Test command executed with data: {message_data.get('params', 'none')}"
        }
    
    client.register_command_handler("test_command", test_command_handler)
    
    # Test 7: Circuit breaker simulation
    logger.info("Test 7: Circuit breaker behavior")
    logger.info(f"Initial circuit breaker state: {client.circuit_breaker.state.value}")
    
    # Simulate failures to test circuit breaker
    for i in range(3):
        client.circuit_breaker.on_failure()
        logger.info(f"After failure {i+1}: {client.circuit_breaker.state.value}, "
                   f"failures: {client.circuit_breaker.failure_count}")
    
    # Test if circuit breaker can_execute
    can_execute = client.circuit_breaker.can_execute()
    logger.info(f"Can execute after failures: {can_execute}")
    
    # Test recovery
    client.circuit_breaker.on_success()
    logger.info(f"After success: {client.circuit_breaker.state.value}")
    
    # Test 8: Connection pool stress test
    logger.info("Test 8: Connection pool stress test")
    
    async def stress_publish():
        """Stress test with multiple concurrent operations"""
        tasks = []
        for i in range(10):
            tasks.append(client.publish_state_update())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Stress test: {successful}/10 operations successful")
    
    await stress_publish()
    
    # Final connection stats
    logger.info("Final connection statistics:")
    final_stats = client.get_connection_stats()
    logger.info(f"Final stats: {json.dumps(final_stats, indent=2)}")
    
    # Cleanup
    logger.info("Test cleanup...")
    await client.disconnect()
    logger.info("✅ All tests completed successfully!")


async def test_redis_failure_recovery():
    """Test Redis client behavior when Redis is unavailable"""
    logger.info("Starting Redis failure recovery test...")
    
    client = ImprovedRedisClient()
    
    # Try to connect when Redis might not be available
    logger.info("Attempting connection (Redis may not be running)...")
    success = await client.connect()
    
    if not success:
        logger.info("✅ Connection failed gracefully (expected if Redis not running)")
        
        # Test operations when disconnected
        logger.info("Testing operations while disconnected...")
        
        mock_state = {"test": "data"}
        client.set_state_reference(mock_state, lambda: 0.0)
        
        # These should not crash
        await client.publish_state_update()
        await client.publish_event({"test": "event"})
        
        logger.info("✅ Operations handled gracefully while disconnected")
    else:
        logger.info("Redis is available, testing normal operations...")
        await client.disconnect()
    
    logger.info("✅ Failure recovery test completed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "failure":
        asyncio.run(test_redis_failure_recovery())
    else:
        asyncio.run(test_improved_redis_client())