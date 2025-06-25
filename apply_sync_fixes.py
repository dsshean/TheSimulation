#!/usr/bin/env python3
"""
Script to apply synchronization fixes to TheSimulation codebase.
This script creates a backup and applies the necessary changes to fix
multi-agent state synchronization issues.

Usage:
    python apply_sync_fixes.py [--backup-only] [--dry-run]
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def create_backup(source_dir: str, backup_name: str = None):
    """Create a backup of the simulation source code."""
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"simulation_backup_{timestamp}"
    
    backup_path = Path(source_dir).parent / backup_name
    
    print(f"Creating backup at: {backup_path}")
    
    # Copy the src directory
    shutil.copytree(source_dir, backup_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
    print(f"Backup created successfully: {backup_path}")
    return backup_path


def apply_imports_patch(simulation_async_path: str, dry_run: bool = False):
    """Apply import changes to simulation_async.py"""
    print("Applying imports patch to simulation_async.py...")
    
    with open(simulation_async_path, 'r') as f:
        content = f.read()
    
    # Find the import section and add new imports
    import_additions = """
# Synchronization components
from .state_manager import StateManager, CircuitBreaker
from .queue_manager import SequencedEventBus, EventProcessor, QueueHealthMonitor
from .perception_manager import SynchronizedPerceptionManager"""
    
    # Insert after the existing imports
    if "from .perception_manager import PerceptionManager" in content:
        content = content.replace(
            "from .perception_manager import PerceptionManager # Import the moved PerceptionManager",
            "from .perception_manager import PerceptionManager # Import the moved PerceptionManager" + import_additions
        )
    
    if not dry_run:
        with open(simulation_async_path, 'w') as f:
            f.write(content)
        print("✓ Imports patch applied")
    else:
        print("✓ Imports patch ready (dry run)")


def apply_initialization_patch(simulation_async_path: str, dry_run: bool = False):
    """Apply initialization changes to simulation_async.py"""
    print("Applying initialization patch...")
    
    with open(simulation_async_path, 'r') as f:
        content = f.read()
    
    # Add global variables after existing globals
    globals_addition = """
# Synchronized components
state_manager_global: Optional[StateManager] = None
circuit_breaker_global: Optional[CircuitBreaker] = None
event_processor_global: Optional[EventProcessor] = None
queue_health_monitor_global: Optional[QueueHealthMonitor] = None"""
    
    if "live_display_object: Optional[Live] = None" in content:
        content = content.replace(
            "live_display_object: Optional[Live] = None",
            "live_display_object: Optional[Live] = None" + globals_addition
        )
    
    # Add initialization function
    init_function = '''
async def initialize_synchronized_components(logger_instance):
    """Initialize synchronized components for better multi-agent handling."""
    global event_bus, narration_queue, state_manager_global, circuit_breaker_global
    global event_processor_global, queue_health_monitor_global, perception_manager_global
    
    logger_instance.info("[SyncInit] Initializing synchronized components...")
    
    # Replace simple queues with sequenced event buses
    old_event_bus = event_bus
    old_narration_queue = narration_queue
    
    event_bus = SequencedEventBus("EventBus", maxsize=100)
    narration_queue = SequencedEventBus("NarrationQueue", maxsize=50)
    
    # Initialize state manager
    state_manager_global = StateManager(state, logger_instance)
    
    # Initialize circuit breaker
    circuit_breaker_global = CircuitBreaker(max_repetitions=3, window_size=5)
    
    # Initialize event processor
    event_processor_global = EventProcessor(logger_instance)
    
    # Initialize queue health monitor
    queue_health_monitor_global = QueueHealthMonitor(check_interval=10.0)
    queue_health_monitor_global.register_queue("event_bus", event_bus)
    queue_health_monitor_global.register_queue("narration_queue", narration_queue)
    
    # Update perception manager to use synchronized version
    perception_manager_global = SynchronizedPerceptionManager(state, state_manager_global)
    
    logger_instance.info("[SyncInit] Synchronized components initialized")
    
    # Start queue monitoring
    asyncio.create_task(queue_health_monitor_global.start_monitoring())
'''
    
    # Insert before the run_simulation function
    if "async def run_simulation(" in content:
        content = content.replace(
            "async def run_simulation(",
            init_function + "\nasync def run_simulation("
        )
    
    if not dry_run:
        with open(simulation_async_path, 'w') as f:
            f.write(content)
        print("✓ Initialization patch applied")
    else:
        print("✓ Initialization patch ready (dry run)")


def apply_queue_patches(simulation_async_path: str, dry_run: bool = False):
    """Replace queue operations with synchronized versions."""
    print("Applying queue operation patches...")
    
    with open(simulation_async_path, 'r') as f:
        content = f.read()
    
    # Replace safe_queue_get calls
    content = content.replace(
        "await safe_queue_get(event_bus",
        "await event_bus.get_event("
    )
    content = content.replace(
        "await safe_queue_get(narration_queue",
        "await narration_queue.get_event("
    )
    
    # Replace safe_queue_put calls
    content = content.replace(
        "await safe_queue_put(event_bus",
        "await event_bus.put_event("
    )
    content = content.replace(
        "await safe_queue_put(narration_queue",
        "await narration_queue.put_event("
    )
    
    # Replace qsize calls
    content = content.replace(
        "event_bus.qsize()",
        "event_bus.qsize()"
    )
    content = content.replace(
        "narration_queue.qsize()",
        "narration_queue.qsize()"
    )
    
    if not dry_run:
        with open(simulation_async_path, 'w') as f:
            f.write(content)
        print("✓ Queue patches applied")
    else:
        print("✓ Queue patches ready (dry run)")


def add_sync_initialization_call(simulation_async_path: str, dry_run: bool = False):
    """Add call to initialize synchronized components in run_simulation."""
    print("Adding synchronization initialization call...")
    
    with open(simulation_async_path, 'r') as f:
        content = f.read()
    
    # Find where to add the initialization call (after state loading)
    init_call = "\n    # Initialize synchronized components\n    await initialize_synchronized_components(logger)\n"
    
    # Look for a good insertion point after state initialization
    if "perception_manager_global = PerceptionManager(state)" in content:
        content = content.replace(
            "perception_manager_global = PerceptionManager(state)",
            "# perception_manager_global will be initialized in initialize_synchronized_components()" + init_call
        )
    
    if not dry_run:
        with open(simulation_async_path, 'w') as f:
            f.write(content)
        print("✓ Synchronization initialization call added")
    else:
        print("✓ Synchronization initialization call ready (dry run)")


def main():
    parser = argparse.ArgumentParser(description="Apply synchronization fixes to TheSimulation")
    parser.add_argument("--backup-only", action="store_true", help="Only create backup, don't apply changes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying")
    parser.add_argument("--src-dir", default="src", help="Source directory path (default: src)")
    
    args = parser.parse_args()
    
    # Determine source directory
    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        sys.exit(1)
    
    simulation_async_path = src_dir / "simulation_async.py"
    if not simulation_async_path.exists():
        print(f"Error: {simulation_async_path} not found")
        sys.exit(1)
    
    print("TheSimulation Synchronization Fixes")
    print("=" * 40)
    
    # Create backup
    backup_path = create_backup(str(src_dir))
    print(f"Backup created at: {backup_path}")
    
    if args.backup_only:
        print("Backup complete. Exiting without applying changes.")
        return
    
    print("\nApplying synchronization fixes...")
    
    # Check that our new files exist
    required_files = ["state_manager.py", "queue_manager.py"]
    missing_files = []
    
    for file in required_files:
        if not (src_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Make sure state_manager.py and queue_manager.py are in the src directory")
        sys.exit(1)
    
    try:
        # Apply patches
        apply_imports_patch(str(simulation_async_path), args.dry_run)
        apply_initialization_patch(str(simulation_async_path), args.dry_run)
        apply_queue_patches(str(simulation_async_path), args.dry_run)
        add_sync_initialization_call(str(simulation_async_path), args.dry_run)
        
        if args.dry_run:
            print("\n✓ Dry run complete - no files were modified")
            print("Run without --dry-run to apply changes")
        else:
            print("\n✓ All synchronization fixes applied successfully!")
            print(f"Backup available at: {backup_path}")
            print("\nNext steps:")
            print("1. Test the simulation with multiple agents")
            print("2. Monitor the logs for improved synchronization")
            print("3. Check that agents can see each other properly")
    
    except Exception as e:
        print(f"\n✗ Error applying fixes: {e}")
        print(f"Your backup is safe at: {backup_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()