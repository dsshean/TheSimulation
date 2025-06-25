#!/usr/bin/env python3
"""
Script to add Pygame visualization integration to TheSimulation.

This script modifies the main simulation code to include real-time visualization
when ENABLE_VISUALIZATION is set to true in the environment.

Usage:
    python add_visualization.py [--dry-run] [--backup]
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


def create_backup(file_path: str):
    """Create a backup of a file."""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Backup created: {backup_path}")
    return backup_path


def add_visualization_imports(content: str) -> str:
    """Add visualization imports to simulation_async.py"""
    import_section = """
# Visualization integration
from .visualization_integration import (
    initialize_visualization, add_visualization_task, cleanup_visualization,
    load_visualization_config_from_env, SimulationVisualizationBridge
)"""
    
    # Insert after existing imports
    if "from .state_loader import parse_location_string" in content:
        content = content.replace(
            "from .state_loader import parse_location_string  # Used in run_simulation",
            "from .state_loader import parse_location_string  # Used in run_simulation" + import_section
        )
    
    return content


def add_visualization_globals(content: str) -> str:
    """Add visualization global variables."""
    globals_addition = """
# Visualization components
visualization_bridge_global: Optional[SimulationVisualizationBridge] = None"""
    
    if "live_display_object: Optional[Live] = None" in content:
        content = content.replace(
            "live_display_object: Optional[Live] = None",
            "live_display_object: Optional[Live] = None" + globals_addition
        )
    
    return content


def add_visualization_initialization(content: str) -> str:
    """Add visualization initialization to run_simulation function."""
    init_code = """
    # Initialize visualization if enabled
    global visualization_bridge_global
    try:
        visualization_config = load_visualization_config_from_env()
        visualization_bridge_global = initialize_visualization(state, logger, visualization_config)
        if visualization_bridge_global:
            logger.info("[Main] Pygame visualization initialized")
    except Exception as e:
        logger.error(f"[Main] Failed to initialize visualization: {e}")
        visualization_bridge_global = None"""
    
    # Insert after state loading and before main task setup
    if "perception_manager_global = PerceptionManager(state)" in content:
        content = content.replace(
            "perception_manager_global = PerceptionManager(state)",
            "perception_manager_global = PerceptionManager(state)" + init_code
        )
    
    return content


def add_visualization_task(content: str) -> str:
    """Add visualization task to the task list."""
    task_code = """
        # Add visualization update task if enabled
        if visualization_bridge_global and visualization_bridge_global.is_running():
            await add_visualization_task(tasks, state, visualization_bridge_global, logger)"""
    
    # Insert after other task creation
    if "narrative_image_task = asyncio.create_task(" in content:
        insertion_point = content.find("tasks.append(narrative_image_task)")
        if insertion_point != -1:
            # Find the end of this line
            line_end = content.find('\n', insertion_point)
            if line_end != -1:
                content = content[:line_end] + task_code + content[line_end:]
    
    return content


def add_visualization_cleanup(content: str) -> str:
    """Add visualization cleanup to the finally block."""
    cleanup_code = """
        # Cleanup visualization
        cleanup_visualization(visualization_bridge_global, logger)"""
    
    # Find the finally block and add cleanup
    if "finally:" in content:
        # Find the finally block
        finally_pos = content.find("finally:")
        if finally_pos != -1:
            # Find the end of the finally line
            line_end = content.find('\n', finally_pos)
            if line_end != -1:
                # Insert after any existing cleanup
                content = content[:line_end] + cleanup_code + content[line_end:]
    
    return content


def apply_visualization_integration(file_path: str, dry_run: bool = False) -> bool:
    """Apply all visualization integration changes."""
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all modifications
        content = add_visualization_imports(content)
        content = add_visualization_globals(content)
        content = add_visualization_initialization(content)
        content = add_visualization_task(content)
        content = add_visualization_cleanup(content)
        
        if content == original_content:
            print("No changes needed - visualization integration may already be present")
            return True
        
        if not dry_run:
            with open(file_path, 'w') as f:
                f.write(content)
            print("✓ Visualization integration applied successfully")
        else:
            print("✓ Visualization integration ready (dry run)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error applying visualization integration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Add Pygame visualization to TheSimulation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying")
    parser.add_argument("--backup", action="store_true", help="Create backup before modifying")
    parser.add_argument("--src-dir", default="src", help="Source directory path (default: src)")
    
    args = parser.parse_args()
    
    # Check source directory
    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        return 1
    
    simulation_async_path = src_dir / "simulation_async.py"
    if not simulation_async_path.exists():
        print(f"Error: {simulation_async_path} not found")
        return 1
    
    # Check that visualization files exist
    required_files = [
        "pygame_visualizer.py",
        "visualization_integration.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (src_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required visualization files: {missing_files}")
        print("Make sure pygame_visualizer.py and visualization_integration.py are in the src directory")
        return 1
    
    print("TheSimulation Pygame Visualization Integration")
    print("=" * 50)
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        create_backup(str(simulation_async_path))
    
    # Apply integration
    success = apply_visualization_integration(str(simulation_async_path), args.dry_run)
    
    if success:
        if args.dry_run:
            print("\n✓ Dry run complete - no files were modified")
            print("Run without --dry-run to apply changes")
        else:
            print("\n✓ Pygame visualization integration applied successfully!")
            print("\nTo enable visualization:")
            print("1. Add ENABLE_VISUALIZATION=true to your .env file")
            print("2. Install pygame: pip install pygame")
            print("3. Run the simulation normally: python main_async.py")
            print("4. Or run the standalone visualizer: python visualizer.py")
            
            print("\nVisualization features:")
            print("• Real-time location and simulacra display")
            print("• Zoom and pan with mouse")
            print("• Click locations to see details")
            print("• Connection lines between locations")
            print("• Color-coded simulacra status")
    else:
        print(f"\n✗ Failed to apply visualization integration")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())