import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.raytracer.raytracing_manager import RaytracingManager
from src.config_manager import ConfigManager
from src.utils.logging_utils import log_info, set_log_file
from src.utils.file_utils import ensure_directory_exists

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Batch execute ray tracing')
    
    # Config file arguments
    parser.add_argument('--config', type=str, 
                        help='Config file path (default: configs/regions_config.yaml)')
    
    # Ray tracing parameters
    parser.add_argument('--scenes-dir', type=str,
                        help='Scenes directory path')
    parser.add_argument('--results-dir', type=str,
                        help='Results output directory')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum parallel workers (default uses config file)')
    parser.add_argument('--gpu-mode', type=str,
                        choices=['auto', 'all', 'cpu_only'],
                        help='GPU usage mode')
    
    # Scene selection parameters
    parser.add_argument('--scene-pattern', type=str,
                        help='Scene name matching pattern')
    parser.add_argument('--max-scenes', type=int,
                        help='Maximum number of scenes to process')
    
    # Other parameters
    parser.add_argument('--log-file', type=str,
                        help='Log file path')
    parser.add_argument('--show-config', action='store_true',
                        help='Show current configuration and exit')
    parser.add_argument('--list-scenes', action='store_true',
                        help='List available scenes and exit')
    
    return parser.parse_args()

def safe_get_frequency(raytracing_config, default=3.66e9):
    """Safely get frequency value, handling string and numeric types"""
    try:
        frequency = raytracing_config.get('simulation', {}).get('frequency', default)
        # If it is a string, try to convert to float
        if isinstance(frequency, str):
            frequency = float(frequency)
        return frequency
    except (ValueError, TypeError):
        log_info(f"Warning: Failed to parse frequency value, using default {default/1e9:.2f} GHz")
        return default

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize config manager
    config_manager = ConfigManager(args.config)
    
    # Handle special commands
    if args.show_config:
        print("Current ray tracing configuration:")
        raytracing_config = config_manager.get('raytracing', {})
        print(f"  Engine: {raytracing_config.get('engine', 'Not set')}")
        print(f"  GPU mode: {raytracing_config.get('gpu_config', {}).get('gpu_mode', 'Not set')}")
        print(f"  Max parallel jobs: {raytracing_config.get('max_parallel_jobs', 'Not set')}")
        
        # Safely get frequency
        frequency = safe_get_frequency(raytracing_config)
        print(f"  Frequency: {frequency/1e9:.2f} GHz")
        
        # Show visualization configuration
        viz_config = config_manager.get('visualization', {})
        print(f"  Heatmap generation: {'Enabled' if viz_config.get('enabled', False) else 'Disabled'}")
        return True
    
    # Merge config file and CLI args
    effective_workers = args.max_workers if args.max_workers is not None else config_manager.get(
        'raytracing.max_parallel_jobs', 1
    ) or 1
    if args.gpu_mode:
        config_manager.config.setdefault('raytracing', {}).setdefault('gpu_config', {})['gpu_mode'] = args.gpu_mode
    config_manager.config.setdefault('raytracing', {}).setdefault('gpu_config', {})
    config_manager.config['raytracing']['max_parallel_jobs'] = effective_workers
    
    # Ensure visualization config exists
    if 'visualization' not in config_manager.config:
        config_manager.config['visualization'] = {
            'enabled': True,
            'point_size': 50,
            'dpi': 300
        }
    
    # Fill in minimum samples ratio default
    sim_cfg = config_manager.config.setdefault('raytracing', {}).setdefault('simulation', {})
    sim_cfg.setdefault('min_samples_ratio', 0.5)

    # Set up logging
    log_file = args.log_file or 'logs/raytracing.log'
    ensure_directory_exists('logs')
    set_log_file(log_file)
    
    # Initialize ray tracing manager
    raytracing_manager = RaytracingManager(config_manager)
    
    # Discover scenes
    scenes = raytracing_manager.discover_scenes()
    
    if args.list_scenes:
        print(f"Found {len(scenes)} available scenes:")
        for scene in scenes:
            print(f"  {scene['scene_name']} - ({scene['latitude']:.6f}, {scene['longitude']:.6f})")
        return True
    
    if not scenes:
        log_info("No available scene files found")
        return False
    
    # Filter scenes
    if args.scene_pattern:
        scenes = [s for s in scenes if args.scene_pattern in s['scene_name']]
        log_info(f"After filtering by pattern '{args.scene_pattern}', {len(scenes)} scenes remain")
    
    if args.max_scenes and len(scenes) > args.max_scenes:
        scenes = scenes[:args.max_scenes]
        log_info(f"Limiting number of scenes to process to {args.max_scenes}")
    
    # Display configuration info
    raytracing_config = config_manager.get('raytracing', {})
    log_info(f"Starting ray tracing, scene count: {len(scenes)}")
    log_info(f"Engine: {raytracing_config.get('engine', 'sionna')}")
    log_info(f"GPU mode: {raytracing_config.get('gpu_config', {}).get('gpu_mode', 'auto')}")
    
    # Safely get and display frequency
    frequency = safe_get_frequency(raytracing_config)
    log_info(f"Frequency: {frequency/1e9:.2f} GHz")
    log_info(f"Max parallel jobs: {raytracing_config.get('max_parallel_jobs', 1)}")
    
    # Execute ray tracing
    results = raytracing_manager.run_batch_raytracing(scenes, max_workers=effective_workers)
    
    # Display results
    successful_results = [r for r in results if r['success']]
    log_info(f"Ray tracing completed! Success: {len(successful_results)}/{len(scenes)}")
    
    if successful_results:
        log_info("Successfully processed scenes:")
        for result in successful_results:
            scene_name = result['scene_info']['scene_name']
            execution_time = result['execution_time']
            num_paths = result.get('num_paths', 0)
            num_receivers = result.get('num_receivers', 0)
            
            log_info(f"  {scene_name}: {execution_time:.1f}s, "
                    f"{num_receivers} receivers, {num_paths} paths")
    
    return len(successful_results) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)