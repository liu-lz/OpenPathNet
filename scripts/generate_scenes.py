import sys
import os
import argparse
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.coordinate_generator import generate_random_coordinates
from src.scene_manager import SceneManager
from src.config_manager import ConfigManager
from src.utils.logging_utils import log_info, set_log_file
from src.utils.file_utils import ensure_directory_exists

def parse_arguments():
    """Parse command-line arguments (supports config overrides)"""
    parser = argparse.ArgumentParser(description='Batch generate ray-tracing scenes')
    
    # Config file arguments
    parser.add_argument('--config', type=str, 
                        help='Path to config file (default: configs/regions_config.yaml)')
    parser.add_argument('--region', type=str,
                        help='Use a predefined region (e.g., hefei, beijing, shanghai)')
    
    # Basic parameters (override config file)
    parser.add_argument('--num-scenes', type=int,
                        help='Number of scenes to generate')
    parser.add_argument('--center-lat', type=float,
                        help='Center latitude')
    parser.add_argument('--center-lon', type=float,
                        help='Center longitude')
    parser.add_argument('--radius-km', type=float,
                        help='Generation radius (km)')
    parser.add_argument('--data-dir', type=str,
                        help='Data storage directory')
    parser.add_argument('--log-file', type=str,
                        help='Log file path')
    parser.add_argument('--size-x', type=int,
                        help='Scene size X (meters)')
    parser.add_argument('--size-y', type=int,
                        help='Scene size Y (meters)')
    
    # Generation mode parameters
    parser.add_argument('--generation-mode', type=str,
                        choices=['fallback', 'osm_retry'],
                        help='Scene generation mode')
    parser.add_argument('--max-osm-attempts', type=int,
                        help='Max attempts in OSM retry mode')
    parser.add_argument('--search-radius-km', type=float,
                        help='Search radius in OSM retry mode (km)')
    
    # Other parameters
    parser.add_argument('--list-regions', action='store_true',
                        help='List all available regions and exit')
    parser.add_argument('--show-config', action='store_true',
                        help='Show current configuration and exit')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize config manager
    config_manager = ConfigManager(args.config)
    
    # Handle special commands
    if args.list_regions:
        regions = config_manager.list_available_regions()
        print("Available regions:")
        for region in regions:
            region_config = config_manager.get_region_config(region)
            print(f"  {region}: {region_config.get('name', 'Unnamed')} - "
                  f"({region_config.get('center_lat')}, {region_config.get('center_lon')})")
        return True
    
    if args.show_config:
        print("Current configuration:")
        print(f"  Config file: {config_manager.config_file_path}")
        print(f"  Default region: {config_manager.get('default_region.name', 'Not set')}")
        print(f"  Generation mode: {config_manager.get('scene_generation.generation_mode', 'Not set')}")
        print(f"  Data directory: {config_manager.get('data_storage.base_data_dir', 'Not set')}")
        return True
    
    # Merge config file and CLI args
    config = config_manager.merge_with_args(args)
    
    # If a region is specified, use its config
    if args.region:
        region_config = config_manager.get_region_config(args.region)
        if region_config:
            config.update(region_config)
            log_info(f"Using region config: {args.region} - {region_config.get('name', 'Unnamed')}")
    
    # Set up logging
    ensure_directory_exists('logs')
    set_log_file(config['log_file'])
    
    # Ensure data directory exists
    data_dir_abs = os.path.abspath(config['data_dir'])
    if not ensure_directory_exists(data_dir_abs):
        log_info(f"Failed to create data directory: {data_dir_abs}")
        return False
    
    # Show configuration info
    log_info(f"Start generating {config['num_scenes']} scenes")
    log_info(f"Center coordinates: ({config['center_lat']}, {config['center_lon']})")
    log_info(f"Generation radius: {config['radius_km']} km")
    log_info(f"Scene size: {config['size_x']}x{config['size_y']} meters")
    log_info(f"Data directory: {data_dir_abs}")
    log_info(f"Generation mode: {config['generation_mode']}")
    
    if config['generation_mode'] == 'fallback':
        log_info("Strategy: Prefer OSM data; if it fails, automatically generate random building scenes")
    elif config['generation_mode'] == 'osm_retry':
        log_info("Strategy: Prefer OSM data; if it fails, re-sample within search radius and retry OSM")
        log_info(f"OSM retry params - Max attempts: {config['max_osm_attempts']}, "
                f"Search radius: {config['search_radius_km']}km")
    
    # Generate random coordinates
    coordinates = generate_random_coordinates(
        center_lat=config['center_lat'],
        center_lon=config['center_lon'],
        radius_km=config['radius_km'],
        num_points=config['num_scenes']
    )
    
    log_info(f"Generated {len(coordinates)} coordinate points")
    
    # Initialize scene manager
    scene_manager = SceneManager(config_manager)
    
    # Batch generate scenes
    successful_scenes = scene_manager.batch_generate_scenes(
        coordinates,
        generation_mode=config['generation_mode'],
        max_osm_attempts=config['max_osm_attempts'],
        search_radius_km=config['search_radius_km']
    )
    
    log_info(f"Scene generation completed! Success: {len(successful_scenes)}/{len(coordinates)}")
    
    # Show details of generated scenes
    if successful_scenes:
        log_info("Details of successfully generated scenes:")
        osm_scenes = []
        random_scenes = []
        
        for i, scene_result in enumerate(successful_scenes, 1):
            scene_file = scene_result['scene_file']
            generation_type = scene_result['generation_type']
            attempts = scene_result['attempts']
            original_coords = scene_result['original_coordinates']
            actual_coords = scene_result['coordinates']
            
            log_info(f"  {i}. {os.path.basename(os.path.dirname(scene_file))} - "
                    f"Type: {generation_type} - Attempts: {attempts} times")
            log_info(f"     Original coordinates: ({original_coords[0]:.6f}, {original_coords[1]:.6f})")
            
            if generation_type == 'OSM':
                osm_scenes.append(scene_result)
                if 'distance_from_original' in scene_result:
                    distance = scene_result['distance_from_original']
                    log_info(f"     Actual coordinates: ({actual_coords[0]:.6f}, {actual_coords[1]:.6f}) "
                            f"Offset: {distance:.2f}km")
                else:
                    log_info(f"     Actual coordinates: ({actual_coords[0]:.6f}, {actual_coords[1]:.6f}) "
                            f"Offset: 0.00km")
            else:
                random_scenes.append(scene_result)
                log_info(f"     Actual coordinates: ({actual_coords[0]:.6f}, {actual_coords[1]:.6f})")
        
        # Statistics
        log_info(f"\nScene type statistics:")
        log_info(f"  OSM scenes: {len(osm_scenes)}")
        log_info(f"  Random scenes: {len(random_scenes)}")
        
        if config['generation_mode'] == 'osm_retry' and osm_scenes:
            total_attempts = sum(s['attempts'] for s in osm_scenes)
            avg_attempts = total_attempts / len(osm_scenes)
            total_distance = sum(s.get('distance_from_original', 0) for s in osm_scenes)
            avg_distance = total_distance / len(osm_scenes)
            log_info(f"  Average attempts for OSM scenes: {avg_attempts:.1f}")
            log_info(f"  Average offset distance for OSM scenes: {avg_distance:.2f}km")
    
    # Adjust scene directory names by removing _attempt/_random suffixes
    successful_scenes = sanitize_scene_directories(successful_scenes)

    # Save generation results
    save_generation_results(successful_scenes, coordinates, config, data_dir_abs)
    
    return len(successful_scenes) > 0

def sanitize_scene_directories(scenes):
    """Remove _attempt/_random suffix from scene directory names"""
    sanitized = []
    for scene in scenes:
        scene_file = scene.get('scene_file')
        if not scene_file:
            sanitized.append(scene)
            continue
        scene_dir = os.path.dirname(scene_file)
        original_name = os.path.basename(scene_dir)
        stripped_name = _strip_scene_suffix(original_name)
        if stripped_name == original_name:
            sanitized.append(scene)
            continue
        parent_dir = os.path.dirname(scene_dir)
        new_dir = os.path.join(parent_dir, stripped_name)
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.rename(scene_dir, new_dir)
        updated_scene = dict(scene)
        updated_scene['scene_file'] = os.path.join(new_dir, os.path.basename(scene_file))
        if 'scene_directory' in updated_scene:
            updated_scene['scene_directory'] = new_dir
        sanitized.append(updated_scene)
    return sanitized

def _strip_scene_suffix(name: str) -> str:
    """Remove suffixes like _attempt/_random from scene directory names"""
    if '_attempt_' in name:
        return name.split('_attempt_')[0]
    if name.endswith('_random'):
        return name[:-7]
    return name

def save_generation_results(successful_scenes, coordinates, config, data_dir_abs):
    """Save generation results to files"""
    # Save list of successful scenes
    scenes_list_file = os.path.join(data_dir_abs, 'generated_scenes.txt')
    try:
        with open(scenes_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# Generated scene list\n")
            f.write(f"# Total: {len(successful_scenes)}/{len(coordinates)}\n")
            f.write(f"# Generation time: {os.popen('date').read().strip()}\n")
            f.write(f"# Center coordinates: ({config['center_lat']}, {config['center_lon']})\n")
            f.write(f"# Generation radius: {config['radius_km']} km\n")
            f.write(f"# Scene size: {config['size_x']}x{config['size_y']} meters\n")
            f.write(f"# Generation mode: {config['generation_mode']}\n")
            if config['generation_mode'] == 'osm_retry':
                f.write(f"# OSM retry params: max attempts {config['max_osm_attempts']}, search radius {config['search_radius_km']}km\n")
            f.write(f"\n")
            f.write(f"# Format: scene_file_path\toriginal_lat\toriginal_lon\tactual_lat\tactual_lon\tgeneration_type\tattempts\toffset_distance(km)\n")
            
            for scene_result in successful_scenes:
                scene_file = scene_result['scene_file']
                original_coords = scene_result['original_coordinates']
                actual_coords = scene_result['coordinates']
                generation_type = scene_result['generation_type']
                attempts = scene_result['attempts']
                distance = scene_result.get('distance_from_original', 0.0)
                
                f.write(f"{scene_file}\t{original_coords[0]:.6f}\t{original_coords[1]:.6f}\t"
                       f"{actual_coords[0]:.6f}\t{actual_coords[1]:.6f}\t{generation_type}\t"
                       f"{attempts}\t{distance:.2f}\n")
                    
        log_info(f"Scene list saved to: {scenes_list_file}")
    except Exception as e:
        log_info(f"Failed to save scene list: {e}")
    
    # Generate scene statistics report
    try:
        stats_file = os.path.join(data_dir_abs, 'generation_stats.txt')
        osm_scenes = [s for s in successful_scenes if s['generation_type'] == 'OSM']
        random_scenes = [s for s in successful_scenes if s['generation_type'] == 'Random']
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Scene generation statistics report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Requested scene count: {len(coordinates)}\n")
            f.write(f"Successfully generated scenes: {len(successful_scenes)}\n")
            f.write(f"Failed scene count: {len(coordinates) - len(successful_scenes)}\n")
            f.write(f"Success rate: {len(successful_scenes)/len(coordinates)*100:.1f}%\n")
            f.write(f"Center coordinates: ({config['center_lat']}, {config['center_lon']})\n")
            f.write(f"Generation radius: {config['radius_km']} km\n")
            f.write(f"Scene size: {config['size_x']}x{config['size_y']} meters\n")
            f.write(f"Generation mode: {config['generation_mode']}\n")
            
            f.write(f"\nScene type statistics:\n")
            f.write(f"OSM scenes: {len(osm_scenes)} ({len(osm_scenes)/len(successful_scenes)*100:.1f}%)\n")
            f.write(f"Random scenes: {len(random_scenes)} ({len(random_scenes)/len(successful_scenes)*100:.1f}%)\n")
            
            if config['generation_mode'] == 'osm_retry' and osm_scenes:
                total_attempts = sum(s['attempts'] for s in osm_scenes)
                avg_attempts = total_attempts / len(osm_scenes)
                total_distance = sum(s.get('distance_from_original', 0) for s in osm_scenes)
                avg_distance = total_distance / len(osm_scenes)
                
                f.write(f"\nOSM retry statistics:\n")
                f.write(f"Configured max attempts: {config['max_osm_attempts']}\n")
                f.write(f"Configured search radius: {config['search_radius_km']} km\n")
                f.write(f"Average attempts: {avg_attempts:.1f}\n")
                f.write(f"Average offset distance: {avg_distance:.2f} km\n")
                f.write(f"Total attempts: {total_attempts}\n")
                
                # Attempt count distribution
                attempt_counts = {}
                for s in osm_scenes:
                    attempts = s['attempts']
                    attempt_counts[attempts] = attempt_counts.get(attempts, 0) + 1
                
                f.write(f"\nAttempts distribution:\n")
                for attempts in sorted(attempt_counts.keys()):
                    count = attempt_counts[attempts]
                    f.write(f"  {attempts} times: {count} scenes\n")
            
        log_info(f"Statistics report saved to: {stats_file}")
    except Exception as e:
        log_info(f"Failed to save statistics report: {e}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)