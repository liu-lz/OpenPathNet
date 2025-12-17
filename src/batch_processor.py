from src.scene_manager import SceneManager
from src.utils.logging_utils import log_info, log_error
import argparse

def process_batch(num_scenes, center_lat, center_lon, area_size, data_dir):
    """Handles the batch processing of scene generation."""
    scene_manager = SceneManager(data_dir)
    
    for i in range(num_scenes):
        # Generate random coordinates based on the center point
        lat, lon = scene_manager.generate_random_coordinates(center_lat, center_lon, area_size)
        
        # Construct the scene generation command
        command = f"scenegen point {lat} {lon} center {area_size} {area_size} --data-dir {data_dir}"
        
        try:
            # Execute the scene generation command
            scene_manager.execute_scene_generation(command)
            log_info(f"Successfully generated scene at ({lat}, {lon})")
        except Exception as e:
            log_error(f"Error generating scene at ({lat}, {lon}): {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch process scene generation.")
    parser.add_argument('--num-scenes', type=int, required=True, help='Number of scenes to generate')
    parser.add_argument('--center-lat', type=float, required=True, help='Central latitude for scene generation')
    parser.add_argument('--center-lon', type=float, required=True, help='Central longitude for scene generation')
    parser.add_argument('--area-size', type=int, default=130, help='Size of the area for scene generation (default: 130m)')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory to save generated scenes')
    
    args = parser.parse_args()
    
    process_batch(args.num_scenes, args.center_lat, args.center_lon, args.area_size, args.data_dir)

if __name__ == "__main__":
    main()