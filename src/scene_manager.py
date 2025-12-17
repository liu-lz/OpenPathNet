import os
import subprocess
import time
import random
import math
from typing import Dict, Any
from .utils.logging_utils import log_info
from .utils.file_utils import ensure_directory_exists
from .config_manager import ConfigManager

class SceneManager:
    """Scene manager responsible for generating and managing scene files"""
    
    def __init__(self, config_manager: ConfigManager = None):
        """
        Initialize the scene manager.
        
        Args:
            config_manager: Config manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Get data storage paths from config
        storage_config = self.config_manager.get_data_storage_config()
        self.base_data_dir = storage_config.get('base_data_dir', '/home/ubuntu/llz/liulz/RFmap')
        self.scenes_subdir = storage_config.get('scenes_subdir', 'scenes')
        self.mesh_subdir = storage_config.get('mesh_subdir', 'mesh')
        
        self.scenes_dir = os.path.join(self.base_data_dir, self.scenes_subdir)
        ensure_directory_exists(self.scenes_dir)
        
        # Get scene generation parameters from config
        self.scene_config = self.config_manager.get_scene_generation_config()
        self.materials_config = self.config_manager.get_materials_config()
        self.mitsuba_config = self.config_manager.get_mitsuba_config()
    
    def generate_scene(self, lat, lon, size_x=None, size_y=None, generation_mode=None, 
                      max_osm_attempts=None, search_radius_km=None):
        """
        Generate a single scene (using default parameters from config).
        """
        # Get default parameters from config
        size_x = size_x or self.scene_config.get('size_x', 130)
        size_y = size_y or self.scene_config.get('size_y', 130)
        generation_mode = generation_mode or self.scene_config.get('generation_mode', 'fallback')
        
        osm_retry_config = self.scene_config.get('osm_retry', {})
        max_osm_attempts = max_osm_attempts or osm_retry_config.get('max_attempts', 10)
        search_radius_km = search_radius_km or osm_retry_config.get('search_radius_km', 2.0)
        
        if generation_mode == "fallback":
            return self._generate_scene_fallback(lat, lon, size_x, size_y)
        elif generation_mode == "osm_retry":
            return self._generate_scene_osm_retry(lat, lon, size_x, size_y, 
                                                max_osm_attempts, search_radius_km)
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
    
    def _generate_scene_fallback(self, lat, lon, size_x, size_y):
        """Original mode: prefer OSM; if it fails, generate a random scene."""
        scene_name = f"scene_{lat:.6f}_{lon:.6f}"
        scene_dir = os.path.join(self.scenes_dir, scene_name)
        ensure_directory_exists(scene_dir)
        
        # Try OSM generation
        osm_result = self._try_osm_generation(lat, lon, size_x, size_y, scene_dir)
        
        if osm_result['success']:
            return {
                'scene_file': osm_result['scene_file'],
                'generation_type': 'OSM',
                'attempts': 1,
                'coordinates': (lat, lon),
                'original_coordinates': (lat, lon)
            }
        else:
            # OSM failed; generate random buildings scene
            log_info("OSM generation failed, generating random buildings scene")
            scene_file = self._generate_random_buildings_scene(lat, lon, size_x, size_y, scene_dir)
            
            if scene_file:
                return {
                    'scene_file': scene_file,
                    'generation_type': 'Random',
                    'attempts': 1,
                    'coordinates': (lat, lon),
                    'original_coordinates': (lat, lon)
                }
            else:
                return None
    
    def _generate_scene_osm_retry(self, original_lat, original_lon, size_x, size_y, 
                                max_attempts, search_radius_km):
        """New mode: prefer OSM; on failure, retry with nearby points, optionally fall back to random."""
        log_info(f"Using OSM retry mode, max attempts: {max_attempts}, search radius: {search_radius_km}km")
        
        # Check if fallback to random generation is enabled
        osm_retry_config = self.scene_config.get('osm_retry', {})
        fallback_to_random = osm_retry_config.get('fallback_to_random', True)
        
        
        # First try the original coordinates
        attempt_coords = [(original_lat, original_lon, 0)]  # (lat, lon, distance)
        
        # Generate candidate coordinates within search radius
        for _ in range(max_attempts - 1):
            # Randomly generate a point within the search radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, search_radius_km)
            
            # Calculate new coordinates
            lat_offset = distance * math.cos(angle) / 111.32  # 1 degree latitude is about 111.32km
            lon_offset = distance * math.sin(angle) / (111.32 * math.cos(math.radians(original_lat)))
            
            new_lat = original_lat + lat_offset
            new_lon = original_lon + lon_offset
            
            attempt_coords.append((new_lat, new_lon, distance))
        # Sort by distance, try closer points first
        attempt_coords.sort(key=lambda x: x[2])
        
        for attempt, (lat, lon, distance) in enumerate(attempt_coords, 1):
            log_info(f"OSM attempt {attempt}/{max_attempts}: ({lat:.6f}, {lon:.6f}) "
                    f"distance from origin: {distance:.2f}km")
            
            # Create a unique scene directory for each attempt
            scene_name = f"scene_{original_lat:.6f}_{original_lon:.6f}"
            if attempt > 1:
                scene_name += f"_attempt_{attempt}"
            
            scene_dir = os.path.join(self.scenes_dir, scene_name)
            ensure_directory_exists(scene_dir)
            
            # Try OSM generation
            osm_result = self._try_osm_generation(lat, lon, size_x, size_y, scene_dir)
            
            if osm_result['success']:
                log_info(f"✓ OSM generation succeeded using coords: ({lat:.6f}, {lon:.6f})")
                return {
                    'scene_file': osm_result['scene_file'],
                    'generation_type': 'OSM',
                    'attempts': attempt,
                    'coordinates': (lat, lon),
                    'original_coordinates': (original_lat, original_lon),
                    'distance_from_original': distance
                }
            else:
                log_info(f"✗ OSM generation failed, coords: ({lat:.6f}, {lon:.6f})")
                # Clean up the failed scene directory
                if os.path.exists(scene_dir):
                    import shutil
                    shutil.rmtree(scene_dir)
        
        # All OSM attempts failed
        if fallback_to_random:
            log_info(f"All {max_attempts} OSM attempts failed, using random generation")
            
            # Use the original coordinates to generate a random scene
            scene_name = f"scene_{original_lat:.6f}_{original_lon:.6f}_random"
            scene_dir = os.path.join(self.scenes_dir, scene_name)
            ensure_directory_exists(scene_dir)
            
            # Generate random buildings scene
            scene_file = self._generate_random_buildings_scene(original_lat, original_lon, 
                                                             size_x, size_y, scene_dir)
            
            if scene_file:
                log_info(f"✓ Random scene generated successfully using original coords: ({original_lat:.6f}, {original_lon:.6f})")
                return {
                    'scene_file': scene_file,
                    'generation_type': 'Random',
                    'attempts': max_attempts + 1,
                    'coordinates': (original_lat, original_lon),
                    'original_coordinates': (original_lat, original_lon),
                    'distance_from_original': 0.0,
                    'fallback_reason': 'OSM retry failed'
                }
            else:
                log_info("✗ Random scene generation also failed")
                return None
        else:
            log_info(f"All {max_attempts} OSM attempts failed and random fallback is disabled")
            return None
    
    def _try_osm_generation(self, lat, lon, size_x, size_y, scene_dir):
        """Attempt OSM scene generation."""
        cmd = [
            "scenegen", "point", 
            str(lon), str(lat), 
            "center", str(size_x), str(size_y),
            "--data-dir", scene_dir
        ]
        
        # Get timeout from config
        timeout_seconds = self.scene_config.get('osm_retry', {}).get('timeout_seconds', 300)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            scene_file = os.path.join(scene_dir, "scene.xml")
            
            if result.returncode == 0 and os.path.exists(scene_file):
                if self._validate_scene_file(scene_file):
                    return {'success': True, 'scene_file': scene_file}
                else:
                    log_info(f"Invalid scene file: {scene_file}")
                    return {'success': False, 'error': 'Invalid scene file'}
            else:
                return {'success': False, 'error': f'scenegen failed: {result.stderr}'}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_scene_file(self, scene_file):
        """Validate the scene file."""
        try:
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic check: required elements must exist
            required_elements = ['<scene', '<shape', '</scene>']
            for element in required_elements:
                if element not in content:
                    return False
            
            # Ensure geometry exists
            if '<shape type="ply"' not in content:
                return False
            
            return True
        except:
            return False
    
    def _generate_random_buildings_scene(self, lat, lon, size_x, size_y, scene_dir):
        """Generate a random buildings scene (keep existing implementation)."""
        try:
            scene_file = os.path.join(scene_dir, "scene.xml")
            
            # Create mesh subdirectory
            mesh_dir = os.path.join(scene_dir, self.mesh_subdir)
            ensure_directory_exists(mesh_dir)
            
            # Create random buildings scene XML
            xml_content = self.create_osm_style_xml(lat, lon, size_x, size_y, mesh_dir)
            
            with open(scene_file, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            log_info(f"Random buildings scene generated successfully: {scene_file}")
            return scene_file
            
        except Exception as e:
            log_info(f"Random buildings scene generation failed: {e}")
            return None
    
    def create_box_mesh(self, width, length, height, x_offset=0, y_offset=0, z_offset=0):
        """Create a triangular mesh for a box."""
        hw, hl, hh = width/2, length/2, height/2
        
        vertices = [
            # Bottom face 4 vertices
            [x_offset - hw, y_offset - hl, z_offset],
            [x_offset + hw, y_offset - hl, z_offset],
            [x_offset + hw, y_offset + hl, z_offset],
            [x_offset - hw, y_offset + hl, z_offset],
            # Top face 4 vertices
            [x_offset - hw, y_offset - hl, z_offset + height],
            [x_offset + hw, y_offset - hl, z_offset + height],
            [x_offset + hw, y_offset + hl, z_offset + height],
            [x_offset - hw, y_offset + hl, z_offset + height],
        ]
        
        faces = [
            # Bottom face (z=0)
            [0, 2, 1], [0, 3, 2],
            # Top face (z=height)  
            [4, 5, 6], [4, 6, 7],
            # Front face (y=-hl)
            [0, 1, 5], [0, 5, 4],
            # Back face (y=+hl)
            [2, 3, 7], [2, 7, 6],
            # Left face (x=-hw)
            [0, 4, 7], [0, 7, 3],
            # Right face (x=+hw)
            [1, 2, 6], [1, 6, 5],
        ]
        
        return vertices, faces
    
    def create_ground_mesh(self, width, length):
        """Create a triangular mesh for the ground."""
        hw, hl = width/2, length/2
        
        vertices = [
            [-hw, -hl, 0],
            [+hw, -hl, 0],
            [+hw, +hl, 0],
            [-hw, +hl, 0],
        ]
        
        faces = [
            [0, 1, 2],
            [0, 2, 3],
        ]
        
        return vertices, faces
    
    def save_ply_file(self, vertices, faces, filename):
        """Save a PLY file."""
        try:
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write faces
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            
            return True
        except Exception as e:
            log_info(f"Failed to save PLY file {filename}: {e}")
            return False
        
    def create_osm_style_xml(self, lat, lon, size_x, size_y, mesh_dir):
        """Create an OSM-style XML scene using config parameters."""
        
        # Get random scene generation parameters from config
        random_config = self.scene_config.get('random_scene', {})
        building_count_range = random_config.get('building_count_range', [6, 12])
        building_size_range = random_config.get('building_size_range', [15, 25])
        building_height_range = random_config.get('building_height_range', [20, 50])
        min_building_distance = random_config.get('min_building_distance', 25)
        
        # Generate scene objects
        shapes = []
        
        # 1. Ground
        ground_vertices, ground_faces = self.create_ground_mesh(size_x * 2, size_y * 2)
        ground_ply = os.path.join(mesh_dir, "ground.ply")
        
        if self.save_ply_file(ground_vertices, ground_faces, ground_ply):
            # Use configured ground material
            ground_material = self._get_material_id('itu_wet_ground')
            shapes.append(f"""    <shape type="ply" id="mesh-ground">
        <string name="filename" value="{self.mesh_subdir}/ground.ply"/>
        <ref id="{ground_material}" name="bsdf"/>
        <boolean name="face_normals" value="true"/>
    </shape>""")
        
        # 2. Buildings
        num_buildings = random.randint(*building_count_range)
        used_positions = []
        
        # Get available building materials
        building_materials = self._get_building_materials()
        
        for i in range(num_buildings):
            # Random position
            attempts = 0
            while attempts < 20:
                x = random.uniform(-size_x/3, size_x/3)
                y = random.uniform(-size_y/3, size_y/3)
                
                # Check for overlap
                too_close = False
                for px, py in used_positions:
                    if abs(x - px) < min_building_distance and abs(y - py) < min_building_distance:
                        too_close = True
                        break
                
                if not too_close:
                    used_positions.append((x, y))
                    break
                attempts += 1
            
            # Building dimensions
            width = random.uniform(*building_size_range)
            length = random.uniform(*building_size_range)
            height = random.uniform(*building_height_range)
            
            # Create building mesh
            building_vertices, building_faces = self.create_box_mesh(width, length, height, x, y, 0)
            building_ply = os.path.join(mesh_dir, f"building_{i}.ply")
            
            if self.save_ply_file(building_vertices, building_faces, building_ply):
                # Randomly select building material
                material = random.choice(building_materials)
                
                shapes.append(f"""    <shape type="ply" id="mesh-building_{i}">
        <string name="filename" value="{self.mesh_subdir}/building_{i}.ply"/>
        <ref id="{material}" name="bsdf"/>
        <boolean name="face_normals" value="true"/>
    </shape>""")
        
        shapes_xml = "\n".join(shapes)
        
        # Use Mitsuba parameters from config
        return self._create_xml_content(lat, lon, size_x, size_y, shapes_xml)
    
    def _get_material_id(self, material_key: str) -> str:
        """Get material ID from config."""
        materials = self.materials_config
        if material_key in materials:
            return materials[material_key].get('id', f'mat-{material_key}')
        return f'mat-{material_key}'
    
    def _get_building_materials(self) -> list:
        """Get available building material list."""
        materials = self.materials_config
        building_materials = []
        
        for key, config in materials.items():
            if key != 'itu_wet_ground':  # Exclude ground material
                building_materials.append(config.get('id', f'mat-{key}'))
        
        return building_materials or ["mat-itu_concrete", "mat-itu_marble", "mat-itu_metal", "mat-itu_wood"]
    
    def _create_materials_xml(self) -> str:
        """Create material definitions XML from config."""
        materials_xml = []
        
        for key, config in self.materials_config.items():
            material_id = config.get('id', f'mat-{key}')
            reflectance = config.get('reflectance', [0.5, 0.5, 0.5])
            
            materials_xml.append(f"""    <bsdf type="twosided" id="{material_id}">
        <bsdf type="diffuse">
            <rgb value="{reflectance[0]} {reflectance[1]} {reflectance[2]}" name="reflectance"/>
        </bsdf>
    </bsdf>""")
        
        return "\n".join(materials_xml)
    
    def _create_xml_content(self, lat, lon, size_x, size_y, shapes_xml) -> str:
        """Create full XML content using config parameters."""
        
        # Get Mitsuba parameters from config
        version = self.mitsuba_config.get('version', '2.1.0')
        
        integrator_config = self.mitsuba_config.get('integrator', {})
        max_depth = integrator_config.get('max_depth', 12)
        
        camera_config = self.mitsuba_config.get('camera', {})
        fov = camera_config.get('fov', 42.854885)
        camera_pos = camera_config.get('position', [0, 0, 100])
        
        film_config = self.mitsuba_config.get('film', {})
        film_width = film_config.get('width', 1024)
        film_height = film_config.get('height', 768)
        
        sampler_config = self.mitsuba_config.get('sampler', {})
        sample_count = sampler_config.get('sample_count', 4096)
        
        materials_xml = self._create_materials_xml()
        
        return f"""<?xml version="1.0" ?>
<scene version="{version}">
    <default name="spp" value="{sample_count}"/>
    <default name="resx" value="{film_width}"/>
    <default name="resy" value="{film_height}"/>
    <integrator type="path">
        <integer name="max_depth" value="{max_depth}"/>
    </integrator>
{materials_xml}
    <emitter type="constant" id="World">
        <rgb value="1.000000 1.000000 1.000000" name="radiance"/>
    </emitter>
    <sensor type="perspective" id="Camera">
        <string name="fov_axis" value="x"/>
        <float name="fov" value="{fov}"/>
        <float name="principal_point_offset_x" value="0.000000"/>
        <float name="principal_point_offset_y" value="-0.000000"/>
        <float name="near_clip" value="0.100000"/>
        <float name="far_clip" value="10000.000000"/>
        <transform name="to_world">
            <rotate x="1" angle="0"/>
            <rotate y="1" angle="0"/>
            <rotate z="1" angle="-90"/>
            <translate value="{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}"/>
        </transform>
        <sampler type="independent">
            <integer name="sample_count" value="$spp"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="$resx"/>
            <integer name="height" value="$resy"/>
        </film>
    </sensor>
{shapes_xml}
</scene>"""
    
    def batch_generate_scenes(self, coordinates_list, generation_mode=None, 
                            max_osm_attempts=None, search_radius_km=None, max_workers=4):
        """
        Batch generate scenes (using config parameters).
        """
        # Get default parameters from config
        generation_mode = generation_mode or self.scene_config.get('generation_mode', 'fallback')
        osm_retry_config = self.scene_config.get('osm_retry', {})
        max_osm_attempts = max_osm_attempts or osm_retry_config.get('max_attempts', 10)
        search_radius_km = search_radius_km or osm_retry_config.get('search_radius_km', 2.0)
        
        successful_scenes = []
        failed_scenes = []
        
        log_info(f"Starting batch generation of {len(coordinates_list)} scenes")
        log_info(f"Generation mode: {generation_mode}")
        if generation_mode == "osm_retry":
            log_info(f"OSM retry params - Max attempts: {max_osm_attempts}, Search radius: {search_radius_km}km")
            fallback_enabled = osm_retry_config.get('fallback_to_random', True)
            log_info(f"Random fallback after OSM failure: {'Enabled' if fallback_enabled else 'Disabled'}")
        
        for i, (lat, lon) in enumerate(coordinates_list):
            log_info(f"Processing scene {i+1}/{len(coordinates_list)}: ({lat:.6f}, {lon:.6f})")
            
            scene_result = self.generate_scene(
                lat, lon, 
                generation_mode=generation_mode,
                max_osm_attempts=max_osm_attempts,
                search_radius_km=search_radius_km
            )
            
            if scene_result:
                successful_scenes.append(scene_result)
                generation_type = scene_result['generation_type']
                attempts = scene_result['attempts']
                fallback_reason = scene_result.get('fallback_reason', '')
                
                log_info(f"✓ Scene generated successfully: {generation_type} (attempts {attempts})")
                if fallback_reason:
                    log_info(f"  Fallback reason: {fallback_reason}")
                # If different coordinates were used, log the offset
                if 'distance_from_original' in scene_result:
                    log_info(f"  Actual coordinates: ({scene_result['coordinates'][0]:.6f}, "
                            f"{scene_result['coordinates'][1]:.6f}) "
                            f"distance from origin: {scene_result['distance_from_original']:.2f}km")
            else:
                failed_scenes.append((lat, lon))
                log_info("✗ Scene generation failed")
            
            time.sleep(1)
        
        # Generate summary report
        osm_count = sum(1 for s in successful_scenes if s['generation_type'] == 'OSM')
        random_count = sum(1 for s in successful_scenes if s['generation_type'] == 'Random')
        fallback_count = sum(1 for s in successful_scenes if s.get('fallback_reason'))
        
        log_info("Batch generation completed!")
        log_info(f"Total success: {len(successful_scenes)}/{len(coordinates_list)}")
        log_info(f"OSM scenes: {osm_count}, Random scenes: {random_count}, Failed: {len(failed_scenes)}")
        if fallback_count > 0:
            log_info(f"Random scenes after OSM retry failure: {fallback_count}")
        
        if generation_mode == "osm_retry" and osm_count > 0:
            total_attempts = sum(s['attempts'] for s in successful_scenes if s['generation_type'] == 'OSM')
            avg_attempts = total_attempts / osm_count
            log_info(f"Average OSM attempts: {avg_attempts:.1f}")
        
        return successful_scenes
    
    def cleanup_scene(self, scene_path):
        """Clean up scene files and related resources."""
        try:
            scene_dir = os.path.dirname(scene_path)
            if os.path.exists(scene_dir):
                import shutil
                shutil.rmtree(scene_dir)
                log_info(f"Cleaned scene directory: {scene_dir}")
        except Exception as e:
            log_info(f"Failed to clean scene: {e}")
    
    def get_scene_info(self, scene_path):
        """Get scene information."""
        try:
            scene_dir = os.path.dirname(scene_path)
            scene_name = os.path.basename(scene_dir)
            
            if scene_name.startswith("scene_"):
                coords_part = scene_name[6:]
                parts = coords_part.split("_")
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    
                    # Count PLY files in the mesh directory
                    mesh_dir = os.path.join(scene_dir, self.mesh_subdir)
                    ply_files = []
                    if os.path.exists(mesh_dir):
                        ply_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply')]
                    
                    return {
                        'scene_name': scene_name,
                        'scene_path': scene_path,
                        'latitude': lat,
                        'longitude': lon,
                        'ply_files_count': len(ply_files),
                        'ply_files': ply_files,
                        'exists': os.path.exists(scene_path)
                    }
        except Exception as e:
            log_info(f"Failed to get scene info: {e}")
        
        return None