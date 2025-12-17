import os
import yaml
from typing import Dict, Any, Optional
from .utils.logging_utils import log_info
import copy

class ConfigManager:
    """Configuration manager responsible for loading and managing all configuration parameters"""
    
    def __init__(self, config_file_path: str = None, config_data: Dict[str, Any] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file_path: Path to the config file; if None, use the default path.
            config_data: Config data passed directly as a dictionary for in-memory testing.
        """
        if config_data is not None:
            self.config_file_path = config_file_path or "<in-memory>"
            self.config = copy.deepcopy(config_data)
        else:
            if config_file_path is None:
                # Default config file path
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_file_path = os.path.join(project_root, "configs", "regions_config.yaml")
            self.config_file_path = config_file_path
            self.config = self._load_config()
        
        self.config.setdefault('raytracing', {}).setdefault('simulation', {}).setdefault('min_samples_ratio', 0.5)
        
        # Preprocess numeric values in the config
        self._preprocess_numeric_values()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            log_info(f"Config file loaded successfully: {self.config_file_path}")
            return config
        except FileNotFoundError:
            log_info(f"Config file not found: {self.config_file_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            log_info(f"Config file format error: {e}")
            return self._get_default_config()
        except Exception as e:
            log_info(f"Failed to load config file: {e}")
            return self._get_default_config()
    
    def _preprocess_numeric_values(self):
        """Preprocess numeric values in the config, converting strings to numeric types."""
        # Ray tracing numeric configs
        raytracing_numeric_configs = [
            # (section, subsection, key, type)
            ('raytracing', 'simulation', 'frequency', float),
            ('raytracing', 'simulation', 'max_depth', int),
            ('raytracing', 'simulation', 'ray_samples', int),
            ('raytracing', 'simulation', 'scat_keep_prob', float),
            ('raytracing', 'receiver_grid', 'grid_size', int),
            ('raytracing', 'receiver_grid', 'area_size', float),
            ('raytracing', 'transmitter', 'power_watts', float),
            ('raytracing', 'batch_processing', 'rx_batch_size', int),
            ('raytracing', 'batch_processing', 'batch_pause', float),
            ('raytracing', 'gpu_config', 'memory_limit', int),
            ('raytracing', 'gpu_config', 'max_gpus', int),
            ('raytracing', 'simulation', 'min_samples_ratio', float),
            ('raytracing', None, 'timeout_per_scene', int),
        ]
        
        # Scene generation numeric configs
        scene_numeric_configs = [
            ('scene_generation', None, 'num_scenes', int),
            ('scene_generation', None, 'size_x', int),
            ('scene_generation', None, 'size_y', int),
            ('scene_generation', 'osm_retry', 'max_attempts', int),
            ('scene_generation', 'osm_retry', 'search_radius_km', float),
            ('scene_generation', 'osm_retry', 'timeout_seconds', int),
            ('scene_generation', 'random_scene', 'min_building_distance', float),
        ]
        
        # Region numeric configs
        region_numeric_configs = [
            ('default_region', None, 'center_lat', float),
            ('default_region', None, 'center_lon', float),
            ('default_region', None, 'radius_km', float),
        ]
        
        all_configs = raytracing_numeric_configs + scene_numeric_configs + region_numeric_configs
        
        for config_path in all_configs:
            self._convert_config_value(*config_path)
        
        # Handle array-type config values
        self._preprocess_array_values()
    
    def _convert_config_value(self, section: str, subsection: Optional[str], 
                            key: str, target_type: type):
        """
        Convert a single config value to a target type.
        
        Args:
            section: Main config section
            subsection: Subsection (optional)
            key: Config key
            target_type: Target type (int, float, bool, etc.)
        """
        try:
            # Get the config section
            if section not in self.config:
                return
            
            section_config = self.config[section]
            
            # If subsection exists, dive into it
            if subsection:
                if subsection not in section_config:
                    return
                target_config = section_config[subsection]
            else:
                target_config = section_config
            
            # Check if the key exists
            if key not in target_config:
                return
            
            current_value = target_config[key]
            
            # If it's already the target type, skip
            if isinstance(current_value, target_type):
                return
            
            # If it's a string, try to convert
            if isinstance(current_value, str):
                try:
                    if target_type == bool:
                        # Handle boolean strings
                        converted_value = current_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        converted_value = target_type(current_value)
                    
                    target_config[key] = converted_value
                    
                    # Remove the log output for successful conversion
                    # Construct the config path for logging
                    # config_path = f"{section}.{subsection}.{key}" if subsection else f"{section}.{key}"
                    # log_info(f"Converted config {config_path} from string '{current_value}' to {target_type.__name__} {converted_value}")
                    
                except (ValueError, TypeError) as e:
                    config_path = f"{section}.{subsection}.{key}" if subsection else f"{section}.{key}"
                    log_info(f"Warning: Unable to convert config {config_path} value '{current_value}' to {target_type.__name__}: {e}")
        except Exception as e:
            config_path = f"{section}.{subsection}.{key}" if subsection else f"{section}.{key}"
            log_info(f"Error processing config {config_path}: {e}")
    
    def _preprocess_array_values(self):
        """Preprocess array-type config values."""
        # Handle building_count_range
        try:
            building_count_range = self.get('scene_generation.random_scene.building_count_range')
            if building_count_range and isinstance(building_count_range, list):
                converted_range = []
                for val in building_count_range:
                    if isinstance(val, str):
                        try:
                            converted_range.append(int(val))
                        except ValueError:
                            converted_range.append(val)
                    else:
                        converted_range.append(val)
                
                if converted_range != building_count_range:
                    self.config['scene_generation']['random_scene']['building_count_range'] = converted_range
                    # Remove the log output for conversion
                    # log_info(f"Converted building_count_range: {building_count_range} -> {converted_range}")
        except Exception as e:
            log_info(f"Error processing building_count_range: {e}")
        
        # Handle building_size_range
        try:
            building_size_range = self.get('scene_generation.random_scene.building_size_range')
            if building_size_range and isinstance(building_size_range, list):
                converted_range = []
                for val in building_size_range:
                    if isinstance(val, str):
                        try:
                            converted_range.append(float(val))
                        except ValueError:
                            converted_range.append(val)
                    else:
                        converted_range.append(val)
                
                if converted_range != building_size_range:
                    self.config['scene_generation']['random_scene']['building_size_range'] = converted_range
                    # Remove the log output for conversion
                    # log_info(f"Converted building_size_range: {building_size_range} -> {converted_range}")
        except Exception as e:
            log_info(f"Error processing building_size_range: {e}")
        
        # Handle building_height_range
        try:
            building_height_range = self.get('scene_generation.random_scene.building_height_range')
            if building_height_range and isinstance(building_height_range, list):
                converted_range = []
                for val in building_height_range:
                    if isinstance(val, str):
                        try:
                            converted_range.append(float(val))
                        except ValueError:
                            converted_range.append(val)
                    else:
                        converted_range.append(val)
                
                if converted_range != building_height_range:
                    self.config['scene_generation']['random_scene']['building_height_range'] = converted_range
                    # Remove the log output for conversion
                    # log_info(f"Converted building_height_range: {building_height_range} -> {converted_range}")
        except Exception as e:
            log_info(f"Error processing building_height_range: {e}")
        
        # Handle transmitter position and orientation
        try:
            tx_position = self.get('raytracing.transmitter.position')
            if tx_position and isinstance(tx_position, list):
                converted_position = []
                for val in tx_position:
                    if isinstance(val, str):
                        try:
                            converted_position.append(float(val))
                        except ValueError:
                            converted_position.append(val)
                    else:
                        converted_position.append(val)
                
                if converted_position != tx_position:
                    self.config['raytracing']['transmitter']['position'] = converted_position
                    # Remove the log output for conversion
                    # log_info(f"Converted transmitter.position: {tx_position} -> {converted_position}")
        except Exception as e:
            log_info(f"Error processing transmitter.position: {e}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Return a fallback default config (used only when the config file is missing)."""
        return {
            'default_region': {
                'center_lat': 31.839380,
                'center_lon': 117.250454,
                'radius_km': 5.0
            },
            'scene_generation': {
                'num_scenes': 10,
                'size_x': 130,
                'size_y': 130,
                'generation_mode': 'fallback',
                'osm_retry': {
                    'max_attempts': 10,
                    'search_radius_km': 2.0,
                    'timeout_seconds': 300,
                    'fallback_to_random': True  # Added default config
                },
                'random_scene': {
                    'building_count_range': [6, 12],
                    'building_size_range': [15, 25],
                    'building_height_range': [20, 50],
                    'min_building_distance': 25
                }
            },
            'raytracing': {
                'engine': 'sionna',
                'simulation': {
                    'frequency': 3.66e9,
                    'max_depth': 10,
                    'synthetic_array': True
                },
                'transmitter': {
                    'position': [0, 0, 30],
                    'power_watts': 100
                },
                'receiver_grid': {
                    'grid_size': 128,
                    'area_size': 128,
                    'center': [0, 0, 1.5]
                },
                'gpu_config': {
                    'gpu_mode': 'auto'
                }
            },
            'data_storage': {
                'base_data_dir': 'data',
                'scenes_subdir': 'scenes',
                'mesh_subdir': 'mesh',
                'logs_dir': 'logs'
            },
            'logging': {
                'log_file': 'logs/scene_generation.log',
                'log_level': 'INFO'
            }
        }
    
    def get(self, key_path: str, default=None):
        """
        Retrieve a config value.
        
        Args:
            key_path: Dotted path for nested keys, e.g., 'scene_generation.size_x'
            default: Default value if not found
        
        Returns:
            The config value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a config value.
        
        Args:
            key_path: Dotted path for nested keys
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Create nested dictionary structure
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        
    def validate_config(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if the config is valid
        """
        try:
            # Check required sections
            required_sections = ['scene_generation', 'data_storage']
            for section in required_sections:
                if section not in self.config:
                    log_info(f"Missing required config section: {section}")
                    return False
            
            # Check ray tracing config (if present)
            if 'raytracing' in self.config:
                raytracing_config = self.config['raytracing']
                if 'simulation' in raytracing_config:
                    frequency = raytracing_config.get('simulation', {}).get('frequency')
                    if frequency is not None and frequency <= 0:
                        log_info("Invalid frequency config")
                        return False
            
            log_info("Config validation passed")
            return True
        except Exception as e:
            log_info(f"Config validation failed: {e}")
            return False
        
    def get_raytracing_config(self) -> Dict[str, Any]:
        """Get ray tracing config."""
        return self.get('raytracing', {})
    
    def get_region_config(self, region_name: str = None) -> Dict[str, Any]:
        """
        Get region config.
        
        Args:
            region_name: Region name; if None, return the default region
        
        Returns:
            Region config dict
        """
        if region_name is None:
            return self.get('default_region', {})
        
        regions = self.get('regions', {})
        if region_name in regions:
            return regions[region_name]
        else:
            log_info(f"Region config not found: {region_name}, using default region")
            return self.get('default_region', {})
    
    def get_scene_generation_config(self) -> Dict[str, Any]:
        """Get scene generation config."""
        return self.get('scene_generation', {})
    
    def get_data_storage_config(self) -> Dict[str, Any]:
        """Get data storage config."""
        return self.get('data_storage', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging config."""
        return self.get('logging', {})
    
    def get_materials_config(self) -> Dict[str, Any]:
        """Get materials config."""
        return self.get('materials', {})
    
    def get_mitsuba_config(self) -> Dict[str, Any]:
        """Get Mitsuba config."""
        return self.get('mitsuba_config', {})
    
    def list_available_regions(self) -> list:
        """List all available regions."""
        regions = self.get('regions', {})
        return list(regions.keys())
    
    def merge_with_args(self, args):
        """
        Merge command-line arguments with the config file.
        
        Args:
            args: argparse-parsed arguments
        
        Returns:
            Merged configuration dictionary
        """
        merged_config = {}
        
        # 区域配置
        region_config = self.get_region_config()
        merged_config['center_lat'] = getattr(args, 'center_lat', None) or region_config.get('center_lat')
        merged_config['center_lon'] = getattr(args, 'center_lon', None) or region_config.get('center_lon')
        merged_config['radius_km'] = getattr(args, 'radius_km', None) or region_config.get('radius_km')
        
        # 场景生成配置
        scene_config = self.get_scene_generation_config()
        merged_config['num_scenes'] = getattr(args, 'num_scenes', None) or scene_config.get('num_scenes')
        merged_config['size_x'] = getattr(args, 'size_x', None) or scene_config.get('size_x')
        merged_config['size_y'] = getattr(args, 'size_y', None) or scene_config.get('size_y')
        merged_config['generation_mode'] = getattr(args, 'generation_mode', None) or scene_config.get('generation_mode')
        
        # OSM重试配置
        osm_retry_config = scene_config.get('osm_retry', {})
        merged_config['max_osm_attempts'] = getattr(args, 'max_osm_attempts', None) or osm_retry_config.get('max_attempts')
        merged_config['search_radius_km'] = getattr(args, 'search_radius_km', None) or osm_retry_config.get('search_radius_km')
        
        # 数据存储配置
        storage_config = self.get_data_storage_config()
        merged_config['data_dir'] = getattr(args, 'data_dir', None) or storage_config.get('base_data_dir')
        
        # 日志配置
        logging_config = self.get_logging_config()
        merged_config['log_file'] = getattr(args, 'log_file', None) or logging_config.get('log_file')
        
        # 区域选择
        merged_config['region'] = getattr(args, 'region', None)
        
        return merged_config
    
    def get_min_samples_ratio(self, default: float = 0.5) -> float:
        return self.get('raytracing.simulation.min_samples_ratio', default)