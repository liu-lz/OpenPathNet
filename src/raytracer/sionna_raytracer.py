# Key: Mitsuba must be imported before any other libraries
import mitsuba as mi

# Safe to import other libraries afterwards
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure GPU before importing TensorFlow
def _configure_gpu_before_tf():
    """Configure GPU environment before importing TensorFlow"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    preconfigured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preconfigured is not None and preconfigured.strip() != "":
        if preconfigured.strip() == "-1":
            print("CUDA_VISIBLE_DEVICES preset to -1, using CPU")
        else:
            print(f"Detected preset CUDA_VISIBLE_DEVICES={preconfigured}, skip automatic GPU selection")
        return
    
    # Check if any GPU is available
    try:
        import subprocess
        result = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')
        
        gpus = []
        for line in result.strip().split('\n'):
            values = [x.strip() for x in line.split(',')]
            if len(values) >= 6:
                gpu_id = int(values[0])
                name = values[1]
                total_memory = int(values[2])
                used_memory = int(values[3])
                free_memory = int(values[4])
                utilization = float(values[5])
                
                gpus.append({
                    'id': gpu_id,
                    'name': name,
                    'total_memory': total_memory,
                    'used_memory': used_memory,
                    'free_memory': free_memory,
                    'utilization': utilization,
                    'free_percentage': (free_memory / total_memory) * 100 if total_memory > 0 else 0
                })
        
        if gpus:
            # Use only the first GPU (with the most free memory)
            sorted_gpus = sorted(gpus, key=lambda g: (-g['free_percentage'], g['utilization']))
            best_gpu = sorted_gpus[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu['id'])
            print(f"Using GPU {best_gpu['id']}: {best_gpu['name']} ({best_gpu['free_percentage']:.1f}% free)")
        else:
            print("No GPU detected, using CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except Exception as e:
        print(f"GPU detection failed, using CPU: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure TensorFlow before importing it
_configure_gpu_before_tf()

import tensorflow as tf
import psutil
import gc
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
import pickle

# Now configure TensorFlow GPU settings
def _setup_tensorflow_gpu():
    """Configure TensorFlow GPU settings"""
    try:
        # Get visible GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            # Enable memory growth for all GPU devices
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No available GPU devices detected")
    except Exception as e:
        print(f"Failed to configure TensorFlow GPU settings: {e}")

# Configure TensorFlow
_setup_tensorflow_gpu()

# Now it is safe to import Sionna components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray

from ..config_manager import ConfigManager
from ..utils.logging_utils import log_info
from ..utils.file_utils import ensure_directory_exists

class SionnaRaytracer:
    """Sionna-based ray tracer"""
    
    def __init__(self, config_manager: ConfigManager = None):
        """
        Initialize the Sionna ray tracer
        
        Args:
            config_manager: Config manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.raytracing_config = self.config_manager.get('raytracing', {})
        
        # GPU was configured at import time; only log status here
        self._log_gpu_status()
        
        # Get visible GPU devices
        self.available_gpus = tf.config.list_physical_devices('GPU')
        self.num_gpus = len(self.available_gpus)
        
        # Speed of light constant
        self.SPEED_OF_LIGHT = 299792458
        
        # Ray type mapping
        self.type_map = {0: "LoS", 1: "Reflected", 2: "Diffracted", 3: "Scattered"}
        
        self.min_samples_ratio = max(
            min(self.config_manager.get_min_samples_ratio(0.5), 1.0),
            0.1
        )
        
        log_info("Sionna ray tracer initialized (single GPU mode)")
    
    def _log_gpu_status(self):
        """Log GPU status"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                log_info(f"TensorFlow detected {len(gpus)} GPU device(s):")
                for i, gpu in enumerate(gpus):
                    log_info(f"  GPU {i}: {gpu.name}")
            else:
                log_info("TensorFlow detected no GPU devices, using CPU")
        except Exception as e:
            log_info(f"Failed to get GPU status: {e}")
    
    def _log_gpu_memory(self, message: str):
        """Log GPU memory usage"""
        gpu_memory = self._get_gpu_memory_usage()
        if gpu_memory:
            # Show only the first (current) GPU
            if len(gpu_memory) > 0:
                gpu = gpu_memory[0]
                log_info(f"{message} - GPU {gpu['id']}: {gpu['used']}MB/{gpu['total']}MB "
                        f"({gpu['usage_percent']:.1f}% used, {gpu['free']}MB free)")
        else:
            # Fallback to system memory
            memory = psutil.virtual_memory()
            log_info(f"{message} - System memory: {memory.used/(1024**3):.1f}GB/{memory.total/(1024**3):.1f}GB "
                    f"({memory.percent:.1f}% used)")
    
    def _get_gpu_memory_usage(self) -> List[Dict[str, Any]]:
        """Get GPU memory usage"""
        try:
            import subprocess
            # Query only currently visible GPUs
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            if visible_devices == "-1":
                return []
            # Query only currently visible GPUs
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=index,memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits',
                '--id=' + visible_devices
            ]).decode('utf-8')
            
            gpu_memory = []
            for line in result.strip().split('\n'):
                if line.strip():
                    values = [x.strip() for x in line.split(',')]
                    if len(values) >= 4:
                        gpu_id = int(values[0])
                        total_memory = int(values[1])
                        used_memory = int(values[2])
                        free_memory = int(values[3])
                        
                        gpu_memory.append({
                            'id': gpu_id,
                            'total': total_memory,
                            'used': used_memory,
                            'free': free_memory,
                            'usage_percent': (used_memory / total_memory) * 100 if total_memory > 0 else 0
                        })
            
            return gpu_memory
        except Exception as e:
            # Silently handle errors without logging
            return []
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Force multiple garbage-collection cycles
        for _ in range(5):
            gc.collect()
        # Try clearing TensorFlow cache
        try:
            if hasattr(tf, 'keras') and hasattr(tf.keras, 'backend'):
                tf.keras.backend.clear_session()
        except:
            pass
        # Run garbage collection again
        gc.collect()
        # Short sleep to allow the system to release memory
        time.sleep(0.5)
    
    def run_raytracing(self, scene_file: str, result_dir: str, 
                      current_scene: int = None, total_scenes: int = None) -> Dict[str, Any]:
        """
        Execute ray tracing for a single scene
        
        Args:
            scene_file: Scene file path
            result_dir: Directory to save results
            current_scene: Current scene index (starting from 1)
            total_scenes: Total number of scenes
        
        Returns:
            Ray tracing result dictionary
        """
        try:
            # Get scene name
            scene_name = os.path.basename(scene_file).replace('.xml', '')
            
            log_info(f"Start loading scene: {scene_file}")
            self._log_gpu_memory("Before scene load")
            
            # Check if scene file exists
            if not os.path.exists(scene_file):
                if current_scene and total_scenes:
                    log_info(f"✗ Scene {current_scene}/{total_scenes} failed: file not found")
                else:
                    log_info(f"✗ Scene failed: file not found")
                return {
                    'success': False,
                    'error': f'Scene file not found: {scene_file}',
                    'execution_time': 0,
                    'num_receivers': 0,
                    'num_paths': 0
                }
            
            # Clean up before loading scene
            self._aggressive_cleanup()
            
            # Load scene
            scene = load_scene(scene_file)
            self._log_gpu_memory("After scene load")
            
            # Configure scene parameters
            self._configure_scene(scene)
            self._log_gpu_memory("After scene configuration")
            
            # Generate receiver positions
            rx_positions = self._generate_receiver_positions(scene)
            self._log_gpu_memory("After receiver position generation")
            
            if len(rx_positions) == 0:
                return {
                    'success': False,
                    'error': 'No valid receiver positions found',
                    'execution_time': 0,
                    'num_receivers': 0,
                    'num_paths': 0
                }
            
            self._save_receiver_distribution(rx_positions, result_dir, scene_name)
            
            # Execute batch ray tracing (single GPU mode)
            start_time = time.time()
            ray_data = self._run_single_gpu_raytracing(scene, rx_positions, scene_file)
            execution_time = time.time() - start_time
            self._log_gpu_memory("After ray tracing")
            
            # Save stats before deleting variables
            num_receivers = len(rx_positions)
            num_paths = len(ray_data) if ray_data else 0
            
            # Clean up scene data
            del scene
            self._aggressive_cleanup()
            self._log_gpu_memory("After scene cleanup")
            
            # Save results
            output_files = self._save_results(ray_data, rx_positions, result_dir, scene_file)
            
            # Generate visualizations if enabled
            if self.raytracing_config.get('output', {}).get('save_visualizations', True):
                self._generate_visualizations(ray_data, rx_positions, result_dir)
            
            # Clean data
            del ray_data, rx_positions
            
            # Final cleanup after scene processing
            self._complete_scene_cleanup()
            self._log_gpu_memory("After final cleanup")
            
            log_info(f"Ray tracing completed, elapsed time: {execution_time:.2f}s")
            log_info(f"Stats: {num_receivers} receivers, {num_paths} paths")
            
            if current_scene and total_scenes:
                log_info(f"✓ Scene {current_scene}/{total_scenes} processed successfully!")
            else:
                log_info(f"✓ Scene processed successfully!")
            
            return {
                'success': True,
                'execution_time': execution_time,
                'num_receivers': num_receivers,
                'num_paths': num_paths,
                'output_files': output_files
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Ray tracing exception: {str(e)}\n{traceback.format_exc()}"
            log_info(error_msg)
            
            if current_scene and total_scenes:
                log_info(f"✗ Scene {current_scene}/{total_scenes} failed!")
            else:
                log_info(f"✗ Scene failed!")
            
            # Clean memory on exception as well
            self._complete_scene_cleanup()
            
            return {
                'success': False,
                'error': error_msg,
                'execution_time': 0,
                'num_receivers': 0,
                'num_paths': 0
            }
    
    def _configure_scene(self, scene):
        """Configure scene parameters"""
        sim_config = self.raytracing_config.get('simulation', {})
        antenna_config = self.raytracing_config.get('antenna', {})
        tx_config = self.raytracing_config.get('transmitter', {})
        
        # Set frequency - ConfigManager already handled type conversion
        frequency = sim_config.get('frequency', 3.66e9)
        scene.frequency = frequency
        scene.synthetic_array = sim_config.get('synthetic_array', True)
        
        # Configure transmitter array (aligned with gen.py)
        scene.tx_array = PlanarArray(
            num_rows=1,                # Rows = 1
            num_cols=1,                # Columns = 1
            vertical_spacing=0.5,      # Vertical spacing 0.5 m
            horizontal_spacing=0.5,    # Horizontal spacing 0.5 m
            pattern="iso",             # Isotropic radiation
            polarization="V"           # Vertical polarization
        )
        # Configure receiver array (aligned with gen.py)
        scene.rx_array = PlanarArray(
            num_rows=1,                # Rows = 1
            num_cols=1,                # Columns = 1
            vertical_spacing=0.5,      # Vertical spacing 0.5 m
            horizontal_spacing=0.5,
            pattern="dipole",          # Dipole radiation
            polarization="V"           # Vertical polarization
        )
        # Add transmitter (aligned with gen.py)
        tx_position = tx_config.get('position', [0, 0, 30])
        tx_orientation = tx_config.get('orientation', [-1 * (210 - 90) / 180 * np.pi, 0, 0])
        tx = Transmitter(
            name="tx",                 # Transmitter name
            position=tx_position,      # High position (height 30 m)
            orientation=tx_orientation # 210° azimuth converted to Sionna format
        )
        scene.add(tx)
        
        log_info(f"Scene configured: frequency={frequency/1e9:.2f}GHz, transmitter position={tx_position}")
    
    def _generate_receiver_positions(self, scene) -> np.ndarray:
        """Generate receiver position grid"""
        grid_config = self.raytracing_config.get('receiver_grid', {})
        
        grid_size = grid_config.get('grid_size', 128)
        area_size = grid_config.get('area_size', 128)
        cm_center = grid_config.get('center', [0, 0, 1.5])
        
        log_info(f"Generate receiver grid: {grid_size}x{grid_size}, coverage: {area_size}m x {area_size}m")
        
        # Compute region boundaries (aligned with gen.py)
        min_x = cm_center[0] - area_size/2
        max_x = cm_center[0] + area_size/2
        min_y = cm_center[1] - area_size/2
        max_y = cm_center[1] + area_size/2
        
        # Generate uniformly distributed coordinates
        x_coords = np.linspace(min_x, max_x, grid_size)
        y_coords = np.linspace(min_y, max_y, grid_size)
        
        # Generate grid center points
        total_points = grid_size * grid_size
        all_positions = np.zeros((total_points, 3))
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                x = x_coords[i]
                y = y_coords[j]
                z = 1  # Height fixed at 1 m above ground
                all_positions[idx] = [x, y, z]
                idx += 1
        
        log_info(f"Generated {len(all_positions)} uniformly distributed candidate receiver positions")
        log_info(f"Receiver area range: X({min_x:.1f}m to {max_x:.1f}m), Y({min_y:.1f}m to {max_y:.1f}m)")
        
        # Filter indoor points (aligned with gen.py)
        if grid_config.get('indoor_filter', True):
            log_info("Batch filtering outdoor receiver points...")
            mi_scene = scene.mi_scene
            # Use the same batching approach as gen.py
            batch_size = 10000
            outdoor_mask = np.zeros(len(all_positions), dtype=bool)
            
            for i in tqdm(range(0, len(all_positions), batch_size), desc="Batch outdoor test"):
                end_idx = min(i + batch_size, len(all_positions))
                batch_positions = all_positions[i:end_idx]
                batch_mask = self._is_outdoor_points_batch(batch_positions, mi_scene)
                outdoor_mask[i:end_idx] = batch_mask
                
                outdoor_count = np.sum(outdoor_mask[:end_idx])
                total_processed = end_idx
                log_info(f"Processed {total_processed}/{len(all_positions)} points, "
                        f"found {outdoor_count} outdoor points ({outdoor_count/total_processed*100:.2f}%)")
            
            # Keep only outdoor positions
            rx_positions = all_positions[outdoor_mask]
            log_info(f"Found {len(rx_positions)} outdoor receiver points "
                    f"({len(rx_positions)/len(all_positions)*100:.2f}% of total)")
        else:
            rx_positions = all_positions
            log_info("Skipping indoor/outdoor filtering; using all positions")
        
        return rx_positions
    
    def _is_outdoor_points_batch(self, positions: np.ndarray, mi_scene) -> np.ndarray:
        """Batch check whether receiver points are outdoor"""
        try:
            # Create upward direction vectors and replicate to match positions
            directions = np.tile(np.array([0, 0, 1]), (len(positions), 1))
            # Build rays (from each candidate position upward)
            rays = mi.Ray3f(
                mi.Vector3f(positions),    # Ray origin (candidate positions)
                mi.Vector3f(directions)    # Ray direction (upward)
            )
            # Intersect rays with scene
            si = mi_scene.ray_intersect(rays)
            # If the upward ray hits nothing, the point is outdoor
            outdoor_mask = ~si.is_valid()
            return outdoor_mask.numpy()
        except Exception as e:
            import traceback
            log_info("Error occurred while batch-checking outdoor points:")
            log_info(traceback.format_exc())
            return np.zeros(len(positions), dtype=bool)
    
    def _run_single_gpu_raytracing(self, scene, rx_positions: np.ndarray, scene_file: str) -> List[Dict[str, Any]]:
        """Single-GPU ray tracing (adaptive error handling)"""
        batch_config = self.raytracing_config.get('batch_processing', {})
        sim_config = self.raytracing_config.get('simulation', {})
        tx_config = self.raytracing_config.get('transmitter', {})
        
        # Get configuration parameters
        rx_batch_size = batch_config.get('rx_batch_size', 2000)
        samples = sim_config.get('ray_samples', 100000)
        batch_pause = batch_config.get('batch_pause_seconds', 1.0)
        tx_power = tx_config.get('power_watts', 1)
        
        total_rx = len(rx_positions)
        num_batches = (total_rx + rx_batch_size - 1) // rx_batch_size
        all_ray_data = []
        
        # Adaptive parameters
        current_batch_size = rx_batch_size
        current_samples = samples
        min_samples = max(int(samples * self.min_samples_ratio), 1)
        # Add periodic deep cleanup config
        DEEP_CLEANUP_INTERVAL = 3  # Deep cleanup every 3 batches
    
        batch_idx = 0
        rx_processed = 0
        
        while rx_processed < total_rx:
            # Dynamically compute current batch
            batch_start = rx_processed
            batch_end = min(rx_processed + current_batch_size, total_rx)
            batch_rx_count = batch_end - batch_start
            batch_idx += 1
            
            log_info(f"Processing batch {batch_idx}: receivers {batch_start} to {batch_end-1} (total {batch_rx_count})")
            log_info(f"Current params: batch_size={current_batch_size}, samples={current_samples}")
            self._log_gpu_memory(f"Before batch {batch_idx}")
            
            success = False
            retry_count = 0
            max_retries = 3

            if success:
                # Check if deep cleanup is needed
                if batch_idx % DEEP_CLEANUP_INTERVAL == 0:
                    self._deep_batch_cleanup(scene, f"Before deep cleanup of batch {batch_idx}")
                    time.sleep(max(batch_pause, 0.5))
    
            
            while not success and retry_count < max_retries:
                try:
                    if batch_idx > 1 or retry_count > 0:
                        self._cleanup_after_batch(scene, f"Before retry of batch {batch_idx}")
                        time.sleep(batch_pause)
                    
                    # Process current batch
                    batch_results = self._process_single_batch(
                        scene, rx_positions, batch_start, batch_rx_count, current_samples, tx_power
                    )
                    all_ray_data.extend(batch_results)
                    self._log_gpu_memory(f"After batch {batch_idx}")
                    
                    success = True
                    rx_processed = batch_end
                    
                    # Increase batch size after success (capped at original)
                    if current_batch_size < rx_batch_size:
                        current_batch_size = min(current_batch_size + 200, rx_batch_size)
                        log_info(f"Batch succeeded, increased batch size to: {current_batch_size}")
                    
                    if batch_pause > 0:
                        time.sleep(batch_pause)
                    
                    # Restore samples after success
                    if current_samples < samples:
                        restore_step = max(samples // 10, 1)
                        current_samples = min(current_samples + restore_step, samples)
                        log_info(f"Batch succeeded, restored samples to: {current_samples}")
                    
                    self._cleanup_after_batch(scene, f"After batch {batch_idx} cleanup")
            
                except tf.errors.ResourceExhaustedError as e:
                    retry_count += 1
                    log_info(f"Batch {batch_idx} out of memory (attempt {retry_count}/{max_retries}): reducing parameters and retrying")
                    
                    self._cleanup_after_batch(scene, f"Cleanup after OOM in batch {batch_idx}")
                    time.sleep(batch_pause * 2)
                    
                    if retry_count < max_retries:
                        current_batch_size = max(100, current_batch_size // 2)
                        log_info(f"OOM, reducing batch size to: {current_batch_size}")
                        current_samples = max(min_samples, current_samples // 2)
                        log_info(f"OOM, reducing samples to: {current_samples}")
                        batch_end = min(batch_start + current_batch_size, total_rx)
                        batch_rx_count = batch_end - batch_start
                        log_info(f"Retrying batch {batch_idx}: receiver range {batch_start}-{batch_end-1}")
                    else:
                        log_info(f"Batch {batch_idx} failed after {max_retries} retries, skipping this batch")
                        rx_processed = batch_end
                        success = True
                
                except Exception as e:
                    retry_count += 1
                    log_info(f"Batch {batch_idx} exception (attempt {retry_count}/{max_retries}): {str(e)}")
                    
                    self._cleanup_after_batch(scene, f"Cleanup after exception in batch {batch_idx}")
                    time.sleep(batch_pause)
                    
                    if retry_count >= max_retries:
                        log_info(f"Batch {batch_idx} failed after {max_retries} retries, skipping this batch")
                        rx_processed = batch_end
                        success = True

            if success and batch_idx % DEEP_CLEANUP_INTERVAL == 0 and rx_processed < total_rx:
                scene = self._deep_batch_cleanup(scene, scene_file, f"After deep cleanup of batch {batch_idx}")
                time.sleep(max(batch_pause, 0.5))

        self._cleanup_after_batch(scene, "After all batches completed")

        log_info(f"Ray tracing completed, processed {len(all_ray_data)} paths")
        return all_ray_data
    
    def _process_single_batch(self, scene, rx_positions: np.ndarray, batch_start: int,
                             batch_rx_count: int, samples: int, tx_power: float) -> List[Dict[str, Any]]:
        """Process a single batch"""
        # Add receivers for the current batch
        for i in range(batch_rx_count):
            rx = Receiver(
                name=f"rx{batch_start + i}",
                position=rx_positions[batch_start + i],
                orientation=[0, 0, 0]
            )
            scene.add(rx)
        
        # Execute ray tracing computation
        paths = scene.compute_paths(
            max_depth=self.raytracing_config.get('simulation', {}).get('max_depth', 1),
            diffraction=self.raytracing_config.get('simulation', {}).get('diffraction', True),
            scattering=self.raytracing_config.get('simulation', {}).get('scattering', True),
            scat_keep_prob=self.raytracing_config.get('simulation', {}).get('scat_keep_prob', 0.001),
            num_samples=samples,
        )
        
        # Process path data
        ray_data = paths.to_dict()
        ray_dict_numpy = {
            "a": ray_data["a"].numpy().squeeze(),
            "tau": ray_data["tau"].numpy().squeeze(),
            "types": ray_data["types"].numpy().squeeze(),
            "mask": ray_data["mask"].numpy().squeeze(),
            "phi_r": ray_data["phi_r"].numpy().squeeze(),
            "phi_t": ray_data["phi_t"].numpy().squeeze(),
            "theta_r": ray_data["theta_r"].numpy().squeeze(),
            "theta_t": ray_data["theta_t"].numpy().squeeze(),
            "min_tau": ray_data["min_tau"].numpy().squeeze()
        }
        
        # Release raw path data immediately
        del paths, ray_data
        self._aggressive_cleanup()
        
        # Collect Top-5 rays for the current batch
        batch_results = self._process_batch_results(
            ray_dict_numpy, rx_positions, batch_start, batch_rx_count, tx_power
        )
        
        # Clean batch data
        del ray_dict_numpy
        self._aggressive_cleanup()
        
        return batch_results
    
    def _process_batch_results(self, ray_dict_numpy: Dict, rx_positions: np.ndarray, 
                             batch_start: int, batch_rx_count: int, tx_power: float) -> List[Dict[str, Any]]:
        """Process batch results"""
        results = []
        type_map = {0: "LoS", 1: "Reflected", 2: "Diffracted", 3: "Scattered"}
        
        for i in range(batch_rx_count):
            rx_idx = i  # Index within the current batch
            global_rx_idx = batch_start + i  # Global index
            
            # Get current receiver position
            rx_position = rx_positions[global_rx_idx]
            
            # Get current receiver mask
            current_mask = ray_dict_numpy["mask"][rx_idx]
            
            # Skip receivers without rays
            if not np.any(current_mask):
                continue
            
            # Get all valid ray indices
            valid_indices = np.where(current_mask)[0]
            
            # Compute received power: P_rx = P_tx * |a|^2
            if len(valid_indices) > 0:
                channel_gains = np.square(np.abs(ray_dict_numpy["a"][rx_idx, valid_indices]))
                sorted_indices = valid_indices[np.argsort(-channel_gains)]
                top_indices = sorted_indices[:min(5, len(sorted_indices))]
                
                for path_idx in top_indices:
                    # Convert angles from radians to degrees and normalize
                    phi_r, theta_r = self._normalize_angles(
                        ray_dict_numpy["phi_r"][rx_idx, path_idx],
                        ray_dict_numpy["theta_r"][rx_idx, path_idx]
                    )
                    phi_t, theta_t = self._normalize_angles(
                        ray_dict_numpy["phi_t"][rx_idx, path_idx],
                        ray_dict_numpy["theta_t"][rx_idx, path_idx]
                    )
                    
                    # Correct delay: convert relative delay to absolute propagation time
                    absolute_delay = ray_dict_numpy["min_tau"][rx_idx] + ray_dict_numpy["tau"][rx_idx, path_idx]
                    
                    result_dict = {
                        'rx_id': global_rx_idx,
                        'type': type_map[int(ray_dict_numpy["types"][path_idx])],
                        'channel_gain': channel_gains[np.where(valid_indices == path_idx)][0],
                        'tau': absolute_delay,
                        'freq': 3.66e9,
                        'rx_coord': rx_position.tolist(),
                        'phi_r': phi_r,
                        'phi_t': phi_t,
                        'theta_r': theta_r,
                        'theta_t': theta_t
                    }
                    
                    if self.raytracing_config.get('simulation', {}).get('save_gain', False):
                        result_dict['a'] = ray_dict_numpy["a"][rx_idx, path_idx]
                        result_dict['relative_tau'] = ray_dict_numpy["tau"][rx_idx, path_idx]
                    
                    results.append(result_dict)
        
        return results
    
    def _normalize_angles(self, phi, theta):
        """Convert radians to degrees and normalize to target ranges"""
        # Radians to degrees
        phi_deg = np.degrees(phi)
        theta_deg = np.degrees(theta)
        
        # Ensure azimuth is in [-180, 180]
        phi_deg = ((phi_deg + 180) % 360) - 180
        
        # Ensure elevation is in [0, 180]
        theta_deg = theta_deg % 360
        if theta_deg > 180:
            theta_deg = 360 - theta_deg
        
        return phi_deg, theta_deg
    
    def _save_results(self, ray_data: List[Dict[str, Any]], rx_positions: np.ndarray, 
                     result_dir: str, scene_file: str) -> List[str]:
        """Save ray tracing results"""
        output_config = self.raytracing_config.get('output', {})
        output_files = []
        
        # Check if there is data
        if not ray_data:
            log_info("Warning: No ray tracing data, creating empty result file")
            # Create an empty DataFrame
            df = pd.DataFrame()
        else:
            # Convert to DataFrame
            df = pd.DataFrame(ray_data)
        
        # Save as CSV
        if output_config.get('save_csv', True):
            csv_file = os.path.join(result_dir, 'raytracing_results.csv')
            df.to_csv(csv_file, index=False)
            output_files.append(csv_file)
            log_info(f"CSV result saved: {csv_file}")
        
        # Save as Pickle
        if output_config.get('save_pickle', True):
            pickle_file = os.path.join(result_dir, 'raytracing_results.pkl')
            df.to_pickle(pickle_file)
            output_files.append(pickle_file)
            log_info(f"Pickle result saved: {pickle_file}")
        
        # Save in DeepMIMO format
        if output_config.get('save_deepmimo', True):
            deepmimo_file = os.path.join(result_dir, 'deepmimo_format.npy')
            tx_location = self.raytracing_config.get('transmitter', {}).get('position', [0, 0, 30])
            self._save_to_deepmimo_format(df, tx_location, deepmimo_file)
            output_files.append(deepmimo_file)
            log_info(f"DeepMIMO format saved: {deepmimo_file}")
        
        return output_files
    
    def _save_to_deepmimo_format(self, df: pd.DataFrame, tx_location: List[float], output_file: str):
        """Save in DeepMIMO format"""
        try:
            log_info(f"Saving data in DeepMIMO format: {output_file}")
            # Check if DataFrame is empty or missing required columns
            if df.empty or 'rx_id' not in df.columns:
                log_info("Warning: No valid ray tracing data, saving empty DeepMIMO file")
                # Create empty result
                result = {
                    'user': [],
                    'location': tx_location
                }
                np.save(output_file, result)
                log_info("Saved empty DeepMIMO file")
                return
            
            # Get all unique receiver IDs
            rx_ids = df['rx_id'].unique()
            rx_ids.sort()
            
            # Build result dictionary
            result = {
                'user': [],
                'location': tx_location  # Transmitter position
            }
            
            # Organize data for each receiver
            for rx_id in rx_ids:
                # Get all rays for the current receiver
                rx_rays = df[df['rx_id'] == rx_id]
                
                # If no ray data, skip this receiver
                if len(rx_rays) == 0:
                    continue
                    
                # Receiver coordinates
                rx_coord = rx_rays.iloc[0]['rx_coord']
                
                # Safely handle power data to avoid zero/negative values
                channel_gains = rx_rays['channel_gain'].values
                safe_gains = np.where(channel_gains <= 0, 1e-15, channel_gains)
                
                # Build paths data dictionary
                paths_data = {
                    'channel_gain': safe_gains,
                    'ToA': np.array(rx_rays['tau']),
                    'DoA_theta': np.array(rx_rays['theta_r']),
                    'DoA_phi': np.array(rx_rays['phi_r']),
                    'num_paths': len(rx_rays)
                }
                
                # Build user data
                user_data = {
                    'location': np.array(rx_coord),  # Receiver coordinates
                    'paths': paths_data              # Path data
                }
                
                # Add to result
                result['user'].append(user_data)
            
            # Save as numpy file
            np.save(output_file, result)
            log_info(f"Successfully saved {len(result['user'])} receiver(s) to DeepMIMO format")
        except Exception as e:
            import traceback
            log_info(f"Failed to save DeepMIMO format: {e}")
            log_info(traceback.format_exc())
    
    def _generate_visualizations(self, ray_data: List[Dict[str, Any]], 
                                rx_positions: np.ndarray, result_dir: str):
        """Generate visualization plots"""
        try:
            if not ray_data:
                log_info("Warning: No ray tracing data, skipping visualization")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(ray_data)
            
            # Check required columns
            required_columns = ['channel_gain', 'tau', 'type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                log_info(f"Warning: Missing required columns {missing_columns}, skipping visualization")
                return
            
            # Power distribution plot (channel gain in dB)
            channel_gains = df['channel_gain']
            valid_gains = channel_gains[channel_gains > 0]
            
            if len(valid_gains) > 0:
                gain_db = 10 * np.log10(valid_gains)
                
                plt.figure(figsize=(10, 6))
                plt.hist(gain_db, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Channel Gain (dB)')
                plt.ylabel('Number of Paths')
                plt.title('Channel Gain Distribution')
                plt.grid(True, alpha=0.3)
                gain_plot = os.path.join(result_dir, 'channel_gain_distribution.png')
                plt.savefig(gain_plot, dpi=300, bbox_inches='tight')
                plt.close()
                
                log_info(f"Channel gain distribution plot saved: {gain_plot}")
            else:
                log_info("Warning: No valid channel gain data, skipping power distribution plot")
            
            # ToA distribution plot
            tau_values = df['tau']
            valid_tau = tau_values[np.isfinite(tau_values) & (tau_values > 0)]
            
            if len(valid_tau) > 0:
                tau_ns = valid_tau * 1e9  # Convert to nanoseconds
                plt.figure(figsize=(10, 6))
                plt.hist(tau_ns, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('ToA (ns)')
                plt.ylabel('Number of Paths')
                plt.title('ToA Distribution')
                plt.grid(True, alpha=0.3)
                delay_plot = os.path.join(result_dir, 'ToA_distribution.png')
                plt.savefig(delay_plot, dpi=300, bbox_inches='tight')
                plt.close()
                
                log_info(f"ToA distribution plot saved: {delay_plot}")
            else:
                log_info("Warning: No valid ToA data, skipping ToA distribution plot")
            
            # Path type distribution plot
            path_types = df['type']
            type_counts = path_types.value_counts()
            
            if len(type_counts) > 0:
                plt.figure(figsize=(8, 6))
                type_counts.plot(kind='bar')
                plt.xlabel('Path Type')
                plt.ylabel('Number of Paths')
                plt.title('Path Type Distribution')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                type_plot = os.path.join(result_dir, 'path_type_distribution.png')
                plt.savefig(type_plot, dpi=300, bbox_inches='tight')
                plt.close()
                
                log_info(f"Path type distribution plot saved: {type_plot}")
            
            log_info("Visualization generation completed")
        except Exception as e:
            import traceback
            log_info(f"Failed to generate visualizations: {e}")
            log_info(traceback.format_exc())
    
    def _complete_scene_cleanup(self):
        """Thorough memory cleanup after scene processing"""
        try:
            # Multiple rounds of garbage collection
            for _ in range(10):
                gc.collect()
            
            # Clean TensorFlow sessions/graphs
            try:
                if hasattr(tf, 'keras') and hasattr(tf.keras, 'backend'):
                    tf.keras.backend.clear_session()
            except:
                pass
            
            # Attempt to clean TensorFlow memory
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, False)
                            tf.config.experimental.set_memory_growth(gpu, True)
                        except:
                            pass
            except:
                pass
            
            # Clean Mitsuba cache if available
            try:
                if hasattr(mi, 'flush') or hasattr(mi, 'clear_cache'):
                    if hasattr(mi, 'flush'):
                        mi.flush()
                    if hasattr(mi, 'clear_cache'):
                        mi.clear_cache()
            except:
                pass
            
            # Additional garbage collection
            for _ in range(5):
                gc.collect()
            
            # Give system time to release memory
            time.sleep(2.0)
            
            log_info("Scene memory cleanup completed")
        except Exception as e:
            log_info(f"Error during scene memory cleanup: {e}")
    
    def _save_receiver_distribution(self, rx_positions: np.ndarray, result_dir: str, scene_name: str):
        """Save outdoor receiver distribution plot"""
        try:
            if rx_positions.size == 0:
                self.logger.warning(f"Scene {scene_name} has no outdoor receiver points, skip distribution plot")
                return
            
            plt.figure(figsize=(10, 10))
            plt.scatter(rx_positions[:, 0], rx_positions[:, 1],
                        s=35, c='#1f77b4', alpha=0.6, edgecolors='none')
            plt.title(f"Outdoor Receiver Distribution - {scene_name}", fontsize=16)
            plt.xlabel('X (m)', fontsize=14)
            plt.ylabel('Y (m)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            
            output_path = os.path.join(result_dir, 'outdoor_receivers.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            log_info(f"Outdoor receiver distribution saved: {output_path}")
        except Exception as e:
            log_info(f"Failed to save outdoor receiver distribution: {e}")
    
    def _cleanup_after_batch(self, scene, reason: str = None):
        """Unified cleanup after a batch"""
        try:
            for rx_name in list(scene.receivers.keys()):
                scene.remove(rx_name)
            if hasattr(scene, '_refresh'):
                scene._refresh()
        except Exception as exc:
            log_info(f"Error while cleaning receivers after batch: {exc}")
        self._reset_tf_memory()
        self._aggressive_cleanup()
        if reason:
            self._log_gpu_memory(reason)

    def _deep_batch_cleanup(self, scene, scene_file: str, reason: str = None):
        """Deep cleanup"""
        try:
            del scene
        except Exception:
            pass
        self._reset_tf_memory()
        self._aggressive_cleanup()
        new_scene = load_scene(scene_file)
        self._configure_scene(new_scene)
        if reason:
            self._log_gpu_memory(reason)
        return new_scene

    def _reset_tf_memory(self):
        """Reset TensorFlow memory state"""
        try:
            if hasattr(tf.config.experimental, "reset_memory_stats"):
                for device in tf.config.list_logical_devices('GPU'):
                    tf.config.experimental.reset_memory_stats(device.name)
            elif hasattr(tf, "keras") and hasattr(tf.keras, "backend"):
                tf.keras.backend.clear_session()
        except Exception as exc:
            log_info(f"Error resetting TensorFlow memory state: {exc}")
    
    def shutdown(self):
        """Explicitly release TensorFlow resources (for GPU/CPU execution)"""
        try:
            self._complete_scene_cleanup()
        finally:
            self._reset_tf_memory()
            try:
                if hasattr(tf, "keras") and hasattr(tf.keras, "backend"):
                    tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
            time.sleep(0.2)