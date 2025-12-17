import os
import glob
import time
import json
from typing import List, Dict, Any, Optional
import copy
from multiprocessing import get_context
import subprocess
import queue as queue_module

from ..config_manager import ConfigManager
from ..utils.logging_utils import log_info
from ..utils.file_utils import ensure_directory_exists

def _scene_worker(config_data: Dict[str, Any],
                  scene_info: Dict[str, Any],
                  results_dir: str,
                  current_scene: int,
                  total_scenes: int,
                  gpu_id: Optional[str],
                  result_queue):
    raytracer = None
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        
        from .sionna_raytracer import SionnaRaytracer
        cfg = ConfigManager(config_data=config_data)
        raytracer = SionnaRaytracer(cfg)
        
        scene_name = scene_info['scene_name']
        result_dir = os.path.join(results_dir, scene_name)
        ensure_directory_exists(result_dir)
        
        result = raytracer.run_raytracing(
            scene_info['scene_file'], result_dir, current_scene, total_scenes
        )
        result.update({
            'scene_info': scene_info,
            'execution_time': result.get('execution_time', 0),
            'result_dir': result_dir
        })
    except Exception as exc:
        result = {
            'success': False,
            'error': str(exc),
            'scene_info': scene_info,
            'execution_time': 0,
            'result_dir': None
        }
    finally:
        if raytracer is not None:
            try:
                raytracer.shutdown()
            except Exception:
                pass
    result_queue.put(result)

class RaytracingManager:
    """Ray tracing manager responsible for batch execution of ray tracing tasks"""
    
    def __init__(self, config_manager: ConfigManager = None):
        """
        Initialize the ray tracing manager
        
        Args:
            config_manager: Config manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # 获取配置
        self.raytracing_config = self.config_manager.get('raytracing', {})
        self.storage_config = self.config_manager.get_data_storage_config()
        
        # 设置路径
        self.base_data_dir = self.storage_config['base_data_dir']
        self.scenes_dir = os.path.join(self.base_data_dir, self.storage_config['scenes_subdir'])
        
        results_subdir = self.raytracing_config.get('output', {}).get('results_subdir', 'raytracing_results')
        self.results_dir = os.path.join(self.base_data_dir, results_subdir)
        ensure_directory_exists(self.results_dir)
        
        # 射线追踪器实例
        self.config_snapshot = copy.deepcopy(self.config_manager.config)
        self.min_samples_ratio = self.config_manager.get_min_samples_ratio()
        self.inter_scene_pause = self.raytracing_config.get('batch_processing', {}).get('inter_scene_pause', 0.0)
        timeout_cfg = self.raytracing_config.get('timeout_per_scene')
        self.timeout_per_scene = timeout_cfg if isinstance(timeout_cfg, (int, float)) and timeout_cfg > 0 else None
        if self.timeout_per_scene:
            log_info(f"Single-scene timeout: {self.timeout_per_scene:.0f}s")
        log_info(f"Minimum samples ratio: {self.min_samples_ratio:.2f}")
        log_info(f"Inter-scene pause: {self.inter_scene_pause:.2f}s")
        
        # 初始化热图生成器
        self.heatmap_generator = None
        if self.config_manager.get('visualization', {}).get('enabled', False):
            try:
                from ..visualization import HeatmapGenerator
                self.heatmap_generator = HeatmapGenerator(self.config_manager.config)
                log_info("Heatmap generator enabled")
            except ImportError as e:
                log_info(f"Failed to import heatmap generator: {e}")
    
    def discover_scenes(self) -> List[Dict[str, Any]]:
        """
        Automatically discover scene files
        
        Returns:
            List of scene file info
        """
        scenes = []
        
        if not os.path.exists(self.scenes_dir):
            log_info(f"Scene directory does not exist: {self.scenes_dir}")
            return scenes
        
        # 查找所有场景目录
        scene_pattern = self.raytracing_config.get('input', {}).get('scene_file_pattern', 'scene.xml')
        
        for scene_dir in os.listdir(self.scenes_dir):
            scene_path = os.path.join(self.scenes_dir, scene_dir)
            
            if os.path.isdir(scene_path):
                scene_file = os.path.join(scene_path, scene_pattern)
                
                if os.path.exists(scene_file):
                    # 解析场景信息
                    scene_info = self._parse_scene_info(scene_dir, scene_file)
                    if scene_info:
                        scenes.append(scene_info)
                else:
                    log_info(f"Scene file does not exist: {scene_file}")
        
        log_info(f"Found {len(scenes)} available scenes")
        return scenes
    
    def _parse_scene_info(self, scene_dir_name: str, scene_file: str) -> Optional[Dict[str, Any]]:
        """Parse scene info"""
        try:
            # Parse coordinate info from directory name
            if scene_dir_name.startswith("scene_"):
                coords_part = scene_dir_name[6:]  # Remove "scene_" prefix
                
                # Handle possible suffixes
                if "_attempt_" in coords_part:
                    coords_part = coords_part.split("_attempt_")[0]
                elif "_random" in coords_part:
                    coords_part = coords_part.replace("_random", "")
                
                parts = coords_part.split("_")
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    
                    return {
                        'scene_name': scene_dir_name,
                        'scene_file': scene_file,
                        'scene_dir': os.path.dirname(scene_file),
                        'latitude': lat,
                        'longitude': lon,
                        'status': 'pending'
                    }
        except Exception as e:
            log_info(f"Failed to parse scene info {scene_dir_name}: {e}")
        
        return None
    
    def run_single_raytracing(self, scene_info: Dict[str, Any], current_scene: int = None, total_scenes: int = None) -> Dict[str, Any]:
        gpu_ids = self._determine_gpu_ids()
        assigned_gpu = gpu_ids[0] if gpu_ids else None
        return self._launch_scene_process(scene_info, current_scene, total_scenes, assigned_gpu)

    def _launch_scene_process(self, scene_info, current_scene, total_scenes, gpu_id):
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_scene_worker,
            args=(self.config_snapshot, scene_info, self.results_dir, current_scene or 1,
                  total_scenes or 1, gpu_id, result_queue)
        )
        proc.start()
        start_time = time.time()
        result = None
        try:
            if self.timeout_per_scene:
                proc.join(self.timeout_per_scene)
                if proc.is_alive():
                    log_info(f"Scene {scene_info['scene_name']} exceeded {self.timeout_per_scene:.0f}s, terminating and skipping")
                    proc.terminate()
                    proc.join(timeout=5)
                    result = {
                        'success': False,
                        'error': f"Scene timed out (>{self.timeout_per_scene:.0f}s)",
                        'scene_info': scene_info,
                        'execution_time': time.time() - start_time,
                        'result_dir': None
                    }
            if result is None:
                proc.join()
                if proc.exitcode != 0:
                    result = {
                        'success': False,
                        'error': f"Child process exit code {proc.exitcode}",
                        'scene_info': scene_info,
                        'execution_time': 0,
                        'result_dir': None
                    }
                else:
                    try:
                        result = result_queue.get(timeout=5)
                    except queue_module.Empty:
                        result = {
                            'success': False,
                            'error': 'No result received from child process',
                            'scene_info': scene_info,
                            'execution_time': time.time() - start_time,
                            'result_dir': None
                        }
        finally:
            if result and result['success'] and self.heatmap_generator:
                self._generate_heatmaps_for_scene(scene_info['scene_name'], result.get('result_dir'))
        if result is None:
            result = {
                'success': False,
                'error': 'Unknown error: no result',
                'scene_info': scene_info,
                'execution_time': time.time() - start_time,
                'result_dir': None
            }
        return result

    def run_batch_raytracing(self, scenes: List[Dict[str, Any]] = None, 
                           max_workers: int = None) -> List[Dict[str, Any]]:
        """Batch execute ray tracing"""
        if scenes is None:
            scenes = self.discover_scenes()
        
        if not scenes:
            log_info("No available scene files found")
            return []
        
        total_scenes = len(scenes)
        max_workers = max_workers or self.raytracing_config.get('max_parallel_jobs', 1)
        log_info(f"Starting batch ray tracing, total scenes: {total_scenes}, concurrency limit: {max_workers}")
        
        gpu_ids = self._determine_gpu_ids()
        concurrency = 1
        gpu_ids = gpu_ids if gpu_ids else [None]
        log_info(f"Available GPUs: {gpu_ids}, fixed concurrency per round: {concurrency}")
        
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        pending_results = {}
        active_processes = []
        results = []
        successful_count = failed_count = 0
        gpu_index = 0

        def finalize_process(process_info, timed_out=False):
            nonlocal successful_count, failed_count
            proc, scene = process_info['process'], process_info['scene']
            start_time = process_info['start_time']
            if timed_out:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                self._drain_result_queue(result_queue, pending_results)
                result = {
                    'success': False,
                    'error': f"Scene timed out (>{self.timeout_per_scene:.0f}s)" if self.timeout_per_scene else 'Scene timed out',
                    'scene_info': scene,
                    'execution_time': time.time() - start_time,
                    'result_dir': None
                }
                failed_count += 1
            else:
                proc.join()
                if proc.exitcode != 0:
                    result = {
                        'success': False,
                        'error': f"Child process exit code {proc.exitcode}",
                        'scene_info': scene,
                        'execution_time': 0,
                        'result_dir': None
                    }
                    failed_count += 1
                else:
                    result = self._fetch_result_for_scene(result_queue, pending_results, scene['scene_name'])
                    if result['success']:
                        successful_count += 1
                        if self.heatmap_generator:
                            self._generate_heatmaps_for_scene(scene['scene_name'], result.get('result_dir'))
                    else:
                        failed_count += 1
            results.append(result)
            active_processes.remove(process_info)
            if self.inter_scene_pause > 0:
                time.sleep(self.inter_scene_pause)

        def process_active_once():
            self._drain_result_queue(result_queue, pending_results)
            for info in list(active_processes):
                proc = info['process']
                if self.timeout_per_scene and proc.is_alive():
                    elapsed = time.time() - info['start_time']
                    if elapsed >= self.timeout_per_scene:
                        log_info(f"Scene {info['scene']['scene_name']} exceeded {self.timeout_per_scene:.0f}s, terminating and skipping")
                        finalize_process(info, timed_out=True)
                        return True
                if not proc.is_alive():
                    finalize_process(info)
                    return True
            return False

        def wait_for_available_slot():
            while len(active_processes) >= concurrency:
                if not process_active_once():
                    time.sleep(0.2)

        for idx, scene in enumerate(scenes, 1):
            wait_for_available_slot()
            assigned_gpu = gpu_ids[gpu_index % len(gpu_ids)] if gpu_ids else None
            gpu_index += 1
            proc = ctx.Process(
                target=_scene_worker,
                args=(self.config_snapshot, scene, self.results_dir, idx, total_scenes, assigned_gpu, result_queue)
            )
            proc.start()
            active_processes.append({'process': proc, 'scene': scene, 'start_time': time.time()})
            log_info(f"Started subprocess for scene {scene['scene_name']}, GPU: {assigned_gpu}")

        while active_processes:
            if not process_active_once():
                time.sleep(0.2)
        
        total_time = sum(r['execution_time'] for r in results)
        log_info(f"Batch ray tracing completed!")
        log_info(f"Total scenes: {len(scenes)}")
        log_info(f"Success: {successful_count}, Failure: {failed_count}")
        log_info(f"Total time: {total_time:.1f}s, Avg time: {total_time/len(scenes):.1f}s")
        self._save_batch_results(results)
        return results

    def _fetch_result_for_scene(self, queue, pending_results, scene_name):
        self._drain_result_queue(queue, pending_results)
        while scene_name not in pending_results:
            try:
                result = queue.get(timeout=1)
            except queue_module.Empty:
                self._drain_result_queue(queue, pending_results)
                continue
            name = result['scene_info']['scene_name']
            pending_results[name] = result
        return pending_results.pop(scene_name)

    def _drain_result_queue(self, queue_obj, pending_results):
        try:
            while True:
                result = queue_obj.get_nowait()
                name = result['scene_info']['scene_name']
                pending_results[name] = result
        except queue_module.Empty:
            pass
        except Exception as e:
            log_info(f"Error while processing result queue: {e}")
    
    def _determine_gpu_ids(self) -> List[Optional[str]]:
        gpu_cfg = self.raytracing_config.get('gpu_config', {})
        mode = gpu_cfg.get('gpu_mode', 'auto')
        if mode == 'cpu_only':
            return []
        if mode and any(ch.isdigit() for ch in mode):
            ids = [gpu.strip() for gpu in mode.split(',') if gpu.strip()]
            return ids or []
        try:
            result = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index', '--format=csv,noheader'
            ]).decode('utf-8').strip()
            gpu_ids = [line.strip() for line in result.splitlines() if line.strip()]
            return gpu_ids
        except Exception:
            return []
    
    def _generate_heatmaps_for_scene(self, scene_name, result_dir):
        """Generate heatmaps for a given scene"""
        try:
            # Locate CSV data files
            csv_files = [
                os.path.join(result_dir, 'raytracing_results.csv'),
                os.path.join(result_dir, 'paths_data.csv'),
                os.path.join(result_dir, f'{scene_name}_raytracing.csv')
            ]
            
            data_file_path = None
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    data_file_path = csv_file
                    break
            
            if data_file_path:
                self.heatmap_generator.generate_scene_heatmaps(
                    scene_name, data_file_path, self.results_dir
                )
                log_info(f"Generated heatmaps for scene {scene_name}")
            else:
                log_info(f"No CSV data file found for scene {scene_name}, skipping heatmap generation")
                
        except Exception as e:
            log_info(f"Error generating heatmaps for scene {scene_name}: {str(e)}")
    
    def _save_batch_results(self, results: List[Dict[str, Any]]):
        """Save batch processing results"""
        try:
            # Save detailed results
            results_file = os.path.join(self.results_dir, 'batch_raytracing_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Save summary statistics
            stats_file = os.path.join(self.results_dir, 'raytracing_stats.txt')
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Sionna ray tracing batch processing report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total scenes: {len(results)}\n")
                f.write(f"Success: {len(successful_results)}\n")
                f.write(f"Failure: {len(failed_results)}\n")
                f.write(f"Success rate: {len(successful_results)/len(results)*100:.1f}%\n")
                
                if successful_results:
                    total_time = sum(r['execution_time'] for r in successful_results)
                    avg_time = total_time / len(successful_results)
                    total_paths = sum(r.get('num_paths', 0) for r in successful_results)
                    total_receivers = sum(r.get('num_receivers', 0) for r in successful_results)
                    
                    f.write(f"Total time: {total_time:.1f}s\n")
                    f.write(f"Average time: {avg_time:.1f}s\n")
                    f.write(f"Total paths: {total_paths}\n")
                    f.write(f"Total receivers: {total_receivers}\n")
                
                if failed_results:
                    f.write(f"\nFailed scenes:\n")
                    for r in failed_results:
                        scene_name = r['scene_info']['scene_name']
                        error = r.get('error', 'Unknown error')
                        f.write(f"  {scene_name}: {error}\n")
            
            log_info(f"Batch results saved to: {results_file}")
            log_info(f"Statistics report saved to: {stats_file}")
            
        except Exception as e:
            log_info(f"Failed to save batch results: {e}")