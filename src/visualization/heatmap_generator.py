import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import Normalize, LinearSegmentedColormap, LogNorm
from matplotlib import cm
import matplotlib.colors as colors
import logging
from pathlib import Path

class HeatmapGenerator:
    """Generate heatmap visualizations for ray tracing results"""
    
    def __init__(self, config):
        """
        Initialize the heatmap generator
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.logger = logging.getLogger(__name__)
        
        # Map parameter names to corresponding column names (based on CSV format)
        self.params = {
            'channel_gain': 'channel_gain',
            'delay': 'tau',
            'elevation': 'theta_r',
            'azimuth': 'phi_r'
        }
        
        # Parameter units
        self.param_units = {
            'channel_gain': 'dB',
            'delay': 'ns',
            'elevation': '°',
            'azimuth': '°'
        }
        
        # Colormaps
        self.colormaps = {
            'channel_gain': self._create_custom_power_colormap(),
            'delay': 'plasma',
            'elevation': self._create_custom_elevation_colormap(),
            'azimuth': 'hsv'
        }
        self.angle_bounds = {
            'elevation': tuple(self.viz_config.get('elevation_bounds', (0, 180))),
            'azimuth': tuple(self.viz_config.get('azimuth_bounds', (-180, 180)))
        }
        self.power_log_ratio = self.viz_config.get('power_log_ratio', 50)
        self.min_range_padding = self.viz_config.get('min_range_padding', 1e-6)
    
    def generate_scene_heatmaps(self, scene_name, data_file_path, output_dir):
        """
        Generate heatmaps for a given scene.
        
        Args:
            scene_name (str): Scene name
            data_file_path (str): Data file path
            output_dir (str): Output directory
        """
        try:
            # Check if data file exists
            if not os.path.exists(data_file_path):
                self.logger.warning(f"Data file not found for scene {scene_name}: {data_file_path}")
                return
            
            # Read data
            raw_data = pd.read_csv(data_file_path)
            self.logger.info(f"Loaded raw data for scene {scene_name} from {data_file_path}")
            
            # Check for required columns
            required_columns = ['rx_id', 'rx_coord'] + list(self.params.values())
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns in raw data for scene {scene_name}: {missing_columns}")
                return
            
            # Process data: extract first path information for each position
            processed_data = self._process_raytracing_data(raw_data)
            
            if processed_data.empty:
                self.logger.warning(f"No valid data after processing for scene {scene_name}")
                return
            
            # Create output directory
            scene_output_dir = os.path.join(output_dir, scene_name, 'heatmaps')
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Calculate parameter ranges
            param_ranges = self._calculate_param_ranges(processed_data)
            
            # Generate heatmaps for each parameter
            for param, column in self.params.items():
                if column in processed_data.columns:
                    vmin, vmax = param_ranges[param]
                    display_name = 'ToA' if param == 'delay' else param
                    filename = "ToA_heatmap.png" if param == 'delay' else f"{param}_heatmap.png"
                    output_path = os.path.join(scene_output_dir, filename)
                    
                    self._plot_parameter_distribution(
                        processed_data, param, column, self.colormaps[param], 
                        output_path, vmin=vmin, vmax=vmax, display_label=display_name
                    )
                    
                    self.logger.info(f"Generated {display_name} heatmap for scene {scene_name}")
                else:
                    self.logger.warning(f"Column {column} not found in processed data for scene {scene_name}")
                    
        except Exception as e:
            self.logger.error(f"Error generating heatmaps for scene {scene_name}: {str(e)}")
    
    def _process_raytracing_data(self, raw_data):
        """
        Process ray tracing data and extract the first path for each position.
        
        Args:
            raw_data: Raw CSV data
            
        Returns:
            pandas.DataFrame: Processed data containing x, y coordinates and parameters
        """
        try:
            # Parse coordinate string like '[1.0, 2.0, 1.5]' -> (1.0, 2.0)
            def parse_coordinates(coord_str):
                """Parse coordinate string and return x, y"""
                try:
                    # Remove brackets and split
                    coord_str = coord_str.strip('[]')
                    coords = [float(x.strip()) for x in coord_str.split(',')]
                    return coords[0], coords[1]  # Return x, y coordinates
                except:
                    return None, None
            
            # Parse all coordinates
            coords = raw_data['rx_coord'].apply(parse_coordinates)
            raw_data['x'] = coords.apply(lambda x: x[0])
            raw_data['y'] = coords.apply(lambda x: x[1])
            
            # Drop rows where coordinate parsing failed
            raw_data = raw_data.dropna(subset=['x', 'y'])
            
            # Group by rx_id and take the first path (first row)
            first_paths = raw_data.groupby('rx_id').first().reset_index()
            
            # Check for valid data
            if first_paths.empty:
                self.logger.warning("No valid first paths found after grouping")
                return pd.DataFrame()
            
            # Select relevant columns
            result_columns = ['x', 'y'] + list(self.params.values())
            available_columns = [col for col in result_columns if col in first_paths.columns]
            
            processed_data = first_paths[available_columns].copy()
            if 'tau' in processed_data.columns:
                processed_data['tau'] = pd.to_numeric(processed_data['tau'], errors='coerce') * 1e9
            if 'channel_gain' in processed_data.columns:
                processed_data['channel_gain'] = pd.to_numeric(processed_data['channel_gain'], errors='coerce')
            self.logger.info(f"Processed {len(processed_data)} receiver positions from {len(raw_data)} total paths")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing raytracing data: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_param_ranges(self, data):
        param_ranges = {}
        for param, column in self.params.items():
            if column in data.columns and not data[column].empty:
                if param == 'channel_gain':
                    positive = data[column] > 0
                    if positive.any():
                        gain_db = 10 * np.log10(data.loc[positive, column].to_numpy())
                        vmin = gain_db.min()
                        vmax = gain_db.max()
                    else:
                        vmin, vmax = (-120.0, -60.0)
                else:
                    vmin = data[column].min()
                    vmax = data[column].max()
                if param in self.angle_bounds:
                    default_min, default_max = self.angle_bounds[param]
                    if default_min is not None:
                        vmin = min(vmin, default_min)
                    if default_max is not None:
                        vmax = max(vmax, default_max)
                if np.isclose(vmin, vmax):
                    pad = self.min_range_padding or 1e-6
                    vmin -= pad
                    vmax += pad
                param_ranges[param] = (vmin, vmax)
            else:
                param_ranges[param] = self.angle_bounds.get(param, (0, 1))
        return param_ranges
    
    def _create_custom_power_colormap(self):
        """Create a custom power colormap"""
        inferno = plt.cm.inferno
        colors_array = np.ones((256, 4))
        threshold_idx = 25
        threshold_color = np.array(inferno(threshold_idx/255))
        below_threshold_color = threshold_color.copy()
        below_threshold_color[:3] = below_threshold_color[:3] * 0.85
        colors_array[0] = below_threshold_color
        
        for i in range(1, 256):
            if i < 38:
                t = i / 38
                colors_array[i] = below_threshold_color * (1-t) + np.array(inferno(i/255)) * t
            else:
                colors_array[i] = inferno(i/255)
        
        custom_cmap = LinearSegmentedColormap.from_list('custom_power', colors_array)
        custom_cmap.set_under(below_threshold_color)
        return custom_cmap
    
    def _create_custom_elevation_colormap(self):
        """Create a custom elevation colormap"""
        elevations = np.array([-90, -30, 0, 30, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 130, 150, 180, 270])
        norm_elevs = (elevations - elevations.min()) / (elevations.max() - elevations.min())
        
        colors_list = [
            (0.0, 0.0, 0.6), (0.0, 0.3, 0.8), (0.0, 0.6, 1.0), (0.0, 0.8, 0.8), (0.0, 1.0, 0.6),
            (0.0, 0.9, 0.4), (0.0, 1.0, 0.2), (0.2, 1.0, 0.0), (0.3, 1.0, 0.0), (0.5, 1.0, 0.0),
            (0.7, 1.0, 0.0), (0.9, 1.0, 0.0), (1.0, 0.9, 0.0), (1.0, 0.8, 0.0), (1.0, 0.6, 0.0),
            (1.0, 0.4, 0.0), (1.0, 0.2, 0.0), (1.0, 0.0, 0.0), (0.6, 0.0, 0.6)
        ]
        
        custom_cmap = LinearSegmentedColormap.from_list('custom_elevation', list(zip(norm_elevs, colors_list)))
        return custom_cmap

    class AngleNormalize(Normalize):
        """Custom angle normalizer"""
        def __init__(self, vmin=-180, vmax=180):
            super().__init__(vmin=vmin, vmax=vmax)
        
        def __call__(self, value, clip=None):
            value = np.asarray(value, dtype=float)
            value = ((value - self.vmin) % (self.vmax - self.vmin + 360)) + self.vmin
            result = (value - self.vmin) / (self.vmax - self.vmin)
            if clip is not None:
                result = np.clip(result, 0, 1)
            return np.ma.array(result)
    
    def _plot_parameter_distribution(self, data, param_name, column_name, colormap, output_path, vmin=None, vmax=None, display_label=None):
        plt.figure(figsize=(12, 10))
        values = data[column_name].to_numpy()
        point_size = self.viz_config.get('point_size', 50)

        if param_name == 'channel_gain':
            positive = values > 0
            gain_db = np.full_like(values, np.nan, dtype=float)
            gain_db[positive] = 10 * np.log10(values[positive])
            finite_vals = gain_db[np.isfinite(gain_db)]
            if finite_vals.size == 0:
                self.logger.warning("No valid channel gain data for heatmap.")
                return
            vmin = finite_vals.min() if vmin is None else vmin
            vmax = finite_vals.max() if vmax is None else vmax
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
            scatter = plt.scatter(data['x'], data['y'], c=gain_db, cmap=colormap,
                                  alpha=0.8, s=point_size, linewidths=0.5,
                                  norm=Normalize(vmin=vmin, vmax=vmax))
        elif param_name == 'azimuth':
            norm = self.AngleNormalize(vmin=vmin, vmax=vmax)
            scatter = plt.scatter(data['x'], data['y'], c=values, cmap=colormap,
                                  alpha=0.8, s=point_size, linewidths=0.5, norm=norm)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            scatter = plt.scatter(data['x'], data['y'], c=values, cmap=colormap,
                                  alpha=0.8, s=point_size, linewidths=0.5, norm=norm)

        cbar = plt.colorbar(scatter)
        label_name = display_label or param_name
        cbar.set_label(f'{label_name} ({self.param_units[param_name]})', fontsize=28)
        cbar.ax.tick_params(labelsize=28)

        plt.xlabel('X Position', fontsize=28)
        plt.ylabel('Y Position', fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=28)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved heatmap: {output_path}")