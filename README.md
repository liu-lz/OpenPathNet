# OpenPathNet: An Open-Source RF Multipath Data Generator for AI-Driven Wireless Systems

OpenPathNet is an open-source toolkit accompanied by a publicly released dataset for automatically generating radio-frequency (RF) environment scenes and running ray-tracing simulations. It can fetch real-world geographic data from OpenStreetMap (OSM), build 3D digital-twin scenes, and run high-performance ray-tracing simulations with NVIDIA Sionna (0.19.2).

## Dataset Access

- **Link 1 (about 3,000 128 m × 128 m scenes from three real cities, default configuration, no randomized scenes)**: [Download link](https://huggingface.co/datasets/liu-lz/OpenPathNet)
- **Link 2 (10 cities, about 1,000 128 m × 128 m scenes per city, mostly real scenes with randomized fallback on failure)**: [Download link](https://pan.ustc.edu.cn/share/index/3edb8705dc4b4ae49f04)  
  For detailed instructions, generation configurations, and checksum/verification information, please refer to the documentation included in the dataset.

## Citation

If you find OpenPathNet useful for your research, please consider citing this paper:

**L. Liu, X. Chen, and W. Zhang, “OpenPathNet: An Open-Source RF Multipath Data Generator for AI-Driven Wireless Systems,” *arXiv preprint arXiv:2512.*, 2025.**

BibTeX:
```bibtex
@article{liu2025openpathnet,
  title={OpenPathNet: An Open-Source RF Multipath Data Generator for AI-Driven Wireless Systems},
  author={Liu, L. and Chen, X. and Zhang, W.},
  journal={arXiv preprint arXiv:2512.},
  year={2025}
}
```

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [FAQ](#faq)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

### Scene Generation
- **Real-scene generation based on OSM**: Fetch real building data from OpenStreetMap and automatically generate 3D scenes
- **Smart retry mechanism**: When OSM data retrieval fails, re-sample a point within a specified range and retry
- **Random-scene generation**: When OSM data cannot be obtained, automatically generate randomized building scenes as a fallback
- **Batch scene generation**: Generate large numbers of scenes in batches, suitable for large-scale dataset construction
- **Multi-region support**: Preconfigured regional settings for 30+ major cities worldwide

### Ray-Tracing Simulation
- **Sionna integration**: High-performance ray tracing powered by NVIDIA Sionna
- **GPU acceleration**: Support single-GPU, multi-GPU, and CPU modes, with automatic detection and resource allocation
- **Batch processing**: Process multiple scenes in batches, with automatic memory and GPU resource management
- **Multiple output formats**: Export results in CSV, Pickle, NPY, and more
- **Visualization support**: Automatically generate visualizations such as path-loss heatmaps

### Other Highlights
- **Flexible configuration system**: YAML-based configuration files with command-line overrides
- **Detailed logging**: A complete logging system for debugging and monitoring
- **Mitsuba compatibility**: Generated scene files are compatible with the Mitsuba renderer
- **Material system**: Support ITU-standard material definitions

## System Requirements
- Operating system: Linux (Ubuntu 20.04+ recommended), macOS, Windows (WSL2)
- Python: ≥ 3.10
- GPU (optional): CUDA 11.8+, VRAM ≥ 8 GB

## Installation Guide

### Using Conda (Recommended)

We provide two Conda environment configuration files:
- `openpathnet-cpu.yml`: CPU version (includes all dependencies, excluding CUDA-related packages)
- `openpathnet-gpu.yml`: GPU version (includes CUDA and TensorFlow GPU support)

#### For most users, we recommend installing the CPU environment by default

```bash
# Clone the repository
git clone https://github.com/liu-lz/OpenPathNet
cd OpenPathNet

# Create and activate a Conda environment
conda create -n <env-new> python=3.10
conda activate <env-new>

# Install the package (rf-scene-generator is required, so please install with this command)
# Ensure 'pip' points to the active environment before running: which pip
pip install -e .
```

#### GPU Installation

If you need GPU support, install CUDA and TensorFlow GPU support on top of the CPU environment (you may refer to `openpathnet-gpu.yml`):

```bash
pip install "tensorflow[and-cuda]==2.19.0"
```

### Common Installation Issues

#### TensorFlow GPU Support

If TensorFlow cannot detect your GPU, install a compatible TensorFlow version, configure CUDA/cuDNN per official TensorFlow requirements, and make sure it is compatible with Sionna 0.19.2.

#### DRJIT/LLVM

If you run into DRJIT-related LLVM library issues:

```bash
# Install llvmlite
conda install llvmlite llvm-tools

# Find the libLLVM.so path
find $CONDA_PREFIX -name "libLLVM.so*" 2>/dev/null

# Set the environment variable (adjust the path based on your system)
export DRJIT_LIBLLVM_PATH=/usr/.../libLLVM.so
```

## Quick Start

### 1. Generate Scenes

```bash
# Generate scenes with the default configuration
python scripts/generate_scenes.py

# Show the current configuration
python scripts/generate_scenes.py --show-config

# List available regions
python scripts/generate_scenes.py --list-regions

# Generate 20 scenes in the Beijing region
python scripts/generate_scenes.py --region beijing --num-scenes 20

# Generate scenes using custom coordinates
python scripts/generate_scenes.py \
    --center-lat 39.9042 \
    --center-lon 116.4074 \
    --radius-km 5.0 \
    --num-scenes 10

# Override specific parameters (you can also edit configs/regions_config.yaml)
python scripts/generate_scenes.py \
    --region hefei \
    --num-scenes 30 \
    --generation-mode osm_retry \
    --search-radius-km 3.0
```

### 2. Run Ray Tracing

```bash
# Basic ray tracing (default configuration)
python scripts/run_raytracing.py

# List available scenes
python scripts/run_raytracing.py --list-scenes

# Specify GPU mode
python scripts/run_raytracing.py --gpu-mode auto

# Process specific scenes
python scripts/run_raytracing.py --scene-pattern "beijing" --max-scenes 5
```

### 3. Use the CLI Tool (advanced scene generation based on [geo2sigmap](https://github.com/functions-lab/geo2sigmap), generated environment files are saved under data/output)

```bash
# Define a scene using a bounding box
scenegen bbox \
    --data-dir output/scene1 \
    -71.06025695800783 42.35128145107633 \
    -71.04841232299806 42.35917815419112

# Define a scene using a point and size
scenegen point \
    --data-dir output/scene2 \
    116.4074 39.9042 center \
    500 500 \
    --enable-building-map
```

## Project Structure

```
OpenPathNet/
├── src/                          # Source code
│   ├── scene_generation/         # Core module for scene generation
│   │   ├── core.py               # Scene class and core scene-generation logic
│   │   ├── cli.py                # Command-line interface
│   │   └── utils.py              # Utility functions
│   ├── raytracer/                # Ray-tracing module
│   │   ├── sionna_raytracer.py   # Wrapper for the Sionna ray tracer
│   │   └── raytracing_manager.py # Ray-tracing manager
│   ├── coordinate_generator.py   # Coordinate generator
│   ├── scene_manager.py          # Scene manager
│   ├── batch_processor.py        # Batch processor
│   ├── config_manager.py         # Configuration manager
│   └── utils/                    # Utility modules
│       ├── logging_utils.py      # Logging utilities
│       └── file_utils.py         # File I/O utilities
├── scripts/                      # Executable scripts
│   ├── generate_scenes.py        # Scene generation script
│   ├── run_raytracing.py         # Ray-tracing script
│   └── cleanup.py                # Cleanup script
├── configs/                      # Configuration files
│   └── regions_config.yaml       # Region and system configuration
├── data/                         # Data directory
│   ├── scenes/                   # Generated scene files (includes example data for 5 real-city scenes)
│   └── raytracing_results/       # Ray-tracing results (includes example results for the corresponding scenes)
├── logs/                         # Log directory (includes example logs for the corresponding scenes)
├── openpathnet-cpu.yml           # Conda environment config (CPU)
├── openpathnet-gpu.yml           # Conda environment config (GPU)
├── requirements.txt              # Python dependency list
├── setup.py                      # Installation script
├── pyproject.toml                # Project metadata
└── README.md                     # This document
```

**Note**: The `data/scenes/` directory includes environment assets for five real-city scenes (XML scene files and mesh files). The `data/raytracing_results/` directory includes the corresponding ray-tracing outputs, and `logs/` includes the corresponding log files. These example assets can help you quickly verify your installation and understand the data formats.

## Configuration

### Configuration File Structure

OpenPathNet uses a YAML configuration file (default location: `configs/regions_config.yaml`). The configuration file includes the following main parts:

#### 1. Region configuration
#### 2. Scene generation configuration
#### 3. Ray-tracing configuration

### Key Notes
- Configuration file path: `configs/regions_config.yaml`
- Generation modes: `fallback` (fall back to randomized scenes when OSM fails) or `osm_retry` (re-sample points and retry on failure)
- Key parameters: `scene_generation.num_scenes/size_x/size_y/generation_mode`; `raytracing.gpu_config.gpu_mode`; `raytracing.receiver_grid.*`
- Priority order: command line > configuration file > defaults

## FAQ

### Q1: What should I do if OSM data retrieval fails?

**A**: The project provides two options:
1. **Fallback mode**: automatically switch to randomized scene generation
2. **OSM retry mode**: re-sample a point within a specified range and retry

```bash
# Use OSM retry mode, increase the number of attempts and the search radius
python scripts/generate_scenes.py     --generation-mode osm_retry     --max-osm-attempts 20     --search-radius-km 5.0
```

### Q2: What should I do if GPU memory is insufficient?

**A**: You can address this in a few ways:
1. Reduce the receiver grid size (`receiver_grid.grid_size`)
2. Reduce the batch size (`batch_processing.rx_batch_size`)
3. Use CPU mode (`--gpu-mode cpu_only`)

### Q3: How do I add a new region configuration?

**A**: Edit `configs/regions_config.yaml` and add a new region under `regions`:

```yaml
regions:
  your_city:
    name: "Your City"
    center_lat: latitude
    center_lon: longitude
    radius_km: radius
    description: "Description"
```

### Q4: What is the format of the generated scene files?

**A**: The project generates Mitsuba 2.1.0–compatible XML scene files, including:
- `scene.xml`: scene description file
- `mesh/ground.ply`: ground mesh
- `mesh/building_*.ply`: building meshes

### Q5: How can I improve ray-tracing performance?

**A**:
1. Use GPU acceleration (`--gpu-mode auto`)
2. Tune the `ray_samples` parameter (more samples means higher accuracy but longer runtime)
3. Reduce `max_depth` (fewer reflections)
4. Use batch processing to run multiple scenes in parallel

### Q6: What output formats are supported?

**A**: Ray-tracing results support the following formats:
- CSV (`save_csv: true`)
- Pickle (`save_pickle: true`)
- NPY (DeepMIMO-style dictionary structure) (`save_deepmimo: true`)
- Visualization plots (`save_visualizations: true`)

For common installation issues, please see “Common Installation Issues” in the [Installation Guide](#installation-guide).

## License

This project is licensed under the Apache-2.0 License. For details, see the [LICENSE](LICENSE) file.

## Acknowledgements

- [NVIDIA Sionna](https://nvidia.github.io/sionna/) - ray-tracing engine
- [OSM](https://www.openstreetmap.org/) - OpenStreetMap
- [Blender](https://www.blender.org/) - Blender
- [Open3D](http://www.open3d.org/) - 3D data processing
- [Mitsuba](https://www.mitsuba-renderer.org/) - renderer
- [geo2sigmap](https://github.com/functions-lab/geo2sigmap) - 

---

Thank you for using OpenPathNet! If you have any questions or suggestions, feel free to open an issue on GitHub.
