from setuptools import setup, find_packages

setup(
    name='rf-scene-generator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for generating RF scene datasets for simulations.',
    license='Apache-2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'PyYAML',
        'psutil',
        'tensorflow',
        'scipy',
        'sionna==0.19.2',
        'osmnx>=2.0.0',
        'pyproj',
        'shapely',
        'rasterio',
        'pillow',
        'open3d',
        'triangle',
    ],
    entry_points={
        'console_scripts': [
            'generate_scenes=scripts.generate_scenes:main',
            'run_batch_simulation=scripts.run_batch_simulation:main',
            'cleanup=scripts.cleanup:main',
        ],
    },
)