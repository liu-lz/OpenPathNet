import random
import math
import numpy as np

def generate_random_coordinates(center_lat, center_lon, radius_km, num_points):
    """
    Generate random coordinates around a center point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius (km)
        num_points: Number of points to generate
    
    Returns:
        list: List of coordinates [(lat1, lon1), (lat2, lon2), ...]
    """
    coordinates = []
    
    # Convert km to degrees (approx.)
    # 1 degree latitude ≈ 111.32 km
    # 1 degree longitude ≈ 111.32 * cos(latitude) km
    lat_degree_km = 111.32
    lon_degree_km = 111.32 * math.cos(math.radians(center_lat))
    
    radius_lat = radius_km / lat_degree_km
    radius_lon = radius_km / lon_degree_km
    
    for i in range(num_points):
        # Generate random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        # Use sqrt for uniform distribution
        distance = math.sqrt(random.uniform(0, 1)) * radius_km
        
        # Convert to lat/lon offsets
        lat_offset = (distance / lat_degree_km) * math.cos(angle)
        lon_offset = (distance / lon_degree_km) * math.sin(angle)
        
        # Calculate new coordinates
        new_lat = center_lat + lat_offset
        new_lon = center_lon + lon_offset
        
        coordinates.append((new_lat, new_lon))
    
    return coordinates

def generate_grid_coordinates(center_lat, center_lon, grid_size_km, spacing_km):
    """
    Generate grid-distributed coordinates.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        grid_size_km: Grid size (km)
        spacing_km: Grid spacing (km)
    
    Returns:
        list: List of coordinates [(lat1, lon1), (lat2, lon2), ...]
    """
    coordinates = []
    
    # Degree conversion
    lat_degree_km = 111.32
    lon_degree_km = 111.32 * math.cos(math.radians(center_lat))
    
    # Grid parameters
    half_size_lat = (grid_size_km / 2) / lat_degree_km
    half_size_lon = (grid_size_km / 2) / lon_degree_km
    spacing_lat = spacing_km / lat_degree_km
    spacing_lon = spacing_km / lon_degree_km
    
    # Calculate number of points in grid
    num_points_lat = int(grid_size_km / spacing_km) + 1
    num_points_lon = int(grid_size_km / spacing_km) + 1
    
    for i in range(num_points_lat):
        for j in range(num_points_lon):
            lat_offset = -half_size_lat + i * spacing_lat
            lon_offset = -half_size_lon + j * spacing_lon
            
            new_lat = center_lat + lat_offset
            new_lon = center_lon + lon_offset
            
            coordinates.append((new_lat, new_lon))
    
    return coordinates

def generate_line_coordinates(start_lat, start_lon, end_lat, end_lon, num_points):
    """
    Generate line-distributed coordinates between two points.
    
    Args:
        start_lat: Start latitude
        start_lon: Start longitude
        end_lat: End latitude
        end_lon: End longitude
        num_points: Number of points to generate
    
    Returns:
        list: List of coordinates [(lat1, lon1), (lat2, lon2), ...]
    """
    coordinates = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        
        coordinates.append((lat, lon))
    
    return coordinates

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance (km) between two coordinates using the Haversine formula.
    
    Args:
        lat1, lon1: First point latitude/longitude
        lat2, lon2: Second point latitude/longitude
    
    Returns:
        float: Distance (km)
    """
    R = 6371  # Earth radius (km)
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def filter_coordinates_by_distance(coordinates, center_lat, center_lon, min_distance_km, max_distance_km):
    """
    Filter coordinates by distance range.
    
    Args:
        coordinates: Original coordinate list
        center_lat: Center latitude
        center_lon: Center longitude
        min_distance_km: Minimum distance (km)
        max_distance_km: Maximum distance (km)
    
    Returns:
        list: Filtered coordinate list
    """
    filtered_coordinates = []
    
    for lat, lon in coordinates:
        distance = calculate_distance_km(center_lat, center_lon, lat, lon)
        if min_distance_km <= distance <= max_distance_km:
            filtered_coordinates.append((lat, lon))
    
    return filtered_coordinates

def get_urban_coordinates_china(center_lat, center_lon, radius_km, num_points):
    """
    Generate coordinates for Chinese urban areas, prioritizing city centers.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius (km)
        num_points: Number of points to generate
    
    Returns:
        list: Coordinate list
    """
    coordinates = []
    
    # Bias strategy for dense urban areas (CBD/residential)
    urban_biases = [
        (0.0, 0.0),      # Center
        (0.005, 0.005),  # NE
        (-0.005, 0.005), # NW
        (0.005, -0.005), # SE
        (-0.005, -0.005),# SW
        (0.01, 0.0),     # East
        (-0.01, 0.0),    # West
        (0.0, 0.01),     # North
        (0.0, -0.01),    # South
    ]
    
    # Split radius into rings and sample in each ring
    for i in range(num_points):
        if i < len(urban_biases):
            # Use predefined urban biases
            lat_bias, lon_bias = urban_biases[i]
            lat = center_lat + lat_bias
            lon = center_lon + lon_bias
        else:
            # Generate random point, biased towards city center
            angle = random.uniform(0, 2 * math.pi)
            # Use smaller radius, concentrated near city center
            distance = random.uniform(0, radius_km * 0.7)  # within 70%
            
            lat_degree_km = 111.32
            lon_degree_km = 111.32 * math.cos(math.radians(center_lat))
            
            lat_offset = (distance / lat_degree_km) * math.cos(angle)
            lon_offset = (distance / lon_degree_km) * math.sin(angle)
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
        
        coordinates.append((lat, lon))
    
    return coordinates

def save_coordinates_to_file(coordinates, filename):
    """
    Save coordinates to file.
    
    Args:
        coordinates: List of coordinates
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            f.write("latitude,longitude\n")
            for lat, lon in coordinates:
                f.write(f"{lat:.6f},{lon:.6f}\n")
        print(f"Coordinates saved to: {filename}")
    except Exception as e:
        print(f"Failed to save coordinates: {e}")

def load_coordinates_from_file(filename):
    """
    Load coordinates from file.
    
    Args:
        filename: Input filename
    
    Returns:
        list: Coordinate list
    """
    coordinates = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                line = line.strip()
                if line:
                    lat, lon = map(float, line.split(','))
                    coordinates.append((lat, lon))
        print(f"Loaded {len(coordinates)} coordinates from file")
    except Exception as e:
        print(f"Failed to load coordinates: {e}")
    
    return coordinates

# Test functions
if __name__ == "__main__":
    # Coordinate generation tests
    print("Testing coordinate generation...")
    
    # Generate 10 random coordinates
    coords = generate_random_coordinates(31.839380, 117.250454, 5.0, 10)
    print(f"Generated {len(coords)} random coordinates:")
    for i, (lat, lon) in enumerate(coords):
        print(f"  {i+1}: ({lat:.6f}, {lon:.6f})")
    
    # Generate grid coordinates
    grid_coords = generate_grid_coordinates(31.839380, 117.250454, 2.0, 0.5)
    print(f"\nGenerated {len(grid_coords)} grid coordinates")
    
    # Distance test
    if len(coords) >= 2:
        dist = calculate_distance_km(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
        print(f"\nDistance between first two points: {dist:.3f} km")