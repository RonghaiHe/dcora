import random
import math
import sys
import argparse
from itertools import islice
import utils

"""
python generate_pyfg.py ../data/range_aided_slam_test_2d.pyfg ../data/range_aided_slam_test_2d_ada.pyfg
"""

t = [[0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]]
# t = [[0.0, 0.0]]*4


def read_g2o_file(filename):
    """
    Read a g2o file and extract VERTEX_SE2 and EDGE_SE2 data.
    Returns a tuple (vertices, edges) containing the extracted data.
    """
    vertices = []
    edges = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(' ')
                if not parts:
                    continue

                if parts[0] == 'VERTEX_SE2':
                    # Process VERTEX_SE2 line
                    if len(parts) < 6:
                        print(f"Warning: Incomplete VERTEX_SE2 line: {line}")
                        continue

                    # Format: VERTEX_SE2 ts sym x y theta
                    timestamp = float(parts[1])
                    sym = parts[2]
                    x = float(parts[3])
                    y = float(parts[4])
                    theta = float(parts[5])

                    vertices.append({
                        'type': 'VERTEX_SE2',
                        'timestamp': timestamp,
                        'sym': sym,
                        'x': x,
                        'y': y,
                        'theta': theta
                    })

                elif parts[0] == 'VERTEX_XY':
                    # Process VERTEX_XY line
                    if len(parts) < 4:
                        print(f"Warning: Incomplete VERTEX_XY line: {line}")
                        continue

                    # Format: VERTEX_XY sym x y
                    sym = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])

                    vertices.append({
                        'type': 'VERTEX_XY',
                        'sym': sym,
                        'x': x,
                        'y': y,
                    })

                elif parts[0] == 'EDGE_SE2':
                    # Process EDGE_SE2 line
                    if len(parts) < 7:
                        print(f"Warning: Incomplete EDGE_SE2 line: {line}")
                        continue

                    # Format: EDGE_SE2 ts sym1 sym2 dx dy dtheta information_matrix
                    timestamp = float(parts[1])
                    sym1 = parts[2]
                    sym2 = parts[3]
                    dx = float(parts[4])
                    dy = float(parts[5])
                    dtheta = float(parts[6])

                    edge = {
                        'type': 'EDGE_SE2',
                        'timestamp': timestamp,
                        'sym1': sym1,
                        'sym2': sym2,
                        'dx': dx,
                        'dy': dy,
                        'dtheta': dtheta
                    }

                    # Extract information matrix if available
                    if len(parts) > 7:
                        info_matrix = [float(x) for x in parts[7:]]
                        edge['information_matrix'] = info_matrix

                    edges.append(edge)

    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, None

    return vertices, edges


def add_range_measurements(vertices, edges, noise_std=0.1, max_range=100, avg_timestamp_diff=1.0):
    """
    Add EDGE_RANGE measurements between vertices based on their positions.
    Measurements include Gaussian noise based on the given standard deviation.
    """
    # Create a mapping from (robot_id, state_id) to vertex index for quick lookup
    vertex_map = {}

    # Create a mapping from landmark_id to positions for quick lookup
    vertex_landamark_map = {}

    for i, vertex in enumerate(vertices):
        try:
            if (vertex['sym'][0] == 'L'):
                vertex_landamark_map[vertex['sym']] = (
                    vertex['x'], vertex['y'])
            else:
                robot_id, state_id = utils.get_robot_and_state_id_from_symbol(
                    vertex['sym'])
                vertex_map[(robot_id, state_id)] = i
        except Exception as e:
            print(
                f"Warning: Could not parse symbol {vertex['sym']} for vertex at index {i}: {str(e)}")

    # For each vertex, find nearby vertices and create range measurements
    new_edges = []
    edge_index = len(edges)

    start_idx2 = 0
    # Iterate over pairs of vertices with different robot_ids
    for (robot_id1, _), idx1 in vertex_map.items():
        vertex1 = vertices[idx1]

        # landmark measurements
        for (sym, state) in vertex_landamark_map.items():
            # Calculate distance between vertices
            dx = vertex1['x'] + t[robot_id1][0] - \
                state[0]
            dy = vertex1['y'] + t[robot_id1][1] - \
                state[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Only add range measurement if within max_range
            if distance < max_range:
                # Add Gaussian noise
                noise = random.gauss(0, noise_std)
                range_measurement = distance + noise

                # Create covariance matrix (simplified - just using distance for now)
                # (noise_std * distance) ** 2
                covariance = 0.0009186531949884554

                # Create timestamp (slightly after the vertex timestamp)
                timestamp = vertex1['timestamp'] + \
                    avg_timestamp_diff * random.uniform(0.5, 1)

                # Create edge
                edge = {
                    'type': 'EDGE_RANGE',
                    'timestamp': timestamp,
                    'sym1': vertex1['sym'],
                    'sym2': sym,
                    'range': range_measurement,
                    'covariance': covariance
                }
                new_edges.append(edge)

                # Print progress
                print(
                    f"Added range measurement {edge_index}: {vertex1['sym']} <-> \
                        {sym}, landmark: {sym}, distance: {distance:.3f}, \
                            measured: {range_measurement:.3f}")
                edge_index += 1

        # inter-robot measurements
        if random.uniform(0, 1) < 0.5:
            continue
        min_time_diff = float('inf')  # Initialize minimum time difference
        closest_vertex_idx = None     # Index of the closest vertex

        for (robot_id2, _), idx2 in islice(vertex_map.items(), start_idx2, None):
            if robot_id1 == robot_id2 or idx1 == idx2:
                continue

            # Calculate time difference between vertices
            time_diff = abs(vertex1['timestamp'] - vertices[idx2]['timestamp'])

            # Only consider vertices with time difference <= 0.05
            if time_diff > 0.05:
                continue

            # Update minimum time difference and closest vertex index
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_vertex_idx = idx2

        # If a closest pair was found, add the range measurement
        if closest_vertex_idx is not None:
            start_idx2 = closest_vertex_idx

            # Calculate distance between vertices
            dx = vertex1['x'] + t[robot_id1][0] - \
                vertices[closest_vertex_idx]['x'] - t[robot_id2][0]
            dy = vertex1['y'] + t[robot_id1][1] - \
                vertices[closest_vertex_idx]['y'] - t[robot_id2][1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Only add range measurement if within max_range
            if distance < max_range:
                # Add Gaussian noise
                noise1 = random.gauss(0, noise_std)
                range_measurement1 = distance + noise1

                noise2 = random.gauss(0, noise_std)
                range_measurement2 = distance + noise2

                # Create covariance matrix (simplified - just using distance for now)
                # (noise_std * distance) ** 2
                covariance = 0.0009186531949884554

                # Create timestamp (slightly after the vertex timestamp)
                timestamp1 = vertex1['timestamp'] + avg_timestamp_diff * 0.1
                timestamp2 = vertices[closest_vertex_idx]['timestamp'] + \
                    avg_timestamp_diff * 0.1

                # Create edge
                edge = {
                    'type': 'EDGE_RANGE',
                    'timestamp': timestamp1,
                    'sym1': vertex1['sym'],
                    'sym2': vertices[closest_vertex_idx]['sym'],
                    'range': range_measurement1,
                    'covariance': covariance
                }
                new_edges.append(edge)

                edge = {
                    'type': 'EDGE_RANGE',
                    'timestamp': timestamp2,
                    'sym1': vertices[closest_vertex_idx]['sym'],
                    'sym2': vertex1['sym'],
                    'range': range_measurement2,
                    'covariance': covariance
                }
                new_edges.append(edge)

                # Print progress
                print(
                    f"Added range measurement {edge_index}: {vertex1['sym']} <-> \
                        {vertices[closest_vertex_idx]['sym']}, distance: {distance:.3f}, \
                            measured1: {range_measurement1:.3f}, \
                                measured2: {range_measurement2:.3f}")
                edge_index += 2

    return new_edges


def write_g2o_file(vertices, edges, filename):
    """Write vertices and edges to a g2o file."""
    try:
        with open(filename, 'w') as f:
            # Write vertices
            for vertex in vertices:
                if vertex['type'] == 'VERTEX_SE2':
                    f.write(
                        f"VERTEX_SE2 {vertex['timestamp']} {vertex['sym']} {vertex['x']} {vertex['y']} {vertex['theta']}\n")
                else:
                    f.write(
                        f"VERTEX_XY {vertex['sym']} {vertex['x']} {vertex['y']}\n")

            # Write edges
            for edge in edges:
                if edge['type'] == 'EDGE_SE2':
                    info_matrix = ' '.join(str(x) for x in edge.get(
                        'information_matrix', [0.0001, 0.0, -0.0, 0.0001, -0.0, 2.5e-05]))
                    f.write(
                        f"EDGE_SE2 {edge['timestamp']} {edge['sym1']} {edge['sym2']} {edge['dx']} {edge['dy']} {edge['dtheta']} {info_matrix}\n")
                elif edge['type'] == 'EDGE_RANGE':
                    f.write(
                        f"EDGE_RANGE {edge['timestamp']} {edge['sym1']} {edge['sym2']} {edge['range']} {edge['covariance']}\n")

        return True
    except Exception as e:
        print(f"Error writing file {filename}: {e}")
        return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate a .pyfg file with VERTEX_SE2, VERTEX_XY, EDGE_SE2, and EDGE_RANGE entries.')
    parser.add_argument('input_file', help='Path to the input .pyfg file')
    parser.add_argument('output_file', help='Path to the output .pyfg file')
    parser.add_argument('--avg_timestamp_diff', type=float, default=1.0,
                        help='Average timestamp difference between consecutive poses')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Standard deviation of Gaussian noise as a fraction of distance (default: 0.1)')
    parser.add_argument('--max_range', type=float, default=100.0,
                        help='Maximum range for adding range measurements (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Read input file
    vertices, edges = read_g2o_file(args.input_file)
    if vertices is None or edges is None:
        print("Error reading input file")
        sys.exit(1)

    print(f"Read {len(vertices)} vertices and {len(edges)} edges")

    # Add range measurements
    new_range_edges = add_range_measurements(
        vertices, edges, args.noise_std, args.max_range, args.avg_timestamp_diff)

    # Combine edges
    all_edges = edges + new_range_edges

    # Write output file
    if write_g2o_file(vertices, all_edges, args.output_file):
        print(f"Generated {len(vertices)} vertices and {len(all_edges)} edges")
        print(f"Output written to {args.output_file}")
        sys.exit(0)
    else:
        print("Error writing output file")
        sys.exit(1)


if __name__ == "__main__":
    main()
