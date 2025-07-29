import utils

# verify_g2o.py
"""
This script reads g2o files, extracts VERTEX_SE2 and EDGE_RANGE data,
validates the EDGE_RANGE data, and saves the results.

python verify_g2o.py ../data/tiers.pyfg tiers_log.txt
"""


def read_g2o_file(filename):
    """
    Read a g2o file and extract VERTEX_SE2 and EDGE_RANGE data.
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

                # elif parts[0] == 'VERTEX_XY':
                #     # Process VERTEX_XY line
                #     if len(parts) < 4:
                #         print(f"Warning: Incomplete VERTEX_XY line: {line}")
                #         continue

                #     # Format: VERTEX_XY sym x y
                #     sym = parts[1]
                #     x = float(parts[2])
                #     y = float(parts[3])

                #     vertices.append({
                #         'type': 'VERTEX_XY',
                #         'sym': sym,
                #         'x': x,
                #         'y': y,
                #     })

                elif parts[0] == 'EDGE_RANGE':
                    # Process EDGE_RANGE line
                    if len(parts) < 6:
                        print(f"Warning: Incomplete EDGE_RANGE line: {line}")
                        continue

                    # Format: EDGE_RANGE ts sym1 sym2 range cov
                    timestamp = float(parts[1])
                    sym1 = parts[2]
                    sym2 = parts[3]
                    range_val = float(parts[4])
                    # Assuming single value for simplicity
                    covariance = float(parts[5])

                    edges.append({
                        'type': 'EDGE_RANGE',
                        'timestamp': timestamp,
                        'sym1': sym1,
                        'sym2': sym2,
                        'range': range_val,
                        'covariance': covariance
                    })

    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, None

    return vertices, edges


def validate_edge_range(edges, vertices):
    """
    Validate EDGE_RANGE data by comparing the range with the actual distance between vertices.
    Returns a list of validation results for each edge.
    """
    results = []

    # Create a mapping from (robot_id, state_id) to vertex index for quick lookup
    vertex_map = {}
    for i, vertex in enumerate(vertices):
        try:
            robot_id, state_id = utils.get_robot_and_state_id_from_symbol(
                vertex['sym'])
            vertex_map[(robot_id, state_id)] = i
        except Exception as e:
            print(
                f"Warning: Could not parse symbol {vertex['sym']} for vertex at index {i}: {str(e)}")

    for i, edge in enumerate(edges):
        result = {
            'index': i,
            'valid': True,
            'errors': [],
            'edge': edge,
            'timestamp': edge['timestamp']  # 新增：保存时间戳
        }

        # Check range is positive
        if edge['range'] <= 0:
            result['valid'] = False
            result['errors'].append(f"Range must be positive: {edge['range']}")

        # Check covariance is non-negative
        if edge['covariance'] < 0:
            result['valid'] = False
            result['errors'].append(
                f"Covariance must be non-negative: {edge['covariance']}")

        # Parse symbols
        try:
            robot_id1, state_id1 = utils.get_robot_and_state_id_from_symbol(
                edge['sym1'])
            robot_id2, state_id2 = utils.get_robot_and_state_id_from_symbol(
                edge['sym2'])

            # Store parsed IDs for reference
            result['robot_id1'] = robot_id1
            result['state_id1'] = state_id1
            result['robot_id2'] = robot_id2
            result['state_id2'] = state_id2

            # Find corresponding vertices
            vertex_index1 = vertex_map.get((robot_id1, state_id1))
            vertex_index2 = vertex_map.get((robot_id2, state_id2))

            if vertex_index1 is None or vertex_index2 is None:
                result['valid'] = False
                result['errors'].append(
                    "Could not find corresponding vertices for one or both symbols")
            else:
                # Calculate actual distance between vertices
                vertex1 = vertices[vertex_index1]
                vertex2 = vertices[vertex_index2]
                actual_distance = (
                    (vertex1['x'] - vertex2['x']) ** 2 + (vertex1['y'] - vertex2['y']) ** 2) ** 0.5

                # Calculate error
                error = abs(edge['range'] - actual_distance)
                result['actual_distance'] = actual_distance
                result['error'] = error

                # Add timestamp information for sym1 and sym2
                result['timestamp_sym1'] = vertex1.get('timestamp', 'N/A')
                result['timestamp_sym2'] = vertex2.get('timestamp', 'N/A')

                # Add error information to results
                result['errors'].append(
                    f"Calculated distance: {actual_distance}, Edge range: {edge['range']}, Error: {error}")

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Symbol parsing error: {str(e)}")

        results.append(result)

    return results


def write_validation_results(results, output_file):
    """
    Write validation results to a file.
    """
    try:
        with open(output_file, 'w') as f:
            f.write("# EDGE_RANGE Validation Results\n")
            f.write(f"Total edges checked: {len(results)}\n\n")

            valid_count = 0
            for result in results:
                if result['valid']:
                    valid_count += 1
                    f.write(f"Edge {result['index']}: VALID\n")
                else:
                    f.write(f"Edge {result['index']}: INVALID\n")

                # 输出时间戳
                f.write(f"  Timestamp: {result.get('timestamp', 'N/A')}\n")
                # 输出sym1时间戳
                f.write(
                    f"  Timestamp_sym1: {result.get('timestamp_sym1', 'N/A')}\n")
                # 输出sym2时间戳
                f.write(
                    f"  Timestamp_sym2: {result.get('timestamp_sym2', 'N/A')}\n")
                f.write(
                    f"  sym1: {result['edge']['sym1']} (robot {result['robot_id1']}, state {result['state_id1']})\n")
                f.write(
                    f"  sym2: {result['edge']['sym2']} (robot {result['robot_id2']}, state {result['state_id2']})\n")
                f.write(
                    f"  range: {result['edge']['range']}, covariance: {result['edge']['covariance']}\n")
                f.write(
                    f"  Calculated distance: {result.get('actual_distance', 'N/A')}, Error: {result.get('error', 'N/A')}\n\n")

                if not result['valid']:
                    f.write("  Errors:\n")
                    for error in result['errors']:
                        f.write(f"    - {error}\n")
                    f.write("\n")

            f.write(
                f"Summary: {valid_count} out of {len(results)} edges are valid.\n")

        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python verify_g2o.py <input_g2o_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read g2o file
    vertices, edges = read_g2o_file(input_file)
    if vertices is None or edges is None:
        print("Error reading input file")
        sys.exit(1)

    print(f"Read {len(vertices)} vertices and {len(edges)} edges")

    # Validate edges
    results = validate_edge_range(edges, vertices)

    # Write results to file
    if write_validation_results(results, output_file):
        print(f"Validation completed. Results written to {output_file}")
        # Count invalid edges
        invalid_count = len(results) - sum(1 for r in results if r['valid'])
        if invalid_count > 0:
            print(f"NOTE: {invalid_count} invalid edges were found")
            sys.exit(1)
        else:
            print("All edges are valid")
            sys.exit(0)
    else:
        print("Error writing output file")
        sys.exit(1)

# Add more functionality here...
