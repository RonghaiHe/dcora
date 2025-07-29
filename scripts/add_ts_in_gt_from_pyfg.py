import os
import sys
import numpy as np

"""
python add_ts_in_gt_from_pyfg.py ../data/range_aided_slam_test_2d.pyfg ../log/range_aided_slam_test_2d_dcora_A_gt.txt ../log/range_aided_slam_test_2d_dcora_A_tum_gt.txt --delete-original
"""


def parse_vertex_line(line):
    """Parse a vertex line from .pyfg file and extract timestamp and symbol."""
    parts = line.strip().split()
    if parts[0] != 'VERTEX_SE2':
        return None, None

    timestamp = float(parts[1])
    symbol = parts[2]
    return timestamp, symbol


def read_pyfg_timestamps(pyfg_path):
    """Read timestamps from .pyfg file and map to symbols."""
    timestamps = {}
    with open(pyfg_path, 'r') as f:
        for line in f:
            if line.startswith('VERTEX_SE2'):
                timestamp, symbol = parse_vertex_line(line)
                if timestamp is not None and symbol is not None:
                    timestamps[symbol] = timestamp
    return timestamps


def process_gt_file(gt_path, timestamps, output_path):
    """Process ground truth file and add timestamps."""
    with open(gt_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('#'):
                outfile.write(line)
                continue

            parts = line.strip().split()
            if not parts:
                continue

            index = parts[0]
            # Find the symbol corresponding to this index
            symbol = None
            for key in timestamps:
                if key.endswith(index):
                    symbol = key
                    break

            if symbol in timestamps:
                timestamp = timestamps[symbol]
                # Replace index with timestamp
                new_line = f"{timestamp} {' '.join(parts[1:])}\n"
                outfile.write(new_line)
            else:
                # If no timestamp found, keep original line
                outfile.write(line)


def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print(
            "Usage: python add_ts_in_gt_from_pyfg.py <pyfg_file> <gt_file> <output_file> [--delete-original]")
        sys.exit(1)

    pyfg_file = sys.argv[1]
    gt_file = sys.argv[2]
    output_file = sys.argv[3]
    delete_original = '--delete-original' in sys.argv

    # Read timestamps from .pyfg file
    print(f"Reading timestamps from {pyfg_file}...")
    timestamps = read_pyfg_timestamps(pyfg_file)

    # Process ground truth file
    print(f"Processing ground truth file {gt_file}...")
    process_gt_file(gt_file, timestamps, output_file)

    # Delete original file if requested
    if delete_original and os.path.exists(gt_file):
        print(f"Deleting original file {gt_file}...")
        os.remove(gt_file)

    print(f"Output written to {output_file}")
    if delete_original:
        print("Original file deleted.")


if __name__ == "__main__":
    main()
