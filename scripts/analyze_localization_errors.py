import os
import argparse
import logging
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync, metrics
import matplotlib.pyplot as plt
import csv

"""
python scripts/analyze_localization_errors.py --no_transform_mode
"""


def load_trajectory(file_path):
    """Load trajectory from file"""
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None

    try:
        traj = file_interface.read_tum_trajectory_file(file_path)
        # Convert timestamps to float type to avoid type mismatch issues
        traj.timestamps = np.array(traj.timestamps, dtype=np.float64)
        if len(traj.positions_xyz) > 0:
            return traj

        # If that fails, try our custom format
        timestamps = []
        positions = []
        quaternions = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                logging.info(parts[:2])
                if len(parts) >= 4:  # We need at least index, x, y, z
                    try:
                        # Format: index x y z [qx qy qz qw]
                        # Convert timestamp to float
                        timestamp = float(parts[0])
                        x, y, z = map(float, parts[1:4])
                        timestamps.append(timestamp)
                        positions.append([x, y, z])
                        qx, qy, qz, qw = map(float, parts[4:8])
                        quaternions.append([qx, qy, qz, qw])
                    except ValueError:
                        continue

        if len(timestamps) == 0:
            logging.error(f"No valid trajectory data found in {file_path}")
            return None

        # Convert timestamps to float type to avoid type mismatch issues
        return PoseTrajectory3D(positions_xyz=np.array(positions),
                                orientations_quat_wxyz=np.array(quaternions),
                                timestamps=np.array(timestamps, dtype=np.float64))

    except Exception as e:
        logging.error(f"Error loading trajectory from {file_path}: {str(e)}")
        return None


def calculate_ate(trajectory_est, trajectory_gt):
    """Calculate Absolute Trajectory Error"""
    # Sync trajectories based on timestamps
    traj_est_sync, traj_gt_sync = sync.associate_trajectories(
        trajectory_est, trajectory_gt)

    # Calculate ATE
    ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_full = metrics.APE(metrics.PoseRelation.full_transformation)
    ape_trans.process_data((traj_est_sync, traj_gt_sync))
    ape_full.process_data((traj_est_sync, traj_gt_sync))

    return ape_full


def plot_error_distribution(errors, title="Error Distribution"):
    """Plot error distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_trajectory_comparison(trajectory_est, trajectory_gt, title="Trajectory Comparison"):
    """Plot estimated vs ground truth trajectory"""
    plt.figure(figsize=(12, 8))
    plt.plot(trajectory_est.positions_xyz[:, 0],
             trajectory_est.positions_xyz[:, 1], 'b-', label='Estimated')
    plt.plot(trajectory_gt.positions_xyz[:, 0],
             trajectory_gt.positions_xyz[:, 1], 'r--', label='Ground Truth')
    plt.title(title)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def merge_trajectories(trajectories):
    """
    Merge multiple trajectories into one by concatenating their data.
    """
    if not trajectories:
        return None

    try:
        # Filter out None trajectories
        valid_trajectories = [t for t in trajectories if t is not None]
        if not valid_trajectories:
            return None

        # Use the built-in merge function from evo
        from evo.core.trajectory import merge
        merged_traj = merge(valid_trajectories)
        return merged_traj
    except Exception as e:
        logging.error(f"Error merging trajectories: {e}")
        return None


def compute_transformation(gt_traj, est_traj):
    """Compute the transformation between ground truth and estimated trajectory"""
    try:
        # For a single trajectory pair
        if hasattr(gt_traj, 'timestamps') and hasattr(est_traj, 'timestamps'):
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)
            # Use trajectory's align method to compute the transformation
            r, t, s = traj_est.align(traj_ref, correct_scale=False)
            return {
                'rotation': r,
                'translation': t
            }
        else:
            # We're dealing with a list of trajectories, fall back to using the first one
            # that has a valid match
            for gt, est in zip(gt_traj, est_traj):
                if gt is None or est is None:
                    continue
                try:
                    # Try to compute with this pair
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt, est, max_diff=0.01)
                    r, t, s = traj_est.align(traj_ref, correct_scale=False)
                    return {
                        'rotation': r,
                        'translation': t
                    }
                except Exception:
                    # Try the next pair
                    continue

            # If we get here, no valid transformation was found
            logging.warning(
                "Could not compute a valid transformation from any trajectory pair")
            return None
    except Exception as e:
        logging.error(f"Error computing transformation: {e}")
        return None


def apply_transformation(est_traj, transformation):
    """Apply a pre-computed transformation to an estimated trajectory"""
    if transformation is None:
        return est_traj

    # Instead of using est_traj.copy() which doesn't exist,
    # create a new trajectory object with the same data
    positions = est_traj.positions_xyz.copy()
    orientations = est_traj.orientations_quat_wxyz.copy()
    timestamps = est_traj.timestamps.copy()

    # Apply the transformation
    r = transformation['rotation']
    t = transformation['translation']

    # Apply to all positions
    transformed_positions = []
    for pos in positions:
        transformed_pos = r.dot(pos) + t
        transformed_positions.append(transformed_pos)

    transformed_positions = np.array(transformed_positions)

    # Create a new trajectory object
    transformed_traj = PoseTrajectory3D(
        positions_xyz=transformed_positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )

    return transformed_traj


def calculate_combined_ate_multi_robot(all_gt_trajectories, all_est_trajectories):
    """
    Calculate ATE metrics for combined trajectories in multi-robot setting.
    First associate trajectories, then merge all, then compute single alignment.
    """
    # First associate trajectories, then merge all, then compute single alignment
    all_associated_gt_trajs = []
    all_associated_est_trajs = []

    # Process each robot
    for i, (gt_traj, est_traj) in enumerate(zip(all_gt_trajectories, all_est_trajectories)):
        if gt_traj is None or est_traj is None:
            continue

        # Associate the trajectories (but don't align yet)
        try:
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)

            # Store the associated trajectories
            all_associated_gt_trajs.append(traj_ref)
            all_associated_est_trajs.append(traj_est)

        except Exception as e:
            logging.error(f"Error associating trajectory for robot {i}: {e}")
            continue

    if not all_associated_gt_trajs or not all_associated_est_trajs:
        logging.error("No valid trajectories found for combined analysis")
        return None

    # Merge all associated trajectories
    merged_gt = merge_trajectories(all_associated_gt_trajs)
    merged_est = merge_trajectories(all_associated_est_trajs)

    if merged_gt is None or merged_est is None:
        logging.error("Failed to merge trajectories for combined analysis")
        return None

    # Now compute a single global alignment transformation
    try:
        r, t, s = merged_est.align(merged_gt, correct_scale=False,
                                   correct_only_scale=False)

        global_transform = {
            'rotation': r,
            'translation': t
        }

        # Apply global transformation to all individual trajectories
        transformed_est_trajs = []
        for i, est_traj in enumerate(all_associated_est_trajs):
            # Create proper copy for transformation
            positions = est_traj.positions_xyz.copy()
            orientations = est_traj.orientations_quat_wxyz.copy()
            timestamps = est_traj.timestamps.copy()

            est_traj_copy = PoseTrajectory3D(
                positions_xyz=positions,
                orientations_quat_wxyz=orientations,
                timestamps=timestamps
            )

            # Apply the global transformation
            transformed_traj = apply_transformation(
                est_traj_copy, global_transform)
            transformed_est_trajs.append(transformed_traj)

        # Calculate ATE metrics using transformed trajectories
        # Use merged trajectory for the final metrics
        merged_transformed_est = merge_trajectories(
            transformed_est_trajs)

        ape_trans_final = metrics.APE(
            metrics.PoseRelation.translation_part)
        ape_full_final = metrics.APE(
            metrics.PoseRelation.full_transformation)

        ape_trans_final.process_data(
            (merged_gt, merged_transformed_est))
        ape_full_final.process_data(
            (merged_gt, merged_transformed_est))

        return {
            'trans': ape_trans_final.get_statistic(metrics.StatisticsType.rmse),
            'full': ape_full_final.get_statistic(metrics.StatisticsType.rmse),
            'path_length': merged_gt.path_length,
            'max_error': ape_full_final.get_statistic(metrics.StatisticsType.max),
            'errors': ape_full_final.error
        }

    except Exception as e:
        logging.error(f"Error calculating global metrics: {e}")
        return None


# 新增函数：将结果写入CSV文件
def write_results_to_csv(results, csv_file_path):
    """Write analysis results to a CSV file"""
    try:
        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = ['dataset', 'robot_id', 'rmse', 'mean_error',
                          'std', 'max_error', 'path_length']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)
        logging.info(f"Results written to {csv_file_path}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze localization errors using evo package')
    parser.add_argument('--log_dir', type=str, default='../log',
                        help='Directory containing trajectory files')
    parser.add_argument('--prefix', type=str, default='range_aided_slam_test_2d_diy_dcora',
                        help='Prefix for trajectory files')
    parser.add_argument('--suffix_est', type=str, default='_tum.txt',
                        help='Suffix for estimated trajectory files')
    parser.add_argument('--suffix_gt', type=str, default='_tum_gt.txt',
                        help='Suffix for ground truth trajectory files')
    parser.add_argument('--robots', type=str, nargs='+', default=['A', 'B'],
                        help='List of robot identifiers')
    parser.add_argument('--plot_individual', action='store_true',
                        help='Plot individual trajectory comparisons')
    parser.add_argument('--plot_distribution', action='store_true',
                        help='Plot error distributions')
    parser.add_argument('--no_transform_mode', action='store_true',
                        help='Compare trajectories directly without any transformation')
    parser.add_argument('--output_csv', type=str, default='localization_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--dataset_name', type=str, default='diy',
                        help='Name of the dataset')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # Analyze individual trajectories
    total_errors = []
    all_est_trajectories = []
    all_gt_trajectories = []

    # 存储结果用于写入CSV
    results_data = []

    for robot_id in args.robots:
        # Build file paths
        est_file = os.path.join(
            args.log_dir, f"{args.prefix}_{robot_id}{args.suffix_est}")
        gt_file = os.path.join(
            args.log_dir, f"{args.prefix}_{robot_id}{args.suffix_gt}")

        # Load trajectories
        logging.info(f"Loading trajectories for robot {robot_id}...")
        traj_est = load_trajectory(est_file)
        traj_gt = load_trajectory(gt_file)

        if traj_est is None or traj_gt is None:
            logging.error(f"Failed to load trajectories for robot {robot_id}")
            continue

        # Store trajectories for merged analysis
        all_est_trajectories.append(traj_est)
        all_gt_trajectories.append(traj_gt)

        # Calculate ATE
        logging.info(f"Calculating ATE for robot {robot_id}...")

        if args.no_transform_mode:
            # Direct comparison without any transformation
            try:
                # Sync trajectories based on timestamps
                traj_est_sync, traj_gt_sync = sync.associate_trajectories(
                    traj_est, traj_gt, max_diff=0.01)

                # Calculate ATE without alignment
                ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
                ape_full = metrics.APE(
                    metrics.PoseRelation.full_transformation)
                ape_trans.process_data(
                    (traj_est_sync, traj_gt_sync))
                ape_full.process_data(
                    (traj_est_sync, traj_gt_sync))

                error = ape_full
                rmse = error.get_statistic(metrics.StatisticsType.rmse)
                mean_error = error.get_statistic(metrics.StatisticsType.mean)
                std = error.get_statistic(metrics.StatisticsType.std)
                max_error = error.get_statistic(metrics.StatisticsType.max)
                total_errors.extend(error.error)
            except Exception as e:
                logging.error(
                    f"Error calculating direct ATE for robot {robot_id}: {e}")
                continue
        else:
            error = calculate_ate(traj_est, traj_gt)
            rmse = error.get_statistic(metrics.StatisticsType.rmse)
            mean_error = error.get_statistic(metrics.StatisticsType.mean)
            std = error.get_statistic(metrics.StatisticsType.std)
            max_error = error.get_statistic(metrics.StatisticsType.max)
            total_errors.extend(error.error)

        logging.info(f"Results for robot {robot_id}:")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  Mean error: {mean_error:.4f}")
        logging.info(f"  Standard deviation: {std:.4f}")
        logging.info(f"  Max error: {max_error:.4f}")

        # Store results for CSV output
        results_data.append({
            'dataset': args.dataset_name,
            'robot_id': robot_id,
            'rmse': rmse,
            'mean_error': mean_error,
            'std': std,
            'max_error': max_error,
            'path_length': traj_gt.path_length if traj_gt else 0
        })

        # Plot individual comparison if requested
        if args.plot_individual:
            plot_trajectory_comparison(
                traj_est, traj_gt, f"Robot {robot_id} Trajectory Comparison")
            if args.plot_distribution:
                plot_error_distribution(
                    error.error, f"Robot {robot_id} Error Distribution")

    # Analyze merged trajectories
    merged_results = None
    if len(all_est_trajectories) > 0 and len(all_gt_trajectories) > 0:
        if args.no_transform_mode:
            # Direct comparison for merged trajectories without transformation
            logging.info(
                "\nCalculating ATE for merged trajectories without transformation...")

            # First associate all trajectory pairs without alignment
            all_associated_gt_trajs = []
            all_associated_est_trajs = []

            for i, (gt_traj, est_traj) in enumerate(zip(all_gt_trajectories, all_est_trajectories)):
                if gt_traj is None or est_traj is None:
                    continue

                try:
                    # Associate trajectories without alignment
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt_traj, est_traj, max_diff=0.01)
                    all_associated_gt_trajs.append(traj_ref)
                    all_associated_est_trajs.append(traj_est)
                except Exception as e:
                    logging.error(
                        f"Error associating trajectory for robot {i}: {e}")
                    continue

            if not all_associated_gt_trajs or not all_associated_est_trajs:
                logging.error(
                    "No valid trajectories found for no-transform analysis")
            else:
                try:
                    # Merge associated trajectories
                    merged_gt = merge_trajectories(all_associated_gt_trajs)
                    merged_est = merge_trajectories(all_associated_est_trajs)

                    if merged_est is not None and merged_gt is not None:
                        # Calculate ATE without alignment
                        ape_trans = metrics.APE(
                            metrics.PoseRelation.translation_part)
                        ape_full = metrics.APE(
                            metrics.PoseRelation.full_transformation)
                        ape_trans.process_data((merged_gt, merged_est))
                        ape_full.process_data((merged_gt, merged_est))

                        rmse = ape_trans.get_statistic(
                            metrics.StatisticsType.rmse)
                        full_rmse = ape_full.get_statistic(
                            metrics.StatisticsType.rmse)
                        max_error = ape_full.get_statistic(
                            metrics.StatisticsType.max)

                        logging.info(
                            "Merged Trajectories Results (No Transform Mode):")
                        logging.info(f"  Translation RMSE: {rmse:.4f}")
                        logging.info(f"  Full pose error: {full_rmse:.4f}")
                        logging.info(f"  Max error: {max_error:.4f}")

                        # Store merged results
                        merged_results = {
                            'dataset': args.dataset_name,
                            'robot_id': 'Merged',
                            'rmse': rmse,
                            'mean_error': full_rmse,
                            'std': 0,  # Not available in this mode
                            'max_error': max_error,
                            'path_length': merged_gt.path_length if merged_gt else 0
                        }
                        results_data.append(merged_results)

                        if args.plot_distribution:
                            plot_error_distribution(
                                ape_full.error, "Merged Trajectories Error Distribution (No Transform)")
                    else:
                        logging.error(
                            "Failed to merge trajectories for no-transform analysis")
                except Exception as e:
                    logging.error(
                        f"Error calculating direct ATE for merged trajectories: {e}")
        else:
            # Use multi-robot mode with global transformation
            logging.info(
                "\nCalculating ATE for merged trajectories with global transformation...")
            combined_result = calculate_combined_ate_multi_robot(
                all_gt_trajectories, all_est_trajectories)

            if combined_result:
                logging.info("Merged Trajectories Results (Multi-Robot Mode):")
                logging.info(f"  RMSE: {combined_result['trans']:.4f}")
                logging.info(
                    f"  Full pose error: {combined_result['full']:.4f}")
                logging.info(
                    f"  Max error: {combined_result['max_error']:.4f}")

                # Store merged results
                merged_results = {
                    'dataset': args.dataset_name,
                    'robot_id': 'Merged',
                    'rmse': combined_result['trans'],
                    'mean_error': combined_result['full'],
                    'std': 0,  # Not calculated in this mode
                    'max_error': combined_result['max_error'],
                    'path_length': combined_result['path_length']
                }
                results_data.append(merged_results)

                if args.plot_distribution:
                    plot_error_distribution(
                        combined_result['errors'], "Merged Trajectories Error Distribution (Multi-Robot)")
            else:
                logging.error(
                    "Failed to calculate combined metrics in multi-robot mode")
    else:
        logging.error("\nNot enough data to analyze merged trajectories")

    # Write results to CSV file
    if results_data:
        csv_file_path = os.path.join(args.log_dir, args.output_csv)
        write_results_to_csv(results_data, csv_file_path)
    else:
        logging.warning("No results to write to CSV file")

    logging.info("\nAnalysis complete")


if __name__ == '__main__':
    main()
