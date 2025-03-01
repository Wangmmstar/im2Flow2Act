# visualizing optical flows from a dataset.  It does this by running another script, viz_point_tracking.py, multiple times using subprocess.Popen().

import os
import subprocess

#nw path
def main():
    dev_path = os.getenv("DEV_PATH")
    viz_pathes = [
        f"{dev_path}/im2Flow2Act/data/simulated_play/rigid",
        f"{dev_path}/im2Flow2Act/data/simulated_play/articulated",
        f"{dev_path}/im2Flow2Act/data/simulated_play/deformable",
    ]
    print(viz_pathes)

    """
    -1: Use precomputed moving masks in the dataset (recommended for simulated data).
    >0: Visualize keypoints that moved more than the specified threshold.
    0: Used for real-world data with bounding boxes.
    """

    # viz_thresholds = [0]  # use it for viz_bbox for realworld dataset
    viz_thresholds = [-1] * len(viz_pathes)  # use it for viz_sam for sim dataset  return [X,X,X] three values loop three data paths
    processes = []
    for v in viz_pathes:
        for thresh in viz_thresholds:
            process = subprocess.Popen( #calls viz_point_tracking.py using subprocess.Popen(), passing several arguments, run several processes in parallel, each file as separate process
                [
                    "python",
                    "viz_point_tracking.py", # ccall this file
                    "--viz_num",
                    str(20),   # Process 20 data samples.
                    "--data_buffer_path",
                    v,  # Split into two separate strings
                    "--viz_threshold",
                    str(thresh),  # Also split, and convert thresh to string
                    "--viz_num_point",
                    str(-1), #Visualizes all keypoints.
                    # "--draw_line",
                    "--viz_offset",
                    str(0),
                    "--viz_save_path",
                    f"{dev_path}/im2Flow2Act/experiment/visual_pt", # Saves results to experiment/visual_pt/.
                    "--viz_sam",
                    # "--viz_bbox",
                ],
            )
            processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
