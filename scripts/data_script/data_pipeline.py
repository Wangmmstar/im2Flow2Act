# This script prepares, processes, and refines dataset for training a flow-conditioned diffusion policy. 
# It automates multiple data processing steps such as converting raw data, filtering points, 
# generating segmentation masks, tracking key points,and labeling objects.


import subprocess

import hydra
import omegaconf


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="data_pipeline",
)
def main(cfg: omegaconf.DictConfig):  #omegaconf → Hydra’s configuration format (DictConfig).
    avaliable_gpu = f"'{cfg.avaliable_gpu}'"
    simulation_herustic_filter = f"'{cfg.simulation_herustic_filter}'" # Filtering strategy for simulations.

    # Define the commands
    commands = [
        ######## 1. Convert Simulation Data → Structured Dataset ##########
        
        f"python convert_all_simulation_dataset.py "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"store_path={cfg.store_path} "
        f"dataset={cfg.dataset} "
        f"downsample_ratio={cfg.downsample_ratio} "
        f"n_sample_frame={cfg.n_sample_frame}",
        # Runs convert_all_simulation_dataset.py
        # Passes dataset location, storage path, and preprocessing parameters.
        ###########################################

        ############ 2. Generate Point Tracking Information ##################

        f"python generate_all_point_tracking.py "   # raw point tracking without SAM
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"num_points={cfg.num_points} "
        f"sam_iterative=False "
        f"dbscan_bbox=False",
        # Runs generate_all_point_tracking.py   
        # Uses the specified GPU (cfg.avaliable_gpu)
        # Tracks num_points in data_buffer_path.
        ###########################################

        ############ 3. Select & Filter Important Points ################## Removes stationary or irrelevant points.
        
        f"python point_selection.py "   # without SAM; initial movement-based points
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f"moving_threshold_args.is_sam=False "
        f"moving_threshold_args.point_move_thresholds=[{cfg.point_move_thresholds}]",
        # ##########################################

         ############ 4. Generate Object Segmentation Masks (Using SAM) ################## Masks robots, objects, and the background.
        
        f"python generate_all_sam_mask.py "
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path}",
        # ###########################################
         
        ############ 5. Refine Point Tracking with Iterative Updates ################## 
        
        f"python generate_all_point_tracking.py "   # with SAM tracking + adiitional filtering
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"sam_iterative=True "
        f'"sam_iterative_additional_kwargs.from_grid={cfg.from_grid}" '
        f"from_bbox={cfg.from_bbox} "
        f"simulation_herustic_filter={simulation_herustic_filter}",
        ##########################################

        ############ refine point selection ################## 

        
        f"python point_selection.py "   # with SAM, + additional filters
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f"moving_threshold_args.is_sam=True "
        f"moving_threshold_args.point_move_thresholds=[{cfg.point_move_thresholds}] "
        f"moving_threshold_args.simulation_herustic_filter={simulation_herustic_filter}",
        ###########################################

        ############ 7. Label Robot Interaction Areas ################## 
        
        f"python point_annotation.py "
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f'mask="robot_mask" '
        f'"robot_mask_args.zero_robot_mask={cfg.zero_robot_mask}" '  #Removes robot artifacts if zero_robot_mask=True.
        f"robot_mask_args.is_sam=True "
        f'"robot_mask_args.simulation_herustic={cfg.simulation_herustic}" '
        f'"robot_mask_args.simulation_herustic_patial={cfg.simulation_herustic_patial}"',
    ]

    # Execute each command sequentially
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)  # shell=True → Runs as a shell command.  #check=True → Raises an error if any command fails.


if __name__ == "__main__":
    main()



""" 
# Script Name                         Times Called	Purpose
# 1️⃣	convert_all_simulation_dataset.py	1	        Converts simulation dataset into required format.
# 2️⃣	generate_all_point_tracking.py	    2	        Tracks key points in the dataset (first pass without SAM, second with SAM).
# 3️⃣	point_selection.py	                2	        Selects the most relevant points (one for general, one for SAM).
# 4️⃣	generate_all_sam_mask.py        	1	        Generates segmentation masks using SAM.
# 5️⃣	point_annotation.py                	1	        Annotates data points, including filtering for robot-related points.
"""
