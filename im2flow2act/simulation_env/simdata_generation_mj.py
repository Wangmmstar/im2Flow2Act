import os
import numpy as np
import pickle
import gymnasium as gym
from spacemouse import Spacemouse
from im2flow2act.simulation_env.environment.wrapper.record import RecordWrapper
from im2flow2act.simulation_env.environment.utility.env_utility import build_evaluation_environment


############ initialize spacemouse and environment ###################


# Initialize SpaceMouse
spacemouse = Spacemouse(deadzone=0.1)  # Adjust deadzone if needed
spacemouse.start()  # Start listening to SpaceMouse input

# Set up the environment
# Path to save dataset
dataset_path = "/path/to/save/dataset"
os.makedirs(dataset_path, exist_ok=True)

# Set up the environment configuration (Modify if needed)
env_cfg = {
    "eval_render_res": (224, 224),
    "eval_render_fps": 30,
    "eval_camera_ids": [0],  # Use first camera
    "eval_store_path": dataset_path,
}

# Load environment state
info = {"env": "RealPick"}  # Example task (Change based on your need)
qpos = np.zeros(6)  # Initial joint positions
qvel = np.zeros(6)  # Initial joint velocities

# Initialize Mujoco environment
env = build_evaluation_environment(env_cfg, info, qpos, qvel)
env = RecordWrapper(env, 224, 224, 30, [0], dataset_path)

# Initialize SpaceMouse
spacemouse = Spacemouse(deadzone=0.1)  # Adjust sensitivity if needed
spacemouse.start()  # Start listening to SpaceMouse input

# Set motion scaling factor (Modify if needed)
action_scale = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])  # Scale for xyz + rotation



############### collect and store actions ##########################

# Store recorded data
action_list = []
obs_list = []
reward_list = []
done_list = []

max_steps = 200  # Limit episode length

# Reset environment
obs = env.reset()

for step in range(max_steps):
    # Get SpaceMouse movement
    motion = spacemouse.get_motion_state_transformed()  # [x, y, z, roll, pitch, yaw]

    # Convert motion to action
    action = np.zeros(7)  # [x, y, z, roll, pitch, yaw, gripper]
    action[:6] = motion[:6] * action_scale  # Scale motion to match robot's range
    action[6] = 1.0 if spacemouse.is_button_pressed(0) else 0.0  # Open/close gripper

    # Apply action to the environment
    obs, reward, done, truncated, info = env.step(action)

    # Store data
    action_list.append(action)
    obs_list.append(obs)
    reward_list.append(reward)
    done_list.append(done)

    # Stop recording if task is done
    if done or truncated:
        break

# Stop SpaceMouse
spacemouse.stop()

##################3️⃣ Save the Collected Data for Training##################

# Save the collected demonstration data
save_path = os.path.join(dataset_path, "expert_demo.pkl")
with open(save_path, "wb") as f:
    pickle.dump({
        "observations": obs_list,
        "actions": action_list,
        "rewards": reward_list,
        "dones": done_list
    }, f)

print(f"✅ Expert demonstration saved at: {save_path}")


