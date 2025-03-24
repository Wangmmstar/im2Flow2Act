import os

import numpy as np
from dm_control import mjcf
from scipy.spatial.transform import Rotation as R

from im2flow2act.simulation_env.environment.mujoco.mujocoEnv import (
    TableTopUR5WSG50FinrayEnv,
)
from im2flow2act.simulation_env.environment.utlity.robot_utlity import (
    quat2euler,
)

file_dir = os.path.dirname(os.path.abspath(__file__))


class PickNPlace(TableTopUR5WSG50FinrayEnv):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.task_description = "Put the red mug onto the green plate."
        self.action_z_lower_bound = 0.02
        self.action_z_upper_bound = 0.26
        self.action_x_lower_bound = 0.3
        self.action_x_upper_bound = 0.8
        self.action_y_lower_bound = -0.34
        self.action_y_upper_bound = 0.34
        self.target_pos = None

    def update_task_description(self, task_description):
        self.task_description = task_description

    def setup_model(self):
        world_model = super().setup_model()
        return world_model

    def setup_objs(self, world_model):
        mug_x_bound = [0.42, 0.76]
        mug_y_bound = [-0.1, 0.02]
        bowl_x_bound = [0.42, 0.65]
        bowl_y_bound = [0.24, 0.32]

        mug_pos = (
            np.random.uniform(*mug_x_bound),
            np.random.uniform(*mug_y_bound),
            0.005,
        )
        target_pos = (
            np.random.uniform(*bowl_x_bound),
            np.random.uniform(*bowl_y_bound),
            0.005,
        )
        while self.is_overlapping(mug_pos, target_pos, min_distance=0.15):
            mug_pos = (
                np.random.uniform(*mug_x_bound),
                np.random.uniform(*mug_y_bound),
                0.005,
            )
            target_pos = (
                np.random.uniform(*bowl_x_bound),
                np.random.uniform(*bowl_y_bound),
                0.005,
            )
        self.target_pos = target_pos

        object_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/mujoco_scanned_objects/train/ACE_Coffee_Mug_Kristen_16_oz_cup_scale/model.xml",
                )
            )
        )
        object_model.model = "mug"
        self.add_obj_from_model(
            world_model,
            object_model,
            mug_pos,
            quat=R.from_euler("z", 0).as_quat()[[3, 0, 1, 2]],
            add_freejoint=True,
        )

        object_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/mujoco_scanned_objects/train/Ecoforms_Plate_S20Avocado/model.xml",
                )
            )
        )
        object_model.model = "green_plate"
        self.add_obj_from_model(
            world_model,
            object_model,
            target_pos,
            add_freejoint=True,
        )
        self.selected_object = "mug"

    def is_overlapping(self, obj1_pos, obj2_pos, min_distance=0.1):
        x1, y1, _ = obj1_pos
        x2, y2, _ = obj2_pos
        if np.linalg.norm((x1 - x2, y1 - y2)) <= (min_distance):
            return True
        return False

    def get_random_z_rotation(self):
        return R.from_euler("z", np.random.uniform(0, 2 * np.pi)).as_quat()[
            [3, 0, 1, 2]
        ]

    def add_obj_from_model(
        self, world_model, obj_model, position, quat=None, add_freejoint=False
    ):
        object_site = world_model.worldbody.add(
            "site",
            name=f"{obj_model.model}_site",
            pos=position,
            quat=quat,
            group=3,
        )
        if add_freejoint:
            object_site.attach(obj_model).add("freejoint")
        else:
            object_site.attach(obj_model)

    def step(self, action):
        _, _, _, _, control_info = super().step(action)
        _, obj_link_states, obj_link_contacts = self.get_state()
        observation = self.get_observation(obj_link_states, obj_link_contacts)
        truncated = False
        info = {
            "env": "RealPick",
            "task_description": self.task_description,
        }
        info = info | control_info
        return observation, 0, False, truncated, info

    def check_success(self):
        reference_site = f"{self.selected_object}/reference_site"
        reference_position_vector = self.mj_physics.data.site(
            reference_site
        ).xpos.copy()
        plate_indices = [15, 16, 17]
        target_pos = self.mj_physics.data.qpos.copy()[plate_indices]
        if (
            np.linalg.norm(reference_position_vector - target_pos) < 0.1
            and reference_position_vector[2] < 0.03
        ):
            return True

        return False

    def get_observation(self, obj_link_states, obj_link_contacts):
        proprioceptive_observation = self.get_proprioceptive_observation()
        if self.visual_observation:
            env_observation = self.get_visual_observation()
            return {**env_observation, "proprioceptive": proprioceptive_observation}
        else:
            env_observation = self.get_low_dim_observation(
                obj_link_states, obj_link_contacts
            )
            return np.concatenate([proprioceptive_observation, env_observation])

    def get_reward(self, obj_link_states, obj_link_contacts):
        return 0

    def get_distance(self, position_a, position_b):
        return np.linalg.norm(position_a - position_b)

    def get_low_dim_observation(self, obj_link_states=None, obj_link_contacts=None):
        if obj_link_contacts is None or obj_link_contacts is None:
            _, obj_link_states, obj_link_contacts = self.get_state()
        return np.concatenate(
            [
                self.get_object_qpos(self.selected_object, obj_link_states),
                self.get_object_quant(self.selected_object, obj_link_states),
                # [int(self.check_contact(obj_link_contacts))],
            ]
        ).astype(np.float32)

    # @profile
    def get_object_qpos(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_qpos = []
        for k, v in object_link_state.items():
            object_links_qpos.append(v.pose.position)
        return object_links_qpos[-1].astype(np.float32)

    def get_object_quant(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return object_links_quat[-1].astype(np.float32)

    def get_object_euler(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return quat2euler(object_links_quat[-1].astype(np.float32))



###################### FACTORY #####################################

class FactoryPickAndPlace(TableTopUR5WSG50FinrayEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.task_description = "Pick up the wood panel and place it onto the conveyor belt."
        # Set any new action or workspace limits if needed
        # e.g., self.action_x_lower_bound, etc.

    def setup_model(self):
        # Override the world model to load a factory scene instead of a table scene.
        # For example, use a new XML file that describes the factory environment.
        world_model = mjcf.from_path(
            os.path.normpath(os.path.join(file_dir, "asset/custom/factory.xml"))
        )
        # Optionally add additional lighting or background modifications here.
        
        # Attach the robot model as before.
        robot_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir, "asset/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
                )
            )
        )
        # Remove any keyframes or adjust if necessary.
        del robot_model.keyframe
        # Add the robot model to the world model (similar to the original code).
        robot_site = world_model.worldbody.add(
            "site",
            name="robot_site",
            pos=(0.0, 0.0, 0),
            quat=R.from_euler("z", np.pi / 2).as_quat()[[3, 0, 1, 2]],
        )
        robot_site.attach(robot_model)
        return world_model

    def setup_objs(self, world_model):
        # Define bounds where the wood panel and conveyor are placed.
        wood_panel_x_bound = [0.4, 0.7]
        wood_panel_y_bound = [-0.2, 0.0]
        conveyor_x_bound = [0.7, 1.0]
        conveyor_y_bound = [0.1, 0.3]

        wood_panel_pos = (
            np.random.uniform(*wood_panel_x_bound),
            np.random.uniform(*wood_panel_y_bound),
            0.01,  # Adjust height based on the wood panel dimensions
        )
        target_pos = (
            np.random.uniform(*conveyor_x_bound),
            np.random.uniform(*conveyor_y_bound),
            0.01,  # Target height on the conveyor belt
        )
        # Ensure there is enough separation between initial and target positions.
        while self.is_overlapping(wood_panel_pos, target_pos, min_distance=0.2):
            wood_panel_pos = (
                np.random.uniform(*wood_panel_x_bound),
                np.random.uniform(*wood_panel_y_bound),
                0.01,
            )
            target_pos = (
                np.random.uniform(*conveyor_x_bound),
                np.random.uniform(*conveyor_y_bound),
                0.01,
            )
        self.target_pos = target_pos

        # Load the wood panel model. Replace the asset path with your wood panel asset.
        wood_panel_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/wood_panel/model.xml"  # New asset file for the wood panel
                )
            )
        )
        wood_panel_model.model = "wood_panel"
        self.add_obj_from_model(
            world_model,
            wood_panel_model,
            wood_panel_pos,
            quat=R.from_euler("z", 0).as_quat()[[3, 0, 1, 2]],
            add_freejoint=True,
        )

        # Load the conveyor belt model. This could be a static object that serves as the target.
        conveyor_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/conveyor/model.xml"  # New asset file for the conveyor belt
                )
            )
        )
        conveyor_model.model = "conveyor"
        self.add_obj_from_model(
            world_model,
            conveyor_model,
            target_pos,  # Place it at the target position or at a fixed location defined in your model
            add_freejoint=False,  # Depending on whether the conveyor should move or remain static
        )

        # You can decide which object is selected for manipulation.
        self.selected_object = "wood_panel"

    def is_overlapping(self, obj1_pos, obj2_pos, min_distance=0.2):
        # Similar to the original check, but you can adjust the minimum distance.
        x1, y1, _ = obj1_pos
        x2, y2, _ = obj2_pos
        return np.linalg.norm((x1 - x2, y1 - y2)) <= min_distance

    # You might need to override other methods (like step, get_observation, etc.)
    # if the factory environment has additional requirements.

