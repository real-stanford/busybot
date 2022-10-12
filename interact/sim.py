import random
import json
import math
import os
from shutil import move
import time
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import scipy.ndimage as sn

import utils
from board import Board

BOARD_COLORS = [[0.9, 0.8, 0.7, 1], [0.8, 0.7, 0.6, 1], [0.8, 0.6, 0.5, 1], [0.9, 0.7, 0.5, 1],
                [0.9, 1, 1, 1], [0.7, 0.6, 0.6, 1], [0.6, 0.4, 0.2, 1], [0.8, 0.5, 0.2, 1]]
BOARD_TEXTURE_FILES = ['table-1.png', 'table-2.png', 'table-3.png', 'table-4.png', 'table-5.png']

"""
Generate all possible functional relation pairs as a dictionary
    key: (trigger_body_id, trigger_joint_id)
    value: relation_type (discrete/continuous), trigger_info, responder_info
"""
def get_func_rel_dict(obj_ids, obj_types, body_ids):
    rel_dict = {}
    single_trigger, single_responder = [], []
    multi_trigger, multi_responder = [], []

    for i in range(len(obj_ids)):
        obj_id, obj_type, body_id = obj_ids[i], obj_types[i], body_ids[i]
        obj_json = json.load(open(os.path.join('../assets/objects', obj_type, obj_id, 'object_meta_info.json')))
        for j in range(len(obj_json["Cause"])):
            joint_id = j+1
            if obj_json["Cause"][j]["IsSmallCause"]:
                single_trigger.append((obj_type, obj_id, body_id, joint_id))
            if obj_json["Cause"][j]["IsLargeCause"]:
                multi_trigger.append((obj_type, obj_id, body_id, joint_id))
        for j in range(len(obj_json["Effect"])):
            joint_id = j+1
            if obj_json["Effect"][j]["IsSingleEffect"]:
                single_responder.append((obj_type, obj_id, body_id, joint_id))
            if obj_json["Effect"][j]["IsMultiEffect"]:
                multi_responder.append((obj_type, obj_id, body_id, joint_id))

    for trigger_info in single_trigger:
        rel_pairs = []
        for responder_info in single_responder:
            rel_pairs.append(("discrete", trigger_info, responder_info))
        rel_dict[(trigger_info[2], trigger_info[3])] = rel_pairs

    for trigger_info in multi_trigger:
        rel_pairs = []
        for responder_info in multi_responder:
            rel_pairs.append(("continuous", trigger_info, responder_info))
        rel_dict[(trigger_info[2], trigger_info[3])] = rel_pairs

    return rel_dict


def get_rel_matrix(rel_pairs, num_objects):
    rel_matrix = np.zeros((num_objects, num_objects), dtype=np.int32)
    for relation_type, trigger_info, responder_info in rel_pairs:
        if relation_type == 'discrete':
            rel_matrix[responder_info[2]-1, trigger_info[2]-1] = 1
        if relation_type == 'continuous':
            rel_matrix[responder_info[2]-1, trigger_info[2]-1] = 1
    return rel_matrix


class PybulletSim(object):
    def __init__(self, gui_enabled, action_distance):
        """Pybullet simulation initialization.
        Args:
            gui_enables: bool
            action_distance: float
        """
        if gui_enabled:
            self.bc = bc.BulletClient(connection_mode=pybullet.GUI)
            self.gui = True
        else:
            self.bc = bc.BulletClient(connection_mode=pybullet.DIRECT)
            self.gui = False
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._action_distance = action_distance
        self.ee_urdf = '../assets/suction/suction_with_mount_no_collision.urdf'
        self._mount_force = 80000
        self._mount_speed = 0.02
        self._constraint_force = 40000
        self._mount_base_position = np.array([0, -3, 3])

        self.body_ids = list()
        self.transform_start_to_link = None
        self.link_id = None

        # RGB-D camera setup
        self._scene_cam_image_size = (480, 640)
        self._scene_cam_z_near = 0.01
        self._scene_cam_z_far = 10.0
        self._scene_cam_fov_w = 69.40
        self._scene_cam_focal_length = (float(self._scene_cam_image_size[1]) / 2) / np.tan(
            (np.pi * self._scene_cam_fov_w / 180) / 2)
        self._scene_cam_fov_h = (math.atan(
            (float(self._scene_cam_image_size[0]) / 2) / self._scene_cam_focal_length) * 2 / np.pi) * 180
        self._scene_cam_projection_matrix = self.bc.computeProjectionMatrixFOV(
            fov=self._scene_cam_fov_h,
            aspect=float(self._scene_cam_image_size[1]) / float(self._scene_cam_image_size[0]),
            nearVal=self._scene_cam_z_near, farVal=self._scene_cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._scene_cam_intrinsics = np.array(
            [[self._scene_cam_focal_length, 0, float(self._scene_cam_image_size[1]) / 2],
             [0, self._scene_cam_focal_length, float(self._scene_cam_image_size[0]) / 2],
             [0, 0, 1]])

    # Get latest RGB-D image from scene camera
    def get_scene_cam_data(self, cam_position, cam_lookat, cam_up_direction):
        cam_view_matrix = np.array(self.bc.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)).reshape(4,4).T
        cam_pose_matrix = np.linalg.inv(cam_view_matrix)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
        camera_data = self.bc.getCameraImage(
            self._scene_cam_image_size[1],
            self._scene_cam_image_size[0],
            self.bc.computeViewMatrix(cam_position, cam_lookat, cam_up_direction),
            self._scene_cam_projection_matrix,
            shadow=1,
            flags=self.bc.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.bc.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_pixels = np.array(camera_data[2]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1], 4))
        color_image = rgb_pixels[:, :, :3].astype(np.float32) / 255.0
        z_buffer = np.array(camera_data[3]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1]))
        depth_image = (2.0 * self._scene_cam_z_near * self._scene_cam_z_far) / (
                self._scene_cam_z_far + self._scene_cam_z_near - (2.0 * z_buffer - 1.0) * (
                self._scene_cam_z_far - self._scene_cam_z_near))
        segmentation_mask = np.array(camera_data[4], int).reshape(self._scene_cam_image_size)

        return color_image, depth_image, segmentation_mask, cam_pose_matrix, cam_view_matrix

    # Oracle (joint state supervision): compute joint reward based on joint difference
    def get_single_step_joint_reward(self):
        joint_state = self.joint_states[(self.body_id, self.link_id)]
        joint_value = self.bc.getJointState(self.body_id, self.link_id)[0]
        reward = abs(joint_value - joint_state['cur_val'])
        move_flag = False

        joint_info = self.bc.getJointInfo(self.body_id, self.link_id)
        if joint_info[2] == self.bc.JOINT_REVOLUTE and reward > 0.1:
            move_flag = True
        if joint_info[2] == self.bc.JOINT_PRISMATIC and reward > 0.05:
            move_flag = True
        if reward * 3.5 > (joint_state['max_val'] - joint_state['min_val']):
            move_flag = True

        reward = 1 if move_flag else 0

        self.scene_images['last_val'] = self.scene_images['cur_val']
        self.scene_images['cur_val'] = self.get_observation()['image']

        joint_state['last_val'] = joint_state['cur_val']
        joint_state['cur_val'] = joint_value

        self.last_move_flag = move_flag

        return reward, move_flag

    # Compute image difference based of number of changing pixels
    def get_single_step_image_difference_reward(self):
        joint_state = self.joint_states[(self.body_id, self.link_id)]
        joint_value = self.bc.getJointState(self.body_id, self.link_id)[0]

        # if image difference is larger than a threshold, reward = 1. Otherwise, reward = 0
        cur_rgb_image = self.get_observation()['image']
        prev_rgb_image = self.scene_images['cur_val']
        dif_pixels = 0
        for v in range(cur_rgb_image.shape[0]):
            for u in range(cur_rgb_image.shape[1]):
                if np.any(np.abs(cur_rgb_image[v, u] - prev_rgb_image[v, u]) > 1e-2):
                    dif_pixels += 1

        threshold = 1000
        move_flag = True if dif_pixels > threshold else False
        reward = 1 if move_flag else 0

        self.scene_images['last_val'] = self.scene_images['cur_val']
        self.scene_images['cur_val'] = self.get_observation()['image']

        joint_state['last_val'] = joint_state['cur_val']
        joint_state['cur_val'] = joint_value

        return reward, move_flag

    def get_suction_target_position(self):
        link_state = self.bc.getLinkState(self.body_id, self.link_id)
        link_pos, link_quat = link_state[0], link_state[1]
        suction_position = np.array(self.bc.multiplyTransforms(link_pos, link_quat, self.transform_start_to_link[0],
                                                               self.transform_start_to_link[1])[0])
        return suction_position

    def _move_to(self, target_position, speed_ratio=1):
        speed_revolute = 0.05 * speed_ratio
        target_position = np.array(target_position) - self._mount_base_position

        step_num = 100
        for _ in range(step_num):
            self.bc.setJointMotorControlArray(
                self._suction_gripper,
                [0, 1, 2],
                self.bc.POSITION_CONTROL,
                targetPositions=target_position,
                forces=[self._mount_force] * 3,
                positionGains=[speed_revolute] * 3
            )

            self.bc.stepSimulation()

    def get_observation(self):
        self.cam_position = np.array([0, -0.1, 4])
        self.cam_lookat = np.array([0, 0, 0])
        self.cam_up_direction = np.array([0, 0, 1])
        self.color_image, self.depth_image, self.segmentation_mask, self.cam_pose_matrix, self.cam_view_matrix = \
            self.get_scene_cam_data(self.cam_position, self.cam_lookat, self.cam_up_direction)

        xyz_pts, color_pts, segmentation_pts = utils.get_pointcloud(self.depth_image, self.color_image,
                                                                    self.segmentation_mask, self._scene_cam_intrinsics,
                                                                    self.cam_pose_matrix)
        cam_pts_img = xyz_pts.reshape(self.color_image.shape)

        self.xyz_pts = xyz_pts
        self.body_id_pts = segmentation_pts & ((1 << 24) - 1)
        self.link_id_pts = (segmentation_pts >> 24) - 1

        sigma = 2
        dh = sn.gaussian_filter1d(
            cam_pts_img, sigma=sigma, 
            axis=0, order=1, mode='nearest')
        dw = sn.gaussian_filter1d(
            cam_pts_img, sigma=sigma, 
            axis=1, order=1, mode='nearest')
            
        dh_normal = dh / np.linalg.norm(dh, axis=-1, keepdims=True)
        dw_normal = dw / np.linalg.norm(dw, axis=-1, keepdims=True)
        self.normals = np.cross(dh_normal, dw_normal, axis=-1)

        self.image = np.concatenate([
            self.color_image,
            self.normals.reshape(self.color_image.shape),
            self.depth_image[:, :, np.newaxis]
        ], axis=2).astype(np.float32)

        observation = {
            'image': self.image,
            'segmentation_mask': self.segmentation_mask,
            'depth_image': self.depth_image,
            'cam_intrinsics': self._scene_cam_intrinsics,
            'cam_pose_matrix': self.cam_pose_matrix,
            'cam_view_matrix': self.cam_view_matrix
        }
        return observation

    def get_scene_state(self):
        scene_state = {
            'body_ids': self.body_ids,
            'obj_types': self.obj_types,
            'obj_ids': self.obj_ids,
            'base_positions': self.base_positions,
            'orientations': self.orientations,
            'scales': self.scales,
            'causal_pairs': self.rel_pairs,
            'joint_states': self.joint_states,
            "movable_links": self.movable_links,
            "is_continuous_cause": self.is_multi_trigger,
            'states': self.get_states(),
            'scene_images': self.scene_images,
            'file_idx': self.file_idx,
            'color_idx': self.color_idx
        }
        return scene_state

    def get_gt_position(self, gt_body_id, gt_link_id):
        while True:
            random_idx = np.random.choice(len(self.body_id_pts))
            body_id = int(self.body_id_pts[random_idx])
            link_id = int(self.link_id_pts[random_idx])
            if body_id == gt_body_id and link_id == gt_link_id:
                break
        w, h = np.unravel_index(random_idx, self.depth_image.shape)
        return [w, h]

    def reset_init_joint_state(self):
        for body_id, joint_id in self.joint_states.keys():
            init_val = self.joint_states[(body_id, joint_id)]['init_val']
            self.bc.resetJointState(body_id, joint_id, init_val)
            self.joint_states[(body_id, joint_id)]['cur_val'] = init_val

            # add functional relations for initial states
            affected_pairs = []
            for rel_pair in self.rel_pairs:
                if rel_pair[1][2] == body_id and rel_pair[1][3] == joint_id:
                    affected_pairs.append(rel_pair)
            self.add_relations(affected_pairs, body_id, joint_id)

        self.bc.stepSimulation()
        self.scene_images['init_val'] = self.get_observation()['image']
        self.scene_images['cur_val'] = self.get_observation()['image']

    def reset(self, scene_state=None, goal_conditioned=False, **kwargs):
        """Remove all objects; load a new scene; return observation
        Args:
            scene_state
            goal_conditioned (true/false)
        Returns:
            observation: dict
        """
        self.bc.resetSimulation()
        self.bc.setGravity(0, 0, -9.8)
        self._suction_gripper = None
        self.goal_conditioned = goal_conditioned

        # Load board
        # Assign a random brown color and a wooden texture to board if not reloading
        if scene_state is None:
            self.file_idx = np.random.choice(len(BOARD_TEXTURE_FILES))
            self.color_idx = np.random.choice(len(BOARD_COLORS))
        else:
            self.file_idx = scene_state['file_idx']
            self.color_idx = scene_state['color_idx']
    
        self.board = Board(bc=self.bc, board_length=4, board_width=4,
                        texture_file=BOARD_TEXTURE_FILES[self.file_idx], board_color=BOARD_COLORS[self.color_idx])

        if scene_state is None:
            # Initialization
            self.joint_states = dict()
            self.scene_images = dict()

            # Load board
            categories = ["Switch", "Lamp", "Door", "Toy"]
            num_cause = np.random.choice([2, 3])
            total_effect = np.random.choice([5, 6]) # [6, 7]
            num_effect = np.random.choice(range(3, total_effect))
            object_num = [num_cause, num_effect, total_effect - num_effect, 1]

            obj_ids, obj_types, body_ids, positions, orientations, scales, \
            is_multi_trigger, movable_links = self.board.reset(
                categories=categories,
                object_num=object_num,
                board_type=kwargs['board_type'],
                instance_type=kwargs['instance_type'])
            while len(obj_ids) != len(body_ids):
                obj_ids, obj_types, body_ids, positions, orientations, scales, \
                is_multi_trigger, movable_links = self.board.reset(
                    categories=categories,
                    object_num=object_num,
                    board_type=kwargs['board_type'],
                    instance_type=kwargs['instance_type'])
            
            self.body_ids = body_ids
            self.obj_types = obj_types
            self.obj_ids = obj_ids
            self.base_positions = positions
            self.orientations = orientations
            self.scales = scales
            self.is_multi_trigger = is_multi_trigger
            self.movable_links = movable_links
        else:
            self.board.reload(scene_state)
            self.body_ids = scene_state["body_ids"]
            self.obj_types = scene_state["obj_types"]
            self.obj_ids = scene_state["obj_ids"]
            self.base_positions = scene_state["base_positions"]
            self.orientations = scene_state["orientations"]
            self.scales = scene_state["scales"]
            self.is_multi_trigger = scene_state["is_continuous_cause"]
            self.movable_links = scene_state["movable_links"]

        # ======================Sample inter-object functional relations============================
        # Assort body ids so that multi-stage triggers are first assigned
        if not self.goal_conditioned:
            assorted_idx = []
            for i in range(len(self.body_ids)):
                if self.is_multi_trigger[i]:
                    assorted_idx.insert(0, i)
                else:
                    assorted_idx.append(i)
            assorted_body_ids = [self.body_ids[idx] for idx in assorted_idx]
            assorted_movable_links = [self.movable_links[idx] for idx in assorted_idx]

            rel_dict = get_func_rel_dict(self.obj_ids, self.obj_types, self.body_ids)
            rel_pairs = []
            for i, body_id in enumerate(assorted_body_ids):
                for j in range(len(assorted_movable_links[i])):
                    joint_id = j+1
                    if (body_id, joint_id) in rel_dict:
                        num_pairs = len(rel_dict[(body_id, joint_id)])
                        if num_pairs == 0:
                            continue
                        add_flag = False # ensure every trigger is matched to a responder effect
                        # while add_flag is False:
                        for _ in range(100):
                            idx = np.random.choice(num_pairs)
                            sampled_rel_pair = rel_dict[(body_id, joint_id)][idx]
                            # filter out many to one relations
                            exist = False
                            for rel_pair in rel_pairs:
                                if sampled_rel_pair[2][2] == rel_pair[2][2]:
                                    exist = True

                            if not exist:
                                rel_pairs.append(sampled_rel_pair)
                                add_flag = True
                                break
                        if add_flag is False:
                            return None
            self.rel_pairs = rel_pairs
        else:
            self.rel_pairs = scene_state["causal_pairs"]
        print("[func relation pairs]: ", self.rel_pairs)

        # Generate scene state
        if scene_state is None:
            for i, body_id in enumerate(self.body_ids):
                if self.obj_ids[i] in ['100849', '100965', '100970', '102812', '102860']:
                    is_large_cause = 1
                else:
                    is_large_cause = 0
                for joint_id in range(self.bc.getNumJoints(body_id)):
                    joint_info = self.bc.getJointInfo(body_id, joint_id)
                    if not joint_info[12].decode('gb2312') in movable_links[i]:
                        continue
                    self.joint_states[(body_id, joint_id)] = {
                        'min_val': joint_info[8],
                        'max_val': joint_info[9],
                        'is_large_cause': is_large_cause
                    }

            self.scene_images = {
                'init_val': None,
                'cur_val': None
            }

        else:
            self.joint_states = scene_state['joint_states']
            self.scene_images = scene_state['scene_images']

        # random initial state
        for body_id, joint_id in self.joint_states.keys():
            min_val = self.joint_states[(body_id, joint_id)]['min_val']
            max_val = self.joint_states[(body_id, joint_id)]['max_val']
            if self.joint_states[(body_id, joint_id)]['is_large_cause']:
                if self.goal_conditioned: # for goal_conditioned manipulation, intialize multi-stage trigger to middle
                    val = (min_val + max_val) / 2
                else:
                    val = np.random.choice([min_val, max_val, (min_val + max_val) / 2])
            else:
                val = np.random.choice([min_val, max_val])

            self.joint_states[(body_id, joint_id)]['init_val'] = val
            self.bc.resetJointState(body_id, joint_id, val)
            self.joint_states[(body_id, joint_id)]['cur_val'] = val

            # add functional relations for initial states
            affected_pairs = []
            for rel_pair in self.rel_pairs:
                if rel_pair[1][2] == body_id and rel_pair[1][3] == joint_id:
                    affected_pairs.append(rel_pair)
            self.add_relations(affected_pairs, body_id, joint_id)
            
        self.scene_images['init_val'] = self.get_observation()['image']
        self.scene_images['cur_val'] = self.get_observation()['image']

        return self.get_observation()

    def add_relations(self, affected_pairs, body_id, joint_id):
        joint_state = self.bc.getJointState(body_id, joint_id)[0]

        joint_min_limit = self.bc.getJointInfo(body_id, joint_id)[8]
        joint_max_limit = self.bc.getJointInfo(body_id, joint_id)[9]

        state = np.clip((joint_state - joint_min_limit) /
                        (joint_max_limit - joint_min_limit), 0, 1)
        # print("[state] ", state, "[original state] ", original_state)

        for affected_pair in affected_pairs:
            effect_type, effect_id, effect_body_id, effect_joint_id = affected_pair[2]
            effect_path = os.path.join("../assets/objects", effect_type, effect_id, 'object_meta_info.json')
            effect_json = json.load(open(effect_path))["Effect"][effect_joint_id-1]

            if affected_pair[0] == "discrete":
                if state < 0.5:
                    if effect_json["Type"] == "link":
                        self.bc.changeVisualShape(effect_body_id, effect_json["LinkId"],
                                                  rgbaColor=effect_json["States"][0])
                    if effect_json["Type"] == "joint":
                        effect_joint_info = self.bc.getJointInfo(effect_body_id, effect_json["JointId"])
                        self.bc.resetJointState(effect_body_id, effect_json["JointId"],
                                                targetValue=effect_joint_info[8])
                else:
                    if effect_json["Type"] == "link":
                        self.bc.changeVisualShape(effect_body_id, effect_json["LinkId"],
                                                  rgbaColor=effect_json["States"][1])
                    if effect_json["Type"] == "joint":
                        effect_joint_info = self.bc.getJointInfo(effect_body_id, effect_json["JointId"])
                        self.bc.resetJointState(effect_body_id, effect_json["JointId"],
                                                targetValue=effect_joint_info[9])

            if affected_pair[0] == "continuous":
                if state < 0.25:
                    step_idx = 1
                elif state > 0.75:
                    step_idx = 2
                else:
                    step_idx = 0

                if effect_json["Type"] == "link":
                    selected_color = np.array(effect_json["Steps"][step_idx])
                    self.bc.changeVisualShape(effect_body_id, effect_json["LinkId"], rgbaColor=selected_color)
                if effect_json["Type"] == "joint":
                    selected_pos = np.array(effect_json["Steps"][step_idx])
                    self.bc.resetJointState(effect_body_id, effect_json["JointId"][0], targetValue=selected_pos[0])
                    self.bc.resetJointState(effect_body_id, effect_json["JointId"][1], targetValue=selected_pos[1])

        self.bc.stepSimulation()

    def get_states(self):
        states = np.zeros((len(self.body_ids), 1))
        for body_id, joint_id in self.joint_states:
            joint_state = self.bc.getJointState(body_id, joint_id)[0]
            states[body_id - 1] = np.clip((joint_state - self.joint_states[(body_id, joint_id)]['min_val']) /
                                            (self.joint_states[(body_id, joint_id)]['max_val'] -
                                            self.joint_states[(body_id, joint_id)]['min_val']), 0, 1)

            affected_pairs = []
            for rel_pair in self.rel_pairs:
                if rel_pair[1][2] == body_id and rel_pair[1][3] == joint_id:
                    affected_pairs.append(rel_pair)

            for affect_pair in affected_pairs:
                _, _, effect_body_id, _ = affect_pair[2]
                if affect_pair[0] == "discrete":
                    if states[body_id - 1] < 0.5:
                        states[effect_body_id - 1] = 0
                    else:
                        states[effect_body_id - 1] = 1
                if affect_pair[0] == "continuous":
                    if states[body_id - 1] < 0.25:
                        states[effect_body_id - 1] = 2
                    elif states[body_id - 1] > 0.75:
                        states[effect_body_id - 1] = 3
                    else:
                        states[effect_body_id - 1] = 0
        return states

    def step(self, action, **kwargs):
        """Execute action and return reward and next observation.

        Args:
            action: [action_type=0, w, h] / [action_type=1, x, y, z]
                Case 0: Choose action position.
                Case 1: Choose direction.
        Returns:
            observation
            reward, move_flag
            terminate
            info
        """

        action_type = action[0]
        if action_type == 0:
            self.position = action[1:]
            pixel_index = np.ravel_multi_index(action[1:], self.depth_image.shape)

            body_id = int(self.body_id_pts[pixel_index])
            link_id = int(self.link_id_pts[pixel_index])

            observation = self.get_observation()

            # wrong position, terminate immediately
            if (body_id, link_id) not in self.joint_states:
                return observation, (0, False), True, dict()

            self.original_joint_state = self.bc.getJointState(body_id, link_id)[0]

            position_start = self.xyz_pts[pixel_index]
            self._suction_gripper = self.bc.loadURDF(
                self.ee_urdf,
                useFixedBase=True,
                basePosition=self._mount_base_position,
                globalScaling=1,
                flags=self.bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
            )
            link_state = self.bc.getLinkState(body_id, link_id)
            vec_inv, quat_inv = self.bc.invertTransform(link_state[0], link_state[1])
            self.transform_start_to_link = self.bc.multiplyTransforms(vec_inv, quat_inv, position_start, [0, 0, 0, 1])
            self.body_id, self.link_id = body_id, link_id

            self._move_to(self._mount_base_position, speed_ratio=10)
            self._move_to(position_start, speed_ratio=1)

            constraint_id = self.bc.createConstraint(
                parentBodyUniqueId=body_id,
                parentLinkIndex=link_id,
                childBodyUniqueId=self._suction_gripper,
                childLinkIndex=2,
                jointType=self.bc.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=self.transform_start_to_link[0],
                parentFrameOrientation=self.transform_start_to_link[1],
                childFramePosition=[0, 0, 0],
            )
            self.bc.changeConstraint(constraint_id, maxForce=self._constraint_force)

            info = {
                'position_start': position_start,
                'body_id': body_id,
                'link_id': link_id
            }
            
            return observation, (1, False), False, info

        elif action_type == 1:
            obj_json_path = os.path.join("../assets/objects", self.obj_types[self.body_id-1], 
                                        self.obj_ids[self.body_id-1], 'object_meta_info.json')
            obj_json = json.load(open(obj_json_path))
            gt_directions = obj_json['Direction']
            
            if self.goal_conditioned:
                if self.joint_states[(self.body_id, self.link_id)]['is_large_cause']:
                    direction_candidates = np.array([action[1:4]])
                else:
                    direction_candidates = np.array([action[1:4], -np.array(action[1:4])])
            else:
                # use gt direction
                if np.array_equal(action[1:4], [0, 0, 0]):
                    direction_candidates = np.array(gt_directions)
                else:
                    direction_candidates = [np.array(action[1:4]), -np.array(action[1:4])]
                # introduce randomness
                np.random.shuffle(direction_candidates)
            action_distances = np.arange(0.03, 0.22, 0.03)
            try:
                for action_distance in action_distances:
                    for idx, direction_vec in enumerate(direction_candidates):
                        position_start = self.get_suction_target_position()
                        position_end = position_start + direction_vec * action_distance

                        self._move_to(position_end, speed_ratio=5)
                        self.bc.removeBody(self._suction_gripper)

                        affected_pairs = []
                        for rel_pair in self.rel_pairs:
                            if rel_pair[1][2] == self.body_id and rel_pair[1][3] == self.link_id:
                                affected_pairs.append(rel_pair)

                        self.add_relations(affected_pairs, self.body_id, self.link_id)

                        if self.gui:
                            self.bc.removeAllUserDebugItems()

                        reward, move_flag = self.get_single_step_image_difference_reward()
                        # ==================Oracle (joint state supervision)====================
                        # reward, move_flag = self.get_single_step_joint_reward()
                        # ======================================================================

                        info = {
                            'position_start': position_start,
                            'direction_vec': direction_vec,
                            'position_end': position_end,
                        }

                        if move_flag:
                            direction_valid = False
                            vector_1 = direction_vec / np.linalg.norm(direction_vec)
                            for gt_dir in gt_directions:
                                vector_2 = gt_dir / np.linalg.norm(gt_dir)
                                angle = np.arccos(np.dot(vector_1, vector_2))
                                if angle < 0.52:
                                    direction_valid = True
                                    break
                            if not direction_valid:
                                print("===================Direction excluded!===============")
                                reward, move_flag = 0, False
                                self.bc.resetJointState(self.body_id, self.link_id, self.original_joint_state)
                                self.add_relations(affected_pairs, self.body_id, self.link_id)
                                self.scene_images['cur_val'] = self.get_observation()['image']
                                # ==================Oracle (joint state supervision)====================
                                # self.joint_states[(self.body_id, self.link_id)]['cur_val'] = self.original_joint_state
                                # ======================================================================
                            return self.get_observation(), (reward, move_flag), info, (self.body_id, self.link_id), direction_vec
                        else:
                            self.bc.resetJointState(self.body_id, self.link_id, self.original_joint_state)
                            self.scene_images['cur_val'] = self.get_observation()['image']
                            # ==================Oracle (joint state supervision)====================
                            # self.joint_states[(self.body_id, self.link_id)]['cur_val'] = self.original_joint_state
                            # ======================================================================
                            position = self.position
                            self.step([0, position[0], position[1]])

                self.bc.removeBody(self._suction_gripper)
                return self.get_observation(), (reward, move_flag), info, (self.body_id, self.link_id), direction_candidates[0]
            
            except Exception:
                info = {
                    'position_start': [0, 0, 0],
                    'direction_vec': [0, 0, 0],
                    'position_end': [0, 0, 0],
                }
                return self.get_observation(), (0, False), info, (self.body_id, self.link_id), direction_candidates[0]


if __name__ == '__main__':
    sim = PybulletSim(False, 0.1)
    observation = sim.reset(board_type='Interact', instance_type='train')
    print("[INFO] Successfully generated a busyboard environment")