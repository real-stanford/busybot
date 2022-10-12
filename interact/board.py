import json
import os
import numpy as np
from obj import load_obj
import transformations
from shapely.geometry import Polygon, Point
from shapely.affinity import *
import random
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

class Board(object):
    def __init__(self, bc, board_length, board_width, texture_file, board_color):
        self.bc = bc
        self.object_dataset = json.load(open(os.path.join("/proj/crv/zeyi/busybot/assets/objects", 'data.json')))
        board_pos = [0, 0, -0.01 * (board_length-1)]
        self.board_id = self.bc.loadURDF("/proj/crv/zeyi/busybot/assets/board/table.urdf",
                                          basePosition=board_pos,
                                          baseOrientation=transformations.quaternion_from_euler(0, 0, 0),
                                          useFixedBase=True,
                                          globalScaling=board_length)

        texUid = self.bc.loadTexture(os.path.join('/proj/crv/zeyi/busybot/assets/board', texture_file))
        self.bc.changeVisualShape(self.board_id, -1, textureUniqueId=texUid, rgbaColor=board_color)

        self.board_length = board_length
        self.board_width = board_width
        self.board_pos = board_pos
        self.texture_file = texture_file
        self.board_color = board_color
        self.initial_board_boundary = np.array([
            [board_pos[0] - board_width/2 + 0.25, board_pos[0] + board_width/2 - 0.25],  # 2x2 rows: x, y cols:min, max
            [board_pos[1] - board_length/2 + 0.25, board_pos[1] + board_length/2 - 0.25]])

        self.initial_board_boundary_polygon = Polygon([
            (self.initial_board_boundary[0, 0], self.initial_board_boundary[1, 0]),
            (self.initial_board_boundary[0, 1], self.initial_board_boundary[1, 0]),
            (self.initial_board_boundary[0, 1], self.initial_board_boundary[1, 1]),
            (self.initial_board_boundary[0, 0], self.initial_board_boundary[1, 1])])
        self.body_ids = []

    def sample_from_category(self, categories, object_num, board_type, instance_type):
        """
        Samples given number of objects from a specific category

        :param categories (list): object categories, e.g. switches, door, ...
        :param object_num (list) : number of objects to sample from each category
        :param category_type : train/test
        :param instance_type : train/test
        :return: object_ids (list): distinctive object ids
        """
        if len(categories) != len(object_num):
            print("Number of categories and sample_num do not match")
            return None
        # read json file for names of objects from the same category
        object_ids = []
        for idx, category in enumerate(categories):
            if object_num[idx] != 0:
                objects = self.object_dataset[board_type][category][instance_type]
                object_ids.extend(np.random.choice(objects, object_num[idx], replace=True))

        # print(object_ids)
        return object_ids

    def reset(self, categories, object_num, board_type, instance_type, board_rot_angle=0):
        """
        Generates board with given number of objects
        - The objects are guaranteed to be non-overlap

        :param categories (list): object categories, e.g. switches, door, ...
        :param object_num (list) : number of objects to sample from each category
        :param board_type (string): Interact/Reason/Plan
        :param instance_type (string): train/test
        :param board_rot_angle (float): The orientation of the board (global orientation)
        :return:
            movable_joints: list of (body_id, joint_id)
            movable_links: list of (body_id, link_id)
        """

        # remove existed objects
        for body_id in reversed(self.body_ids):
            self.bc.removeBody(body_id)
        self.body_ids = []
        self.occupied_area = []
        self.movable_links = []
        is_multi_trigger = []
        positions = []
        orientations = []
        scales = []
        self.bc.removeAllUserDebugItems()

        # visualize the board boundary
        self.board_boundary = rotate(self.initial_board_boundary_polygon, board_rot_angle, use_radians=True)
        self.visualize_bounding_box(self.board_boundary, color=[0, 0, 1], radius=10.0)

        # start to load new objects
        object_ids = self.sample_from_category(categories=categories, object_num=object_num, 
                                                board_type=board_type, instance_type=instance_type)
        object_types = []
        for i in range(len(categories)):
            object_types += [categories[i]] * object_num[i]

        # Permute the index of objects
        temp = list(zip(object_ids, object_types))
        random.shuffle(temp)
        object_ids, object_types = zip(*temp)

        for i, object_id in enumerate(object_ids):
            obj_json_path = os.path.join("/proj/crv/zeyi/busybot/assets/objects", object_types[i], object_id, 'object_meta_info.json')
            obj_json = json.load(open(obj_json_path))
            obj_ori = self.get_obj_orientation(obj_json)
            euler = transformations.euler_from_quaternion(obj_ori)

            # Switch and Toy have fixed rotation, Lamp and Door can rotate
            if object_types[i] in ['Lamp', 'Door']:
                z_angle = np.random.choice([0, np.pi/2, np.pi, 3 * np.pi/2])
            else:
                z_angle = 0
            orientation = transformations.quaternion_from_euler(euler[0]+z_angle, euler[1], euler[2])

            global_scale = self.get_scale(obj_json)

            object_pos = self.find_position(obj_json, z_angle, global_scale)
            # If no valid position is found, object_pos is None
            if object_pos is None:
                # print("Fail to find position for the current object, skip to the next object...")
                break

            offset_z = self.get_offset_z(obj_json)

            body_id = load_obj(
                bc=self.bc,
                object_type=object_types[i],
                object_id=object_id, scale=global_scale,
                position=[object_pos[0], object_pos[1], offset_z],
                orientation=orientation
            )

            self.body_ids.append(body_id)
            self.movable_links.append(obj_json["Movable_link"])
            positions.append([object_pos[0], object_pos[1], offset_z])
            orientations.append(orientation)
            scales.append(global_scale)

            if len(obj_json["Cause"]) != 0 and obj_json["Cause"][0]["IsLargeCause"]:
                is_multi_trigger.append(1)
            else:
                is_multi_trigger.append(0)

        self.bc.stepSimulation()
        # assert(not self.sanity_check_overlap(self.body_ids))

        return object_ids, object_types, self.body_ids, positions, orientations, scales, is_multi_trigger, self.movable_links


    def reload(self, scene_state):
        # reproduce objects
        self.bc.removeAllUserDebugItems()

        for i, object_id in enumerate(scene_state['obj_ids']):
            load_obj(
                bc=self.bc,
                object_type=scene_state['obj_types'][i],
                object_id=object_id, scale=scene_state['scales'][i],
                position=scene_state['base_positions'][i],
                orientation=scene_state['orientations'][i]
            )

    def get_bounding_box_in_board_coordinates(self, obj_json, object_pos, z_angle, global_scale):
        """
        Gets the bounding box of the given object in the board coordinates

        :param obj_json: json string of object information
        :param object_pos (Numpy array [1x2]): base position of the object [x, y]
        :param z_angle (float): The orientation of the board (global orientation)
        :param global_scale: scaling factor of the object when load into the environment
        :return: bounding box of the object (Polygon)
        """
        max_bounds = np.array(obj_json["MaxBBox"])
        max_bounds[0] += 0.2
        max_bounds[1] += 0.2
        min_bounds = np.array(obj_json["MinBBox"])
        min_bounds[0] -= 0.2
        min_bounds[1] -= 0.2
        bbox = Polygon([(min_bounds[0], min_bounds[1]), (max_bounds[0], min_bounds[1]),
                       (max_bounds[0], max_bounds[1]), (min_bounds[0], max_bounds[1])])
        bbox = scale(bbox, xfact=global_scale, yfact=global_scale)
        bbox = translate(bbox, xoff=object_pos[0], yoff=object_pos[1])
        bbox = rotate(bbox, z_angle, origin=object_pos, use_radians=True)

        return bbox

    def get_offset_z(self, obj_json):
        return obj_json["Offset_z"]

    def get_obj_orientation(self, obj_json):
        orientation = np.asarray(obj_json["Orientation"])
        return orientation

    def get_movable_links(self, obj_json):
        movable_links = np.asarray(obj_json["Movable_link"])
        return movable_links

    def get_scale(self, obj_json):
        return obj_json["Scale"]

    def find_position(self, obj_json, z_angle, global_scale, MAX_ITERATIONS=50):
        """
        Helper function to search for valid position to place the given object

        :param obj_json: json string of object information
        :param z_angle (float): The orientation of the board (global orientation)
        :param global_scale: scaling factor of the object when load into the environment
        :param MAX_ITERATIONS: max number of trials
        :return: x and y position of the object on the board;
                 if no position is found, return None
        """
        # Generate random positions for objects
        offset_x, offset_y, bbox = 0, 0, None
        found = False
        for i in range(MAX_ITERATIONS):
            offset_x = np.random.uniform(self.initial_board_boundary[0, 0], self.initial_board_boundary[0, 1], 1)[0]
            offset_y = np.random.uniform(self.initial_board_boundary[1, 0], self.initial_board_boundary[1, 1], 1)[0]
            rotated_point = rotate(Point(offset_x, offset_y), z_angle, origin=self.board_pos, use_radians=True)
            object_pos = list(rotated_point.coords)[0]

            bbox = self.get_bounding_box_in_board_coordinates(obj_json, object_pos, z_angle, global_scale)

            if not self.check_overlap(bbox) and not self.check_off_board(bbox):
                found = True
                break

        if found:
            self.occupied_area.append(bbox)
            self.visualize_bounding_box(bbox)
            return object_pos
        if not found:
            return None

    def check_overlap(self, bbox):
        """
        Checks if the placement of a new object overlaps with existing objects

        :param bbox (Polygon): bounding box of an object
        :return: True if there is overlap; false if there isn't
        """
        for area in self.occupied_area:
            if bbox.intersects(area):
                return True
        return False

    def check_off_board(self, bbox):
        """
        Checks if the object exceeds the bound of the board

        :param bbox (Polygon): bounding box of an object
        :return: True if object is out of bound; false if object is within the bound
        """
        if self.board_boundary.contains(bbox):
            return False
        else:
            return True

    def sanity_check_overlap(self, obj_info):
        """
        A sanity check for overlap during simulation.

        :param obj_info: bodyIds of all objects in the pybullet simulation environment
        :return: True if there is overlap; false if there isn't
        """
        for i in range(len(obj_info)):
            for j in range(i+1, len(obj_info)):
                object_A, object_B = obj_info[2], obj_info[2]
                overlap_points = self.bc.getContactPoints(object_A, object_B)
                if overlap_points is not None and len(overlap_points) != 0:
                    return True
        return False

    def visualize_bounding_box(self, bbox, color=[0, 1, 0], radius=3.0):
        coords = list(bbox.exterior.coords)
        for i in range(4):
            self.bc.addUserDebugLine(coords[i]+(0,), coords[i+1]+(0,), color, radius)
