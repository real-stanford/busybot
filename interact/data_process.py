import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm

import utils

source_path_root = '/local/crv/dataset/partnet-mobility-v0' # TODO: replace with your path to partnet-mobility dataset
target_path_root = '/proj/crv/zeyi/busybot/assets/objects' # TODO: replace with your desired path to store the object info

COLOR_CHOICES = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]
INVALID_SWITCH_IDS = ["100368", "100845", "100846", "100847", "100866",
                      "100870", "100871", "100888", "100900", "100902",
                      "100911", "100915", "100919", "100924", "100953",
                      "100935", "100948", "100954", "100957", "100966",
                      "100971", "100978", "102856", "102844", "102864",
                      "100974", "102839", "100885", "100980"]


def modify_urdf(source_path, target_path, category_name):
    # remove collision in URDF
    tree = ET.parse(os.path.join(source_path, 'mobility.urdf'))
    for link_node in tree.findall('link'):
        inertial_tag = ET.SubElement(link_node, 'inertial')
        mass_tag = ET.SubElement(inertial_tag, 'mass')
        mass_tag.set('value', str(10))
        inertia_tag = ET.SubElement(inertial_tag, 'inertia')
        inertia_tag.set('ixx', str(1))
        inertia_tag.set('ixy', str(0))
        inertia_tag.set('ixz', str(0))
        inertia_tag.set('iyy', str(1))
        inertia_tag.set('iyz', str(0))
        inertia_tag.set('izz', str(1))
    if category_name in ['Kettle', 'KitchenPot']:
        for joint_node in tree.findall('joint'):
            if joint_node.attrib['type'] == 'prismatic':
                limit_node = joint_node.find('limit')
                lower_val = float(limit_node.attrib['lower'])
                upper_val = lower_val + 0.3
                limit_node.set('upper', str(upper_val))

    for joint_node in tree.findall('joint'):
        for limit_node in joint_node.findall('limit'):
            limit_node.set('effort', '30')
            limit_node.set('velocity', '1.0')

    tree.write(os.path.join(target_path, 'mobility.urdf'))


def copy_files(instance_id, category_name, moveable_link):
    source_path = os.path.join(source_path_root, instance_id)
    target_path = os.path.join(target_path_root, category_name, instance_id)
    utils.mkdir(target_path, clean=True)
    for file_name in ['bounding_box.json']:
        shutil.copy(os.path.join(source_path, file_name), os.path.join(target_path, file_name))
    for dir_name in ['textured_objs', 'images']:
        shutil.copytree(os.path.join(source_path, dir_name), os.path.join(target_path, dir_name))
    modify_urdf(source_path, target_path, category_name)


def get_orientation(category_name, instance_id):
    if category_name in ['Switch', 'Door']:
        return [0, 0.7071068, 0, 0.7071068]
    if category_name in ['Lamp']:
        return [0, 0, 0, 1]
    else:
        print("Invalid category: ", category_name)
        assert False


# Return max bound and min bound
def adjust_bbox(bbox, new_orientation):
    max_bbox = bbox["max"]
    min_bbox = bbox["min"]
    if new_orientation == [0, 0, 0, 1]:
        max_bbox = [max_bbox[2]] + [max_bbox[0]] + [max_bbox[1]]
        min_bbox = [min_bbox[2]] + [min_bbox[0]] + [min_bbox[1]]
    if new_orientation == [0, 0.7071068, 0, 0.7071068]:
        max_bbox = [max_bbox[1]] + [max_bbox[0]] + [max_bbox[2]]
        min_bbox = [min_bbox[1]] + [min_bbox[0]] + [min_bbox[2]]
    return max_bbox, min_bbox


def get_scale(category_name, instance_id):
    if category_name in ['Switch']:
        # large objects
        if instance_id in ['100920', '100928', '100366', '100848', '100850', '100955', '100963']:
            return 0.15
        elif instance_id in ['102860', '100849', '100970', '102812', '100955']:
            return 0.2
        elif instance_id in ['100965']:
            return 0.3
        else:  # small objects
            return 0.2
    elif category_name in ['Lamp']:
        if instance_id in ['14127']:
            return 0.3
        elif instance_id in ['13491', '14605']:
            return 0.4
        elif instance_id in ['16237']:
            return 0.2
        else:
            return 1.2
    else:
        return 0.4


def get_cause(category_name, instance_id, links):
    cause_list = []
    if category_name in ['Switch']:
        if len(links.keys()) > 1:
            for link_id in links.keys():
                cause_json = {
                    "JointId": int(link_id[-1]) + 1,
                    "IsSmallCause": 1,
                    "IsLargeCause": 0,
                }
                cause_list.append(cause_json)
        elif instance_id in ['100849', '100965', '100970', '102812', '102860']:
            cause_json = {
                "JointId": 1,
                "IsSmallCause": 0,
                "IsLargeCause": 1,
            }
            cause_list.append(cause_json)
        else:
            cause_json = {
                "JointId": 1,
                "IsSmallCause": 1,
                "IsLargeCause": 0,
            }
            cause_list.append(cause_json)
    return cause_list


def get_effect(category_name, instance_id):
    effect_list = []
    if category_name in ['Lamp']:
        color_indices = np.random.choice(len(COLOR_CHOICES), 3, replace=False)
        if instance_id in ['14605', '14563']:
            effect_json = {
                "Type": "link",
                "JointId": None,
                "LinkId": 3,
                "IsMultiEffect": 1,
                "Steps": [[0, 0, 0, 1], COLOR_CHOICES[color_indices[1]], COLOR_CHOICES[color_indices[2]]],
                "IsSingleEffect": 1,
                "States": [[0, 0, 0, 1], COLOR_CHOICES[color_indices[0]]]}
            effect_list.append(effect_json)
        else:
            effect_json = {
                "Type": "link",
                "JointId": None,
                "LinkId": 1,
                "IsMultiEffect": 1,
                "Steps": [[0, 0, 0, 1], COLOR_CHOICES[color_indices[1]], COLOR_CHOICES[color_indices[2]]],
                "IsSingleEffect": 1,
                "States": [[0, 0, 0, 1], COLOR_CHOICES[color_indices[0]]]}
            effect_list.append(effect_json)

    if category_name in ['Door']:
        effect_json = {
            "Type": "joint",
            "JointId": 1,
            "LinkId": None,
            "IsMultiEffect": 0,
            "IsSingleEffect": 1,
        }
        effect_list.append(effect_json)
    return effect_list


def check_link(category_name, instance_id, link_name):
    if category_name in ['Switch']:
        if link_name in ['slider', 'lever', 'toggle_button', 'button', 'knob']:
            return True
    else:
        return False


def check_valid_instance(category_name, instance_id, link_info_list):
    cnt = 0
    for link_id, joint_type, link_name in link_info_list:
        if joint_type in ['hinge', 'slider']:
            cnt += 1
    if cnt >= 2:
        if category_name in ['FoldingChair']:
            return False
    return True


def get_gt_direction(category_name, instance_id):
    if category_name in ['Switch']:
        if instance_id in ['102860', '102812', '100970', '100955', '100965', '100849', '100920']:
            return [[1, 0, 0], [-1, 0, 0]]
        else:
            return [[-1, 0, 1], [1, 0, -1], [-1, 0, -1], [1, 0, 1], [0, 0, 1], [0, 0, -1]]


def collect_data(category_name):
    target_path = os.path.join(target_path_root, category_name)
    utils.mkdir(target_path, clean=True)

    all_instance_ids = os.listdir(source_path_root)

    for instance_id in all_instance_ids:
        if category_name == 'Switch' and instance_id in INVALID_SWITCH_IDS:
            continue
        if category_name == 'Lamp' and instance_id not in ["13491", "14127", "14205", "14605", "16237"]:
            continue
        if category_name == 'Door' and instance_id not in ["8867", "8893", "8897", "8903", "8983",
                                                           "8994", "9065", "9117", "9277", "9280"]:
            continue
        with open(os.path.join(source_path_root, instance_id, 'meta.json')) as f:
            meta_data = json.load(f)
            if meta_data['model_cat'] == category_name:
                # print('==>')
                # print(instance_id)
                with open(os.path.join(source_path_root, instance_id, 'bounding_box.json'), 'r') as f:
                    bbox = json.load(f)

                orientation = get_orientation(category_name, instance_id)
                scale = get_scale(category_name, instance_id)
                max_bbox, min_bbox = adjust_bbox(bbox, orientation)
                object_meta_info = {
                    'Category': category_name,
                    'InstanceId': instance_id,
                    'Scale': get_scale(category_name, instance_id),
                    'Movable_link': dict(),
                    'Orientation': orientation,
                    "MaxBBox": max_bbox,
                    "MinBBox": min_bbox,
                    "Offset_z": -min_bbox[2] * scale,
                    "Direction": get_gt_direction(category_name, instance_id),
                    "Effect": get_effect(category_name, instance_id),
                }

                link_info_list = list()
                for link_info in open(os.path.join(source_path_root, instance_id, 'semantics.txt')).readlines():
                    link_id, joint_type, link_name = link_info.split()
                    link_info_list.append((link_id, joint_type, link_name))
                if not check_valid_instance(category_name, instance_id, link_info_list):
                    print("[invalid instance id]", instance_id)
                    continue
                for link_id, joint_type, link_name in link_info_list:
                    if check_link(category_name, instance_id, link_name):
                        object_meta_info['Movable_link'][link_id] = {
                            'link_id': link_id,
                            'link_name': link_name,
                            'joint_type': joint_type
                        }
                        assert joint_type in ['hinge', 'slider']
                object_meta_info["Cause"] = get_cause(category_name, instance_id, object_meta_info['Movable_link'])
                copy_files(instance_id, category_name, object_meta_info['Movable_link'])
                with open(os.path.join(target_path, instance_id, 'object_meta_info.json'), 'w') as outfile:
                    json.dump(object_meta_info, outfile)


def visualize_data(category_name):
    target_path = os.path.join(target_path_root, category_name)
    all_instance_ids = [x for x in os.listdir(target_path) if x.isnumeric()]

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    # RGB-D camera setup
    scene_cam_image_size = (480, 640)
    scene_cam_z_near = 0.01
    scene_cam_z_far = 10.0
    scene_cam_fov_w = 69.40
    scene_cam_focal_length = (float(scene_cam_image_size[1]) / 2) / np.tan((np.pi * scene_cam_fov_w / 180) / 2)
    scene_cam_fov_h = (math.atan((float(scene_cam_image_size[0]) / 2) / scene_cam_focal_length) * 2 / np.pi) * 180
    scene_cam_projection_matrix = p.computeProjectionMatrixFOV(
        fov=scene_cam_fov_h,
        aspect=float(scene_cam_image_size[1]) / float(scene_cam_image_size[0]),
        nearVal=scene_cam_z_near, farVal=scene_cam_z_far
    )  # notes: 1) FOV is vertical FOV 2) aspect must be float

    camera_look_at = np.array([0, 0, 0])
    camera_up_direction = np.array([0, 0, 1])
    camera_position_list = [np.array([0, -2, 2]), [2, 0, 2], [-2, 0, 0.001]]

    visualization_data = dict()
    rows = list()
    cols = list()
    for i in range(len(camera_position_list)):
        cols.append(f'{i}-color')
        cols.append(f'{i}-seg')

    invalid_ids = list()
    for instance_id in tqdm(all_instance_ids):
        print('==> instance_id = ', instance_id)
        urdf_path = os.path.join(target_path, instance_id, 'mobility.urdf')
        with open(os.path.join(target_path, instance_id, 'object_meta_info.json'), 'r') as f:
            meta_info = json.load(f)
        try:
            base_orientation = \
                p.multiplyTransforms([0, 0, 0],
                                     p.getQuaternionFromEuler([0, 0, np.random.rand() * np.pi / 2 - np.pi / 4]),
                                     [0, 0, 0], meta_info['Orientation'])[1]
            obj_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, meta_info['Offset_z']],
                baseOrientation=base_orientation,
                globalScaling=meta_info['Scale'],
                useFixedBase=True,
                flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            )
        except:
            invalid_ids.append(instance_id)
            continue

        for _ in range(40):
            p.stepSimulation()
        images = list()

        link_name_list = meta_info['Movable_link'].keys()
        moveable_link_id_list = list()
        num_joints = p.getNumJoints(obj_id)

        flag = True
        for i in range(num_joints):
            joint_info = p.getJointInfo(obj_id, i)
            link_name = joint_info[12].decode('gb2312')
            if link_name in link_name_list:
                moveable_link_id_list.append(i)
                if joint_info[9] - joint_info[8] > 3.2 or joint_info[9] < joint_info[8]:
                    print(joint_info[9] - joint_info[8])
                    flag = False
        if not flag:
            invalid_ids.append(instance_id)
            p.removeBody(obj_id)
            continue

        # print('==> ', category_name, instance_id)
        rows.append(instance_id)

        for camera_position in camera_position_list:
            camera_view_matrix = np.array(
                p.computeViewMatrix(camera_position, camera_look_at, camera_up_direction)).reshape(4, 4).T
            camera_pose_matrix = np.linalg.inv(camera_view_matrix)
            camera_pose_matrix[:, 1:3] = -camera_pose_matrix[:, 1:3]
            camera_data = p.getCameraImage(
                scene_cam_image_size[1],
                scene_cam_image_size[0],
                p.computeViewMatrix(camera_position, camera_look_at, camera_up_direction),
                scene_cam_projection_matrix,
                shadow=1,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_pixels = np.array(camera_data[2]).reshape((scene_cam_image_size[0], scene_cam_image_size[1], 4))
            color_image = rgb_pixels[:, :, :3].astype(np.uint8)  # remove alpha channel
            z_buffer = np.array(camera_data[3]).reshape((scene_cam_image_size[0], scene_cam_image_size[1]))
            segmentation_mask = np.array(camera_data[4], int).reshape(
                scene_cam_image_size)  # - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
            link_id_pts = (segmentation_mask >> 24) - 1
            seg_image = np.zeros([scene_cam_image_size[0], scene_cam_image_size[1]], dtype=np.float32)
            for moveable_link_id in moveable_link_id_list:
                seg_image[moveable_link_id == link_id_pts] = 1
            images.append({
                'color_image': color_image,
                'seg_image': seg_image
            })
        p.removeBody(obj_id)

        for i in range(len(camera_position_list)):
            visualization_data[f'{instance_id}_{i}-color'] = images[i]['color_image']
            visualization_data[f'{instance_id}_{i}-seg'] = images[i]['seg_image']

    print('==>', category_name)
    print(invalid_ids)
    visualization_path = os.path.join('exp', 'data_visualization', category_name)
    utils.html_visualize(visualization_path, visualization_data, rows, cols, title=category_name)


def split_data(file_name='split-full.json'):
    np.random.seed(0)

    train_categories = [
        'Switch',
        'Lamp',
        'Door'
    ]
    test_categories = []
    split_meta = {'train': dict(), 'test': dict()}

    tot_train_num, tot_test_num = 0, 0
    for category_name in train_categories:
        target_path = os.path.join(target_path_root, category_name)
        all_instance_ids = [x for x in os.listdir(target_path) if x.isnumeric()]
        np.random.shuffle(all_instance_ids)
        # train_num = min(int(len(all_instance_ids) * 0.5), 5)
        train_num = int(len(all_instance_ids) * 0.8)
        test_num = len(all_instance_ids) - train_num
        print('==>', category_name, train_num, test_num)
        tot_train_num += train_num
        tot_test_num += test_num
        if category_name == 'Microwave':
            split_meta['train'][category_name] = {
                'train': all_instance_ids[test_num:],
                'test': all_instance_ids[:test_num]
            }
        else:
            split_meta['train'][category_name] = {
                'train': all_instance_ids[:train_num],
                'test': all_instance_ids[train_num:]
            }

    print('==> training categories:')
    print(tot_train_num + tot_test_num, '=', tot_train_num, '+', tot_test_num)

    tot_test_num = 0
    for category_name in test_categories:
        target_path = os.path.join(target_path_root, category_name)
        all_instance_ids = [x for x in os.listdir(target_path) if x.isnumeric()]
        tot_test_num += len(all_instance_ids)
        print('==>', category_name, len(all_instance_ids))
        split_meta['test'][category_name] = {
            'train': [],
            'test': all_instance_ids
        }
    print('==> testing categories:')

    with open(os.path.join(target_path_root, file_name), 'w') as f:
        json.dump(split_meta, f)


if __name__ == '__main__':
    collect_data("Switch")
    visualize_data("Switch")
    collect_data("Lamp")
    visualize_data("Lamp")
    collect_data("Door")
    visualize_data("Door")
    
    # split_data()
