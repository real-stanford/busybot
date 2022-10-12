import os
import random


def load_obj(bc, position, orientation, object_type, object_id, scale=1, **kwargs):
    """Load object in pybullet env.

    Args:
        bc: bullet client.
        position: position array (xyz)
        orientation: orientation array (quaternion)

        object_type, object_id, scale **kwargs:

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """

    # TODO: replace all urdf_path with yours
    if object_type == 'Switch':
        urdf_path = os.path.join('/proj/crv/zeyi/busybot/assets/objects', object_type, object_id, 'mobility.urdf')

        obj_id = bc.loadURDF(
            fileName=urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
            useFixedBase=True,
            flags=bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )

    elif object_type == 'Door':
        urdf_path = os.path.join('/proj/crv/zeyi/busybot/assets/objects', object_type, object_id, 'mobility.urdf')

        obj_id = bc.loadURDF(
            fileName=urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
            useFixedBase=True,
            flags=bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )

        # Use random door colors to introduce variance
        # random_color = [random.random(), random.random(), random.random(), 1]
        # bc.changeVisualShape(obj_id, 1, rgbaColor=random_color)

    elif object_type == 'Lamp':
        urdf_path = os.path.join('/proj/crv/zeyi/busybot/assets/objects', object_type, object_id, 'mobility.urdf')

        obj_id = bc.loadURDF(
            fileName=urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
            useFixedBase=True,
            flags=bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )

        if object_id == '14605':
            bc.changeVisualShape(obj_id, 3, rgbaColor=[0, 0, 0, 1])
        else:
            bc.changeVisualShape(obj_id, 1, rgbaColor=[0, 0, 0, 1])

    elif object_type == 'Toy':
        urdf_path = os.path.join('/proj/crv/zeyi/busybot/assets/objects', object_type, object_id, 'mobility.urdf')

        obj_id = bc.loadURDF(
            fileName=urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
            useFixedBase=True,
            flags=bc.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )

        bc.resetJointState(obj_id, 1, 0)
        bc.resetJointState(obj_id, 2, 0)

    else:
        raise NotImplementedError(f'does not support {object_type}')

    # step simulation to update joints
    bc.stepSimulation()

    return obj_id
