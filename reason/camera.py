import math
import numpy as np
import utils

class Camera():
    def __init__(self, bc):
        self.bc = bc

        # RGB-D camera setup
        self._scene_cam_image_size = (720, 960)
        self._scene_cam_z_near = 0.01
        self._scene_cam_z_far = 10.0
        self._scene_cam_fov_w = 69.40
        self._scene_cam_focal_length = (float(self._scene_cam_image_size[1])/2)/np.tan((np.pi*self._scene_cam_fov_w/180)/2)
        self._scene_cam_fov_h = (math.atan((float(self._scene_cam_image_size[0])/2)/self._scene_cam_focal_length)*2/np.pi)*180
        self._scene_cam_projection_matrix = self.bc.computeProjectionMatrixFOV(
            fov=self._scene_cam_fov_h,
            aspect=float(self._scene_cam_image_size[1])/float(self._scene_cam_image_size[0]),
            nearVal=self._scene_cam_z_near, farVal=self._scene_cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._scene_cam_intrinsics = np.array([[self._scene_cam_focal_length, 0, float(self._scene_cam_image_size[1])/2],
                                             [0, self._scene_cam_focal_length, float(self._scene_cam_image_size[0])/2],
                                             [0, 0, 1]])
        
        # heightmap config
        self._heightmap_pix_size = 0.01
        self._view_bounds = np.array([[-1.28, 1.28], [-1.28, 1.28], [0, 1]])


    def get_scene_cam_data(self, cam_position, cam_lookat, cam_up_direction):
        cam_view_matrix = np.array(self.bc.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)).reshape(4, 4).T
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
        color_image = rgb_pixels[:,:,:3].astype(np.uint8) # remove alpha channel
        z_buffer = np.array(camera_data[3]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1]))
        depth_image = (2.0*self._scene_cam_z_near*self._scene_cam_z_far)/(self._scene_cam_z_far+self._scene_cam_z_near-(2.0*z_buffer-1.0)*(self._scene_cam_z_far-self._scene_cam_z_near))
        segmentation_mask = np.array(camera_data[4], np.int).reshape(self._scene_cam_image_size) # - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
        
        return color_image, depth_image, segmentation_mask, cam_pose_matrix, cam_view_matrix
    

    def get_heightmap(self, color_image, depth_image, segmentation_mask, cam_pose):
        """Get point cloud and RGB-D heightmap from RGB-D image
        Args:
            color_image: HxWx3 uint8 array of color image
            depth_image: HxW float array of depth values in meters aligned with color_img
            segmentation_mask: HxW int array of segmentation image
            cam_pose: 3x4 float array of camera pose matrix
        
        Returns:
            position_points: Nx3 float array of point cloud in world coordinate
            color_points: Nx3 uint8 array of point colors
            segmentation_points: Nx1 int array of segmentations
            depth_heightmap: WxH float array
            color_heightmap: WxHx3 uint8 array
            segmentation_heightmap: WxH int array
        """
        position_points, color_points, segmentation_points = utils.get_pointcloud(depth_image, color_image, segmentation_mask, self._scene_cam_intrinsics, cam_pose)
        color_heightmap, depth_heightmap, segmentation_heightmap = utils.get_heightmap(position_points, color_points, segmentation_points, self._view_bounds, self._heightmap_pix_size, self._view_bounds[2,0])
        return position_points, color_points, segmentation_points, depth_heightmap, color_heightmap, segmentation_heightmap