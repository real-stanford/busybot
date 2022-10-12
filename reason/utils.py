import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model = models.resnet50(pretrained=True).eval()

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def get_tableau_palette():
    """Get Tableau color palette (10 colors) https://www.tableau.com/.

    Returns:
        palette: 10x3 uint8 array of color values in range 0-255 (each row is a color)
    """
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink 
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


def get_heightmap(cam_pts, color_pts, segmentation_pts, view_bounds, heightmap_pix_sz, zero_level):
    """Get top-down (along z-axis) orthographic heightmap image from 3D pointcloud

    Args:
        cam_pts: Nx3 float array of 3D points in world coordinates
        color_pts: Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
        segmentation_pts: Nx1 int array of segmentation instance corresponding to cam_pts
        view_bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of heightmap in world coordinates
        heightmap_pix_sz: float value defining size of each pixel in meters (determines heightmap resolution)
        zero_level: float value defining z coordinate of zero level (i.e. bottom) of heightmap
    
    Returns:
        depth_heightmap: HxW float array of height values (from zero level) in meters
        color_heightmap: HxWx3 uint8 array of backprojected color values in range 0-255 aligned with depth_heightmap
        segmentation_heightmap: HxW int array of segmentation instance aligned with depth_heightmap
    """

    heightmap_size = np.round(((view_bounds[1,1]-view_bounds[1,0])/heightmap_pix_sz,
                               (view_bounds[0,1]-view_bounds[0,0])/heightmap_pix_sz)).astype(int)

    # Remove points outside workspace bounds
    heightmap_valid_ind = np.logical_and(np.logical_and(
                          np.logical_and(np.logical_and(cam_pts[:,0] >= view_bounds[0,0],
                                                        cam_pts[:,0] <  view_bounds[0,1]),
                                                        cam_pts[:,1] >= view_bounds[1,0]),
                                                        cam_pts[:,1] <  view_bounds[1,1]),
                                                        cam_pts[:,2] <  view_bounds[2,1])
    cam_pts = cam_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    segmentation_pts = segmentation_pts[heightmap_valid_ind]

    # Sort points by z value (works in tandem with array assignment to ensure heightmap uses points with highest z values)
    sort_z_ind = np.argsort(cam_pts[:,2])
    cam_pts = cam_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    segmentation_pts = segmentation_pts[sort_z_ind]

    # Backproject 3D pointcloud onto heightmap
    heightmap_pix_x = np.floor((cam_pts[:,0]-view_bounds[0,0])/heightmap_pix_sz).astype(int)
    heightmap_pix_y = np.floor((cam_pts[:,1]-view_bounds[1,0])/heightmap_pix_sz).astype(int)

    # Get height values from z values minus zero level
    depth_heightmap = np.zeros(heightmap_size)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = cam_pts[:,2]
    depth_heightmap = depth_heightmap-zero_level
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -zero_level] = 0

    # Map colors
    color_heightmap = np.zeros((heightmap_size[0],heightmap_size[1],3),dtype=np.uint8)
    for c in range(3):
        color_heightmap[heightmap_pix_y,heightmap_pix_x,c] = color_pts[:,c]
    
    # Map segmentations
    segmentation_heightmap = np.zeros((heightmap_size[0],heightmap_size[1]),dtype=np.int)
    segmentation_heightmap[heightmap_pix_y,heightmap_pix_x] = segmentation_pts[:, 0]

    return color_heightmap, depth_heightmap, segmentation_heightmap


def get_pointcloud(depth_img, color_img, segmentation_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.
    
    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        segmentation_img: HxW int array of segmentation image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
        
    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
        color_pts: Nx1 int array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)
    segmentation_pts = None if segmentation_img is None else segmentation_img.reshape(-1)

    return cam_pts, color_pts, segmentation_pts


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    """Project points to 2D.

    Args:
        pts: Nx3 float array of 3D points in world coordinates.
        camera_view_matrix: 4x4 float array. A wrd2cam transformation defining camera's totation and translation.
        camera_intrisic: 3x3 float array. [ [f,0,0],[0,f,0],[0,0,1] ]. f is focal length.

    Returns:
        coord_2d: Nx3 float array of 2D pixel. (w, h, d) the last one is depth
    """
    pts_c = transform_pointcloud(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_pointcloud(pts_c, rot_algix) # Nx3
    coord_2d = np.dot(camera_intrisic, pts_c.T) # 3xN
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[:, 2]
    coord_2d = np.array([coord_2d[1], coord_2d[0], coord_2d[2]])
    return coord_2d.T 

def imretype(im, dtype):
    """Image retype.
    
    Args:
        im: original image. dtype support: float, float16, float32, float64, uint8, uint16
        dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    Returns:
        image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    # assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im

def draw_action(image, position_start, position_end, cam_intrinsics, cam_view_matrix, thickness=3, tipLength=0.2, color=(186,176,172)):
    coord_3d = [position_start, position_end]
    coord_2d = project_pts_to_2d(np.array(coord_3d), cam_view_matrix, cam_intrinsics)
    p_start = (int(coord_2d[0, 1]), int(coord_2d[0, 0]))
    p_end = (int(coord_2d[1, 1]), int(coord_2d[1, 0]))

    image = cv2.arrowedLine(imretype(image, 'uint8'), p_start, p_end, color, thickness=thickness, tipLength=tipLength)
    return image

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def extract_central_feature(t_images, bbox):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        t_images = t_images.cuda()
        model.cuda()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.layer1.register_forward_hook(get_activation('out'))
    model(t_images)
    feature = F.interpolate(activation['out'], size=t_images.shape[-2:], mode='bilinear', align_corners=False).data.cpu().numpy()

    v = np.array(((bbox[0, :, 0] + bbox[0, :, 2]) // 2) * (224 / 480), dtype=np.int32)
    u = np.array(((bbox[0, :, 1] + bbox[0, :, 3]) // 2) * (224 / 640), dtype=np.int32)
    output = feature[:, :, v, u]

    output = np.swapaxes(output, 1, 2)
    return output
