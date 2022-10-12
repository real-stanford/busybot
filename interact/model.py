import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch import nn
from scipy import signal
import matplotlib.pyplot as plt

from module_utils import MLP
from unet_parts import *


def gkern(H, W, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_h = signal.gaussian(H, std=std).reshape(H, 1)
    gkern1d_w = signal.gaussian(W, std=std).reshape(W, 1)
    gkern2d = np.outer(gkern1d_h, gkern1d_w)
    return gkern2d


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Conv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DirModel(nn.Module):
    def __init__(self, num_directions, model_type):
        super().__init__()
        self.num_directions = num_directions
        self.model_type = model_type

        candidate_directions_positive = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0],
                                                  [0, 0, 1], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]])
        self.raw_directions = np.concatenate(
            (candidate_directions_positive, -candidate_directions_positive))

        image_feature_dim = 256
        action_feature_dim = 0
        output_dim = self.raw_directions.shape[0]

        if 'mag' in model_type:
            num_channels = 5 if model_type == 'mag' else 5
            self.mag_image_encoder_1 = Conv(num_channels, 32)
            self.mag_image_encoder_2 = Down(32, 64)
            self.mag_image_encoder_3 = Down(64, 128)
            self.mag_image_encoder_4 = Down(128, 256)
            self.mag_image_encoder_5 = Down(256, 512)
            self.mag_image_encoder_6 = Down(512, 512)
            self.mag_image_encoder_7 = Down(512, 512)
            self.mag_image_feature_extractor = MLP(
                512*7*10, image_feature_dim, [image_feature_dim])
            self.mag_decoder = MLP(
                image_feature_dim + action_feature_dim, output_dim, [1024, 1024, 1024])

        # Initialize random weights
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Conv3d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm3d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, observation, directions=None):
        if 'mag' in self.model_type:
            x0 = observation
            x1 = self.mag_image_encoder_1(x0)
            x2 = self.mag_image_encoder_2(x1)
            x3 = self.mag_image_encoder_3(x2)
            x4 = self.mag_image_encoder_4(x3)
            x5 = self.mag_image_encoder_5(x4)
            x6 = self.mag_image_encoder_6(x5)
            x7 = self.mag_image_encoder_7(x6)
            embedding = x7.reshape([x7.size(0), -1])
            mag_feature = self.mag_image_feature_extractor(embedding)
        batch_size = observation.size(0)
        
        assert directions is None or len(directions.shape) == 2
        if directions is not None:
            diff = np.sum(
                (self.raw_directions - directions[:, np.newaxis]) ** 2, axis=2)
            max_id = np.argmin(diff, axis=1)

        mag_output = None
        if 'mag' in self.model_type:
            mag_output = self.mag_decoder(mag_feature)
            if directions is not None:
                mag_output = mag_output[np.arange(batch_size), max_id].unsqueeze(1)

        directions = [self.raw_directions] * batch_size
        output = mag_output, directions
        return output


class Model():
    def __init__(self, num_directions, model_type):
        self.num_directions = num_directions
        self.model_type = model_type

        self.pos_model = UNet(4, 2)
        self.dir_model = DirModel(num_directions, model_type)

    def get_direction_affordance(self, observations, positions, model_type, torch_tensor=False, directions=None):
        """Get position affordance maps.

        Args:
            observations: list of dict
                - image: [H, W, 10]. dtype: float32
                - image_init: [H, W, 10]. dtype: float32
            positions: list of selected pixel position
            model_type: 'sgn', 'mag', 'sgn_mag'
            torch_tensor: Whether the retuen value is torch tensor (default is numpy array). torch tensor is used for training.
        Return:
            affordance_maps: numpy array/torch tensor, [B, K, H, W]
            directions: list of direction vector
        """
        skip_id_list = list()
        scene_inputs = []
        position_masks = list()
        for id, observation in enumerate(observations):
            if observation is None:
                skip_id_list.append(id)
                continue
            h, w = positions[id]
            H, W = observation['image'].shape[0], observation['image'].shape[1]
            position_mask = np.zeros((H, W, 1), dtype=np.float32)
            r1, r2 = max(0, h-40), min(h+40, H)
            c1, c2 = max(0, w-40), min(w+40, W)
            position_mask[r1:r2, c1:c2, :] = gkern(
                r2-r1, c2-c1, 10).reshape((r2-r1, c2-c1, 1))
            scene_inputs.append(np.concatenate([observation['image'][:, :, 3:].transpose(
                [2, 0, 1]), position_mask.transpose([2, 0, 1])], axis=0))
            position_masks.append(position_mask)

        scene_input_tensor = torch.from_numpy(np.stack(scene_inputs))
        mag_output, skipped_directions = self.dir_model.forward(
            scene_input_tensor.to(self.device_dir), directions=directions)  # [B, K, W, H]
        if torch_tensor:
            assert len(skip_id_list) == 0
            return mag_output, None
        else:
            if model_type == 'mag':
                affordance_maps = mag_output

            skipped_affordance_maps = affordance_maps.data.cpu().numpy()

            affordance_maps = list()
            directions = list()
            cur = 0
            for id in range(len(skipped_affordance_maps) + len(skip_id_list)):
                if id in skip_id_list:
                    affordance_maps.append(None)
                    directions.append(None)
                else:
                    affordance_maps.append(skipped_affordance_maps[cur])
                    directions.append(skipped_directions[cur])
                    cur += 1

        return affordance_maps, directions, position_masks

    def get_position_affordance(self, observations, torch_tensor=False):
        """Get position affordance maps.

        Args:
            observations: list of dict
                - image: [W, H, 10]. dtype: float32
            torch_tensor: Whether the retuen value is torch tensor (default is numpy array). torch tensor is used for training.
        Return:
            affordance_maps: numpy array/torch tensor, [B, K, W, H]
        """
        skip_id_list = list()
        scene_inputs = []
        for observation in observations:
            scene_inputs.append(
                observation['image'][:, :, 3:].transpose([2, 0, 1]))
        scene_input_tensor = torch.from_numpy(np.stack(scene_inputs))

        affordance_maps = self.pos_model.forward(
            scene_input_tensor.to(self.device_pos))  # [B, K, W, H]
        if not torch_tensor:
            affordance_maps = 1 - F.softmax(affordance_maps, dim=1)[:, 0]
            affordance_maps = affordance_maps.data.cpu().numpy()

        return affordance_maps

    def to(self, device_pos, device_dir):
        self.device_pos = device_pos
        self.device_dir = device_dir
        self.pos_model = self.pos_model.to(device_pos)
        self.dir_model = self.dir_model.to(device_dir)
        return self

    def eval(self):
        self.pos_model.eval()
        self.dir_model.eval()

    def train(self):
        self.pos_model.train()
        self.dir_model.train()


if __name__ == '__main__':
    plt.imshow(gkern(480, 640, 3), interpolation='none')
    plt.savefig('gaussian-vis.png')
    