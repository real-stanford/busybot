import os
import cv2
import h5py
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from config import gen_args
from models_dy import DynaNetGNN
from utils import count_parameters
from decoder import NodeDecoder

args = gen_args()
use_gpu = torch.cuda.is_available()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

'''
get model for classification
'''

lamp_predictor = NodeDecoder().to(device)
door_predictor = NodeDecoder().to(device)
toy_predictor = NodeDecoder().to(device)

lamp_predictor.load_state_dict(torch.load('decoders/lamp_predictor.pth', map_location=device))
door_predictor.load_state_dict(torch.load('decoders/door_predictor.pth', map_location=device))
toy_predictor.load_state_dict(torch.load('decoders/toy_predictor.pth', map_location=device))

lamp_predictor.eval()
door_predictor.eval()
toy_predictor.eval()

'''
define model for dynamics prediction
'''
model_dy = DynaNetGNN(args, use_gpu=use_gpu)
print("model_dy #params: %d" % count_parameters(model_dy))
model_dy_path = 'pre-trained/dynamics_best.pth'
print("Loading saved ckp for dynamics net from %s" % model_dy_path)
model_dy.load_state_dict(torch.load(model_dy_path))

# criterion
criterionMSE = nn.MSELoss()

if use_gpu:
    criterionMSE = criterionMSE.cuda()
    model_dy = model_dy.cuda()

file_dir = '../interact/data/valid/plan-multi'
model_dy.eval()

def visualize_graph(sample_path, pred_graph, observation_length=0):
    image_path = os.path.join(sample_path, 'fig_{}.png'.format(observation_length-1))
    img = np.array(Image.open(image_path).convert('RGBA'))
    data_path = os.path.join(sample_path, 'data.h5')
    data = h5py.File(data_path, "r")
    bbox = np.array(data['bbox'])[0]
    centers = []
    for i in range(len(bbox)):
        central_pos = ((bbox[i, 1] + bbox[i, 3]) // 2, (bbox[i, 0] + bbox[i, 2]) // 2)
        centers.append(central_pos)

    arrow_layer = np.ones_like(img) * 255
    arrow_layer[:, :, 3] = 0
    gt_graph = np.array(data['relation'])
    causal_pairs = np.where(gt_graph.T == 1)
    for i in range(len(causal_pairs[0])):
        cause, effect = causal_pairs[0][i], causal_pairs[1][i]
        start_pts, end_pts = centers[cause], centers[effect]
        arrow_layer = cv2.arrowedLine(arrow_layer, start_pts, end_pts, color=(0, 255, 0, 255), thickness=3, tipLength=0.04)
        img = cv2.arrowedLine(img, start_pts, end_pts, color=(0, 255*0.6, 0, 255), thickness=3, tipLength=0.04)

    pred_causal_pairs = np.where(pred_graph.T == 1)
    for i in range(len(pred_causal_pairs[0])):
        pred_cause, pred_effect = pred_causal_pairs[0][i], pred_causal_pairs[1][i]
        start_pts, end_pts = centers[pred_cause], centers[pred_effect]
        angle = np.arctan2(end_pts[1] - start_pts[1], end_pts[0] - start_pts[0]) * 180 / np.pi
        if abs(angle) > 45 and abs(angle) < 135:
            arrow_layer = cv2.arrowedLine(arrow_layer, (start_pts[0]+15, start_pts[1]), (end_pts[0]+15, end_pts[1]), 
                color=(255, 0, 0, 255), thickness=3, tipLength=0.04)
            img = cv2.arrowedLine(img, (start_pts[0]+15, start_pts[1]), (end_pts[0]+15, end_pts[1]), 
                color=(255*0.6, 0, 0, 255), thickness=3, tipLength=0.04)
        else:
            arrow_layer = cv2.arrowedLine(arrow_layer, (start_pts[0], start_pts[1]+15), (end_pts[0], end_pts[1]+15), 
                color=(255, 0, 0, 255), thickness=3, tipLength=0.04)
            img = cv2.arrowedLine(img, (start_pts[0], start_pts[1]+15), (end_pts[0], end_pts[1]+15), 
                color=(255*0.6, 0, 0, 255), thickness=3, tipLength=0.04)
    
    res = cv2.addWeighted(img, 0.6, arrow_layer, 0.4, 0.0)
    im = Image.fromarray(res).convert('RGB')
    os.system('mkdir -p {}'.format(os.path.join('vis/graphs')))
    im.save(os.path.join('vis/graphs', 'scene_graph_{}.png'.format(observation_length)))


if __name__ == '__main__':
    input_data = h5py.File(os.path.join(file_dir, 'stats.h5'), "r")
    sample_idx = 29 # file number = sample_idx + 1
    sample_path = os.path.join(file_dir, str(sample_idx+1))

    features = torch.tensor(input_data["features"][sample_idx], dtype=torch.float32)
    states = torch.tensor(input_data['states'][sample_idx], dtype=torch.float32)
    actions = torch.tensor(input_data['actions'][sample_idx], dtype=torch.float32)
    positions = torch.tensor(input_data['positions'][sample_idx], dtype=torch.float32)
    gt_edge_type = torch.tensor(input_data['relation'][sample_idx], dtype=torch.long)

    file_path = os.path.join(file_dir, str(sample_idx+1), 'data.h5')
    sample_data = h5py.File(file_path, "r")
    obj_types = np.array([obj_type.decode('ascii') for obj_type in sample_data['object_type']])
    cause_indices = np.where(obj_types=='Switch')[0]
    obj_types_cause_mask = np.zeros(args.n_kp)
    obj_types_cause_mask[cause_indices] = 1
    cause_indices = np.where(obj_types_cause_mask == 1)[0]
    effect_indices = np.where(obj_types_cause_mask == 0)[0]

    kps_preload = torch.cat((features, positions), -1)
    kps_preload[:, cause_indices, :args.feature_dim] = 0

    n_samples = args.n_identify + args.n_his + args.n_roll

    B = 1
    kps = kps_preload.view(B, n_samples, args.n_kp, args.state_dim).cuda()
    actions = actions.view(B, n_samples, args.n_kp, 6).cuda()
    states = states.view(B, n_samples, args.n_kp, 1).cuda()

    for observe_length in range(4, 31):
        graph = model_dy.graph_inference(kps[:, :observe_length], actions[:, :observe_length])

        edge_attr, edge_type_logits = graph[1], graph[3]
        idx_pred = torch.argmax(edge_type_logits, dim=3)[0]
        idx_pred[:, effect_indices] = 0
        for row in cause_indices:
            for col in cause_indices:
                idx_pred[row, col] = 0

        print("pred graph: \n", idx_pred)
        print("gt graph: \n", gt_edge_type)
        visualize_graph(sample_path, idx_pred.data.cpu().numpy(), observe_length)
