import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Visualize the scene graph
def visualize_graph(sample_path, sample_planning_path, pred_graph):
    image_path = os.path.join(sample_planning_path, 'init.png')
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
    im.save(os.path.join(sample_planning_path, 'scene_graph.png'))

def visualize_affordance(sample_planning_path, affordance_map, observation):
    affordance_map -= np.min(affordance_map)
    affordance_map /= np.max(affordance_map)
    cmap = plt.get_cmap('jet')
    affordance_map = cmap(affordance_map)[..., :3]
    color_image = observation['image'][:, :, :3]
    affordance_map = affordance_map * 0.8 + 0.2 * color_image
    affordance_image = Image.fromarray((affordance_map * 255).astype(np.uint8))
    affordance_image.save(os.path.join(sample_planning_path, 'affordance_map.png'))
