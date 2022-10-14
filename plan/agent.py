import os
import sys
sys.path.append('../reason')
sys.path.append('../interact')
import h5py
import pickle
import numpy as np
import torch

from PIL import Image
from model import Model
from sim import PybulletSim
from models_dy import DynaNetGNN
from visualize import visualize_graph, visualize_affordance
from utils import *

class Agent(object):
    def __init__(self, args, phase, data_folder):
        use_gpu = torch.cuda.is_available()
        
        self.args = args
        self.phase = phase
        self.data_folder = data_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''
        Load dynamics predictive model
        '''
        self.model_dy = DynaNetGNN(args, use_gpu=use_gpu)
        model_dy_path = '../reason/pre-trained/dynamics_best.pth'
        self.model_dy.load_state_dict(torch.load(model_dy_path))
        self.model_dy.to(self.device)
        self.model_dy.eval()
        print("==> dynamics network loaded from {}".format(model_dy_path))

        self.model_graph = DynaNetGNN(args, use_gpu=use_gpu)
        if phase == 'train' or phase == 'valid':
            model_graph_path = '../reason/pre-trained/dynamics_best.pth'
        if phase == 'unseen':
            model_graph_path = '../reason/pre-trained/graph_best.pth'
        self.model_graph.load_state_dict(torch.load(model_graph_path))
        self.model_graph.to(self.device)
        self.model_graph.eval()
        print("==> relation network loaded from {}".format(model_graph_path))

        '''
        Load interaction model
        '''
        self.model_interact = Model(num_directions=18, model_type='mag')
        self.model_interact.to(self.device, self.device)
        checkpoint = torch.load(os.path.join('../interact', 'pre-trained', 'latest.pth'), map_location=self.device)
        self.model_interact.pos_model.load_state_dict(checkpoint['pos_state_dict'])
        print('==> pos model loaded')
        self.model_interact.dir_model.load_state_dict(checkpoint['dir_state_dict'])
        print('==> dir model loaded')
        self.model_interact.eval()

        self.sim = PybulletSim(False, 0.05)
        self.img_path = "../interact/data/{}/{}".format(phase, data_folder)
        stats_path = "../interact/data/{}/{}/stats.h5".format(phase, data_folder)
        self.stats = h5py.File(stats_path, "r")

    def get_data(self, sample_idx):
        features = torch.tensor(self.stats["features"][sample_idx], dtype=torch.float32).to(self.device)
        n_frames, n_kp, feature_dim = features.size()
        features = features.view(1, n_frames, n_kp, feature_dim)

        data_path = os.path.join(self.img_path, str(sample_idx+1), 'data.h5')
        data = h5py.File(data_path, "r")

        obj_types = np.array([obj_type.decode('ascii') for obj_type in data['object_type']])
        cause_indices = np.where(obj_types=='Switch')[0]
        obj_types_cause_mask = np.zeros(self.args.n_kp)
        obj_types_cause_mask[cause_indices] = 1
        
        positions = torch.tensor(self.stats["positions"][sample_idx], dtype=torch.float32).to(self.device)
        positions = positions.view(1, n_frames, n_kp, 3)
        base_position = positions[:, 0]
        actions = torch.tensor(self.stats["actions"][sample_idx], dtype=torch.float32).to(self.device)
        actions = actions.view(1, n_frames, n_kp, 6)
        
        kps = torch.cat((features, positions), -1).to(self.device)
        # zero out switch feature
        kps[:, :, cause_indices, :self.args.feature_dim] = 0

        gt_edge_type = np.array(self.stats["relation"][sample_idx], dtype=np.int32)

        return kps, actions, gt_edge_type, base_position, obj_types_cause_mask

    def run_agent(self, sample_idx):
        _, _, _, base_position, obj_types_cause_mask = self.get_data(sample_idx)
        cause_indices = np.where(obj_types_cause_mask == 1)[0]
        sample_path = os.path.join(self.img_path, str(sample_idx+1))

        # restore simulation environment
        with open(os.path.join(sample_path, 'scene_state.pickle'), 'rb') as f:
            init_scene = pickle.load(f)
        observation = self.sim.reset(init_scene, goal_conditioned=True)

        init_image = Image.fromarray((observation['image'][:, :, :3] * 255).astype(np.uint8))
        init_state = self.get_state(sample_idx, init_image, cause_indices, base_position)[0]

        goal_image_path = os.path.join(self.img_path, str(sample_idx+1), 'fig_{}.png'.format(np.random.choice(30)))
        goal_image = Image.open(goal_image_path).convert('RGB')
        goal_state = self.get_state(sample_idx, goal_image, cause_indices, base_position)[0]

        # if goal is same as init, choose another goal image
        while torch.allclose(init_state[:, :self.args.feature_dim], goal_state[:, :self.args.feature_dim], atol=0.3, rtol=0.2):
            goal_image_path = os.path.join(self.img_path, str(sample_idx+1), 'fig_{}.png'.format(np.random.choice(30)))
            goal_image = Image.open(goal_image_path).convert('RGB')
            goal_state = self.get_state(sample_idx, goal_image, cause_indices, base_position)[0]

        success_rates = []
        for agent_type in ['graph', 'predictive', 'graph-predictive']:
            self.agent_type = agent_type
            if agent_type == 'graph':
                success_rate = self.run_graph_based_agent(sample_idx, init_image, goal_image, use_predictive=False)
                success_rates.append(success_rate)
            if agent_type == 'predictive':
                self.sim.reset_init_joint_state()
                success_rate = self.run_predictive_agent(sample_idx, init_image, goal_image)
                success_rates.append(success_rate)
            if agent_type == 'graph-predictive':
                self.sim.reset_init_joint_state()
                success_rate = self.run_graph_based_agent(sample_idx, init_image, goal_image, use_predictive=True)
                success_rates.append(success_rate)
        return success_rates
    
    def run_predictive_agent(self, sample_idx, init_image, goal_image, max_steps=8):
        kps, actions, gt_edge_type, base_position, obj_types_cause_mask = self.get_data(sample_idx)
        
        # ================Phase 1: Infer relation graph==================
        graph = self.model_dy.graph_inference(kps, actions)
        n_kp = kps.size(2)
        
        _, edge_type_logits = graph[1], graph[3]
        idx_pred = torch.argmax(edge_type_logits, dim=3)[0]
        # zero out effect columns
        cause_indices = np.where(obj_types_cause_mask == 1)[0]
        effect_indices = np.where(obj_types_cause_mask == 0)[0]
        idx_pred[:, effect_indices] = 0
        for row in cause_indices:
            for col in cause_indices:
                idx_pred[row, col] = 0
        idx_pred = idx_pred.data.cpu().numpy()

        # print("pred graph: \n", idx_pred)
        # print("gt graph: \n", gt_edge_type)

        sample_path = os.path.join(self.img_path, str(sample_idx+1))

        # ================Phase 2: Goal-conditioned manipulation============
        observation = self.sim.get_observation()
        position_affordance = self.model_interact.get_position_affordance([observation])[0]

        sample_planning_path = os.path.join('vis/{}/{}/planning-{}'.format(self.phase, self.data_folder, self.agent_type), str(sample_idx+1))
        os.system('mkdir -p {}'.format(sample_planning_path))
        visualize_affordance(sample_planning_path, position_affordance.copy(), observation)

        init_image.save(os.path.join(sample_planning_path, 'init.png'))
        goal_image.save(os.path.join(sample_planning_path, 'goal.png'))
        goal_state = self.get_state(sample_idx, goal_image, cause_indices, base_position)[0]
        visualize_graph(sample_path, sample_planning_path, idx_pred)

        for step in range(max_steps):
            cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
            cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]

            # Termination condition
            if torch.allclose(cur_state[:, :self.args.feature_dim], goal_state[:, :self.args.feature_dim], atol=0.3, rtol=0.2):
                cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
                print("Sample {} Success!".format(sample_idx+1))
                return 1

            # =================================================================
            # get action candidates for current scene
            observation = self.sim.get_observation()
            position_affordance = self.model_interact.get_position_affordance([observation])[0]
            action_candidates = self.get_action_space(position_affordance, observation)
            action_candidates = torch.FloatTensor(action_candidates).to(self.device)
            if action_candidates.size(0) == 0:
                continue
            # =================================================================
            min_dist = float('inf')
            kp_cur = cur_state.view(1, 1, n_kp, self.args.state_dim)
            for action_cur in action_candidates:
                kp_pred = self.model_dy.dynam_prediction(kp_cur, graph, action_cur[:, 2:])
                kp_pred[:, cause_indices, :] = 0
                dist = torch.linalg.norm(kp_pred - goal_state[:, :self.args.feature_dim]) # L2 distance between predicted state and goal state

                if dist < min_dist:
                    min_dist = dist
                    best_action = torch.clone(action_cur)

            action_idx = torch.nonzero(best_action, as_tuple=True)[0][0]
            best_action = best_action[action_idx].data.cpu().numpy()

            cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
            self.execute_action(best_action)

        # get and return per-object acc
        cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
        cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(max_steps+1)))
        cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]
        return self.get_per_object_acc(gt_edge_type, cause_indices, cur_state, goal_state)


    def run_graph_based_agent(self, sample_idx, init_image, goal_image, use_predictive=False, max_steps=8):
        kps, actions, gt_edge_type, base_position, obj_types_cause_mask = self.get_data(sample_idx)
        # ================Phase 1: Infer relation graph==================
        graph = self.model_graph.graph_inference(kps, actions)
        graph_2 = self.model_dy.graph_inference(kps, actions)
        n_kp = kps.size(2)
        
        _, edge_type_logits = graph[1], graph[3]
        idx_pred = torch.argmax(edge_type_logits, dim=3)[0]
        # zero out effect columns
        cause_indices = np.where(obj_types_cause_mask == 1)[0]
        effect_indices = np.where(obj_types_cause_mask == 0)[0]
        idx_pred[:, effect_indices] = 0
        for row in cause_indices:
            for col in cause_indices:
                idx_pred[row, col] = 0
        idx_pred = idx_pred.data.cpu().numpy()

        # print("pred graph: \n", idx_pred)
        # print("gt graph: \n", gt_edge_type)

        # ================Phase 2: Goal-conditioned manipulation============
        sample_path = os.path.join(self.img_path, str(sample_idx+1))
        observation = self.sim.get_observation()
        position_affordance = self.model_interact.get_position_affordance([observation])[0]

        sample_planning_path = os.path.join('vis/{}/{}/planning-{}'.format(self.phase, self.data_folder, self.agent_type), str(sample_idx+1))
        os.system('mkdir -p {}'.format(sample_planning_path))
        visualize_affordance(sample_planning_path, position_affordance.copy(), observation)

        init_image.save(os.path.join(sample_planning_path, 'init.png'))
        goal_image.save(os.path.join(sample_planning_path, 'goal.png'))
        init_state = self.get_state(sample_idx, init_image, cause_indices, base_position)[0]
        goal_state = self.get_state(sample_idx, goal_image, cause_indices, base_position)[0]
        visualize_graph(sample_path, sample_planning_path, idx_pred)

        diff_indices = []
        for i in range(init_state.shape[0]):
            if not torch.allclose(init_state[i, :self.args.feature_dim], goal_state[i, :self.args.feature_dim], atol=0.3, rtol=0.2):
                diff_indices.append(i)

        if use_predictive:
            for step in range(max_steps):
                cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
                cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]

                diff_indices = []
                for i in range(cur_state.shape[0]):
                    if not torch.allclose(cur_state[i, :self.args.feature_dim], goal_state[i, :self.args.feature_dim], atol=0.3, rtol=0.2):
                        diff_indices.append(i)

                proposed_cause_ids = set()
                for diff_idx in diff_indices:
                    tmp = np.argwhere(idx_pred[diff_idx] == 1).flatten()
                    proposed_cause_ids.update(tmp)

                # Termination condition
                if torch.allclose(cur_state[:, :self.args.feature_dim], goal_state[:, :self.args.feature_dim], atol=0.3, rtol=0.2):
                    cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
                    print("Sample {} Success!".format(sample_idx+1))
                    return 1

                # =================================================================
                # filter out action that does are not suggested by the relation graph
                observation = self.sim.get_observation()
                position_affordance = self.model_interact.get_position_affordance([observation])[0]
                action_candidates = self.get_action_space(position_affordance, observation)

                best_actions = []
                for action in action_candidates:
                    action_idx = np.nonzero(action)[0][0]
                    position = action[action_idx, :2].astype(np.int32)
                    pixel_index = np.ravel_multi_index(position, position_affordance.shape)
                    action_body_id = int(self.sim.body_id_pts[pixel_index])
                    if (action_body_id-1) in proposed_cause_ids:
                        best_actions.append(action)
                if len(best_actions) == 0:
                    continue
                # =================================================================
                best_actions = torch.FloatTensor(best_actions).to(self.device)

                cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
                cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]
                kp_cur = cur_state.view(1, 1, n_kp, self.args.state_dim)
            
                min_dist = float('inf')
                for action_cur in best_actions:
                    kp_pred = self.model_dy.dynam_prediction(kp_cur, graph_2, action_cur[:, 2:])
                    kp_pred[:, cause_indices, :] = 0
                    dist = torch.linalg.norm(kp_pred - goal_state[:, :self.args.feature_dim]) # L2 distance between predicted state and goal state

                    if dist < min_dist:
                        min_dist = dist
                        best_action = torch.clone(action_cur)

                action_idx = torch.nonzero(best_action, as_tuple=True)[0][0]
                best_action = best_action[action_idx].data.cpu().numpy()
            
                cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
                self.execute_action(best_action)

            # get and return per-object acc
            cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
            cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(max_steps+1)))
            cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]
            return self.get_per_object_acc(gt_edge_type, cause_indices, cur_state, goal_state)
        
        else:
            for step, diff_idx in enumerate(diff_indices):
                cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
                cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]

                # Termination condition
                if torch.allclose(cur_state[:, :self.args.feature_dim], goal_state[:, :self.args.feature_dim], atol=0.3, rtol=0.2):
                    cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
                    print("Sample {} Success!".format(sample_idx+1))
                    return 1

                observation = self.sim.get_observation()
                position_affordance = self.model_interact.get_position_affordance([observation])[0]
                action_space = self.get_action_space(position_affordance, observation)

                # retrieve cause index, if there are multiple, randomly select one
                proposed_cause_ids = np.argwhere(idx_pred[diff_idx] == 1).flatten()
                if len(proposed_cause_ids) == 0:
                    continue
                chosen_cause_id = np.random.choice(proposed_cause_ids)

                best_actions = []
                for action in action_space:
                    action_idx = np.nonzero(action)[0][0]
                    position = action[action_idx, :2].astype(np.int32)
                    pixel_index = np.ravel_multi_index(position, position_affordance.shape)
                    action_body_id = int(self.sim.body_id_pts[pixel_index])
                    if (action_body_id-1) == chosen_cause_id:
                        best_actions.append(action[action_idx])
                if len(best_actions) == 0:
                    continue
                best_action = best_actions[np.random.choice(len(best_actions))]
                # print("[action] ", best_action)
            
                cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(step+1)))
                self.execute_action(best_action)

            # get and return per-object acc
            cur_image = Image.fromarray((self.sim.get_observation()['image'][:, :, :3] * 255).astype(np.uint8))
            cur_image.save(os.path.join(sample_planning_path, 'step_{}.png'.format(len(diff_indices)+1)))
            cur_state = self.get_state(sample_idx, cur_image, cause_indices, base_position)[0]
            return self.get_per_object_acc(gt_edge_type, cause_indices, cur_state, goal_state)

    def get_per_object_acc(self, gt_edge_type, cause_indices, cur_state, goal_state):
        count = 0
        effect_indices = []
        for cause_idx in cause_indices:
            effect_indice = np.argwhere(gt_edge_type[:, cause_idx] == 1).flatten()
            effect_indices.extend(effect_indice)
        for effect_idx in effect_indices:
            if torch.allclose(cur_state[effect_idx, :self.args.feature_dim], goal_state[effect_idx, :self.args.feature_dim], atol=0.3, rtol=0.2):
                count += 1
        return count / len(effect_indices)

    def execute_action(self, action):
        best_pos_2d, best_pos, best_dir = action[:2], action[2:5], action[5:]
        observation, (reward, move_flag), terminate_flag, info = self.sim.step([0, int(best_pos_2d[0]), int(best_pos_2d[1])])
        if not terminate_flag: # if position is valid
            self.sim.step([1, best_dir[0], best_dir[1], best_dir[2]])

    def get_action_space(self, position_affordance, observation):
        action_space = []
        for _ in range(7):
        # for body_id, link_id in self.sim.joint_states.keys():
            pos_idx = np.argmax(position_affordance)
            body_id = int(self.sim.body_id_pts[pos_idx])
            link_id = int(self.sim.link_id_pts[pos_idx])
            if (body_id, link_id) not in self.sim.joint_states:
                continue
            position = np.array(np.unravel_index(pos_idx, position_affordance.shape))
            position = position.tolist()
            position_affordance[position[0]-10:position[0]+10, position[1]-10:position[1]+10] = 0

            # Get direction for each position
            direction_affordance, directions, _ = self.model_interact.get_direction_affordance([observation], [position], 'mag')
            dir_idx = np.argmax(direction_affordance[0])
            direction = directions[0][dir_idx]
            directions = [direction, -direction] # add both directions into action candidates

            position_3d = self.sim.xyz_pts[pos_idx]
            body_id = int(self.sim.body_id_pts[pos_idx])
            
            for direct in directions:
                action = np.zeros((self.args.n_kp, 2+6))
                action[body_id-1, :] = np.concatenate((position, position_3d, direct))
                # print("inferred action: ", np.concatenate((position_3d, direct)))
                action_space.append(action)

        return np.array(action_space)

    def get_state(self, sample_idx, image, cause_indices, base_position):
        data_path = os.path.join(self.img_path, str(sample_idx+1), 'data.h5')
        data = h5py.File(data_path, "r")
        bbox = np.array(data["bbox"])
        num_objects = bbox.shape[1]

        img_feature = torch.zeros(1, self.args.n_kp, self.args.feature_dim).to(self.device)
        img_feature[:, :num_objects] = torch.tensor(extract_central_feature(transform(image).view(1, 3, 224, 224), bbox))
        img_feature[:, cause_indices, :] = 0
        state = torch.cat((img_feature, base_position), -1)
        return state
