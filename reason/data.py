import multiprocessing as mp
import os

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

class BoardDataset(Dataset):
    def __init__(self, args, phase):
        self.phase = phase
        if phase == 'train':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_train_data)
        if phase == 'valid':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_valid_data)
        if phase == 'unseen':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_valid_data)

        self.num_samples = len([f for f in os.listdir(self.data_dir) if f.isdigit()])
        self.stats_path = os.path.join(self.data_dir, 'stats.h5')
        self.n_kp = args.n_kp
        self.feature_dim = 256
        self.args = args

    def __len__(self):
        data = h5py.File(self.stats_path, "r")
        return np.array(data["features"]).shape[0]

    def __getitem__(self, idx):
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError
        else:
            data = h5py.File(self.stats_path, "r")
            features = torch.tensor(data["features"][idx], dtype=torch.float32)
            
            file_path = os.path.join(self.data_dir, str(idx+1), 'data.h5')
            stats = h5py.File(file_path, "r")
            obj_types = np.array([obj_type.decode('ascii') for obj_type in stats['object_type']])

            # zero out cause features to achieve more stable training
            cause_indices = np.where(obj_types=='Switch')[0]
            
            positions = torch.tensor(data["positions"][idx], dtype=torch.float32)
            actions = torch.tensor(data["actions"][idx], dtype=torch.float32)
            
            kps_preload = torch.cat((features, positions), -1)
            # zero out switch feature
            kps_preload[:, cause_indices, :self.feature_dim] = 0

            gt_edge_type, states = torch.tensor(data["relation"][idx], dtype=torch.long), \
                                   torch.tensor(data["states"][idx], dtype=torch.float32)

            obj_types_cause_mask = np.zeros(self.n_kp)
            obj_types_cause_mask[len(obj_types):] = -1
            obj_types_cause_mask[cause_indices] = 1
        return idx, kps_preload, actions, gt_edge_type, states, obj_types_cause_mask

class FeatureValidateDataset(Dataset):
    def __init__(self, exp_name, obj_type, load_data=False):
        self.pred_dir = '../interact/data/{}'.format('prediction')
        self.stats_path = os.path.join(self.pred_dir, 'stats.h5')
        self.data_path = os.path.join(self.pred_dir, '{}_prediction_data.h5'.format(obj_type))
        self.feature_dim = 256
        self.state_dim = 1
        self.exp_name = exp_name
        if load_data:
            self.load_all_data()

    def __len__(self):
        data = h5py.File(self.data_path, "r")
        self.num_samples = np.array(data["features"]).shape[0]
        return self.num_samples
    
    def __getitem__(self, idx):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError
        else:
            data = h5py.File(self.data_path, "r")
            feature, state = torch.tensor(data["features"][idx], dtype=torch.float32),\
                             torch.tensor(data["states"][idx], dtype=torch.float32)
        return feature, state

    def load_all_data(self):
        switch_feature_arrays, lamp_feature_arrays, door_feature_arrays, toy_feature_arrays =\
            [np.empty(shape=(0, self.feature_dim))] * 4
        switch_state_arrays, lamp_state_arrays, door_state_arrays, toy_state_arrays =\
            [np.empty(shape=(0, self.state_dim))] * 4

        for phase in ['train', 'valid', 'unseen']:
            data_dir = '../interact/data/{}/{}'.format(phase, self.exp_name)
            if phase == 'train':
                num_samples = 2000
            if phase == 'valid':
                num_samples = 2000
            if phase == 'unseen':
                num_samples = 2000
            for idx in range(num_samples):
                try:
                    file_path = os.path.join(data_dir, str(idx+1), 'data.h5')
                    data = h5py.File(file_path, "r")
                        
                    # Get features for each category (only for evaluation)
                    # features: total_num_objects * 1024
                    # states: total_num_objects * 1
                    # ==================================================================================
                    obj_types = np.array([obj_type.decode('ascii') for obj_type in data['object_type']])

                    feature_array = np.array(data["features"])
                    state_array = np.array(data["states"])

                    lamp_features = feature_array[:, np.where(obj_types=='Lamp')[0]]
                    lamp_features = lamp_features.reshape(-1, lamp_features.shape[-1])
                    lamp_feature_arrays = np.concatenate((lamp_feature_arrays, lamp_features), axis=0)

                    lamp_states = state_array[:, np.where(obj_types=='Lamp')[0]]
                    lamp_states = lamp_states.reshape(-1, lamp_states.shape[-1])
                    lamp_state_arrays = np.concatenate((lamp_state_arrays, lamp_states), axis=0)

                    door_features = feature_array[:, np.where(obj_types=='Door')[0]]
                    door_features = door_features.reshape(-1, door_features.shape[-1])
                    door_feature_arrays = np.concatenate((door_feature_arrays, door_features), axis=0)

                    door_states = state_array[:, np.where(obj_types=='Door')[0]]
                    door_states = door_states.reshape(-1, door_states.shape[-1])
                    door_state_arrays = np.concatenate((door_state_arrays, door_states), axis=0)

                    toy_features = feature_array[:, np.where(obj_types=='Toy')[0]]
                    toy_features = toy_features.reshape(-1, toy_features.shape[-1])
                    toy_feature_arrays = np.concatenate((toy_feature_arrays, toy_features), axis=0)

                    toy_states = state_array[:, np.where(obj_types=='Toy')[0]]
                    toy_states = toy_states.reshape(-1, toy_states.shape[-1])
                    toy_state_arrays = np.concatenate((toy_state_arrays, toy_states), axis=0)
                    # ================================================================================== 
                    print("[{}][{}] Finish processing feature-state pairs".format(phase, idx+1))
                # log error to file
                except Exception as e:
                    print(e)
                    break

        # Create feature-state files for training classifiers for all categories
        for obj_type in ['lamp', 'door', 'toy']:
            file_path = os.path.join(self.pred_dir, '{}_prediction_data.h5'.format(obj_type))
            hf = h5py.File(file_path, "w")
            
            if obj_type == 'lamp':
                hf.create_dataset('features', data=lamp_feature_arrays)
                hf.create_dataset('states', data=lamp_state_arrays)
            elif obj_type == 'door':
                hf.create_dataset('features', data=door_feature_arrays)
                hf.create_dataset('states', data=door_state_arrays)
            elif obj_type == 'toy':
                hf.create_dataset('features', data=toy_feature_arrays)
                hf.create_dataset('states', data=toy_state_arrays)
            hf.close()
