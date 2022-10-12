import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
import h5py
from config import gen_args
from PIL import Image

class FeatureExtractDataset(Dataset):
    def __init__(self, data_dir, idx, num_frames):
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.num_frames = num_frames
        self.idx = idx
        self.data_dir = data_dir

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        img_path = os.path.join(self.data_dir, str(self.idx+1), 'fig_{}.png'.format(i))
        img = Image.open(img_path).convert('RGB')
        t_img = self.transform(img)
        return t_img

class DataProcessor():
    def __init__(self, args, phase):
        self.phase = phase
        if phase == 'train':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_train_data)
        if phase == 'valid':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_valid_data)
        if phase == 'unseen':
            self.data_dir = '../interact/data/{}/{}'.format(phase, args.exp_valid_data)
        
        self.num_samples = len([f for f in os.listdir(self.data_dir) if f.isdigit()])
        self.n_kp = args.n_kp
        self.num_frames = 30
        self.feature_dim = 256
        self.state_dim = 1
        self.action_dim = 6
        self.position_dim = 3
        self.model = models.resnet50(pretrained=True).eval()
        self.stats_path = os.path.join(self.data_dir, 'stats.h5')

    def add_feature_to_data(self):
        for idx in range(self.num_samples):
            try:
                image_dataset = FeatureExtractDataset(self.data_dir, idx, self.num_frames)
                file_path = os.path.join(self.data_dir, str(idx+1), 'data.h5')
                data = h5py.File(file_path, "a")

                if 'features' not in data.keys():
                    bbox = np.array(data["bbox"], dtype=np.int32)

                    image_loader = DataLoader(image_dataset, batch_size=64, shuffle=False, num_workers=4)
                    feature_array = []
                    for batch_images in image_loader:
                        feature_array.append(self.extract_central_feature(batch_images, bbox))
                    # feature_array: num_frames * num_objects * feature_dim
                    feature_array = np.vstack(feature_array)
                    data.create_dataset('features', data=np.array(feature_array))
                
                    print("[{}] Finish feature extraction".format(idx+1))
                else:
                    print("[{}] skip...".format(idx+1))

                data.close()
                
            # log error to file
            except Exception as e:
                print("[Error] ", e)
                break
    
    def load_data(self):
        # if not os.path.exists(self.stats_path):
        if True:
            interval = 10
            first_batch = True
            feature_arrays, action_arrays, state_arrays, relation_arrays, position_arrays, relation_annotated_arrays = [], [], [], [], [], []
            for idx in range(self.num_samples):
                try:
                    file_path = os.path.join(self.data_dir, str(idx+1), 'data.h5')
                    data = h5py.File(file_path, "r")
                    bbox = np.array(data["bbox"], dtype=np.int32)
                    num_objects = bbox.shape[1]

                    # feature_array: num_frames * n_kp * 256
                    feature_array = np.zeros((self.num_frames, self.n_kp, self.feature_dim))
                    feature_array[:, :num_objects, :] = np.array(data["features"])

                    # state_array: num_frames * n_kp * 1
                    state_array = np.zeros((self.num_frames, self.n_kp, self.state_dim))
                    state_array[:, :num_objects, :] = np.array(data["states"])

                    # action_array: num_frames * n_kp * 6
                    action_array = np.zeros((self.num_frames, self.n_kp, self.action_dim))
                    action_array[:, :num_objects, :] = np.array(data["actions"])

                    # relation_array: n_kp * n_kp
                    relation_array = np.zeros((self.n_kp, self.n_kp))
                    relation_array[:num_objects, :num_objects] = np.array(data['relation'])

                    # relation_annotated_array: n_kp * n_kp
                    relation_annotated_array = np.zeros((self.n_kp, self.n_kp))
                    relation_annotated_array[:num_objects, :num_objects] = np.array(data['relation_annotated'])

                    # position_array: num_frames * n_kp * 3
                    position_array = np.zeros((self.num_frames, self.n_kp, self.position_dim))
                    position_array[:, :num_objects, :] = np.array(data["positions"])

                    feature_arrays.append(feature_array)
                    position_arrays.append(position_array)
                    action_arrays.append(action_array)
                    state_arrays.append(state_array)
                    relation_arrays.append(relation_array)
                    relation_annotated_arrays.append(relation_annotated_array)

                    print("[{}] Finish processing data".format(idx+1))

                # log error to file
                except Exception as e:
                    print("[Error] ", e)
                    break

                if (idx + 1) % interval == 0:
                    if first_batch == True:
                        hf = h5py.File(self.stats_path, 'w')
                        hf.create_dataset('actions', (self.num_samples, self.num_frames, self.n_kp, self.action_dim))
                        hf.create_dataset('positions', (self.num_samples, self.num_frames, self.n_kp, self.position_dim))

                        hf.create_dataset('states', (self.num_samples, self.num_frames, self.n_kp, self.state_dim))
                        hf.create_dataset('features', (self.num_samples, self.num_frames, self.n_kp, self.feature_dim))

                        hf.create_dataset('relation', (self.num_samples, self.n_kp, self.n_kp))
                        hf.create_dataset('relation_annotated', (self.num_samples, self.n_kp, self.n_kp))

                        first_batch = False
                    else:
                        hf = h5py.File(self.stats_path, 'a')
                    
                    start_idx, end_idx = idx + 1 - interval, idx + 1
                    hf['actions'][start_idx:end_idx] = action_arrays
                    hf['positions'][start_idx:end_idx] = position_arrays
                    hf['states'][start_idx:end_idx] = state_arrays
                    hf['features'][start_idx:end_idx] = feature_arrays
                    hf['relation'][start_idx:end_idx] = relation_arrays
                    hf['relation_annotated'][start_idx:end_idx] = relation_annotated_arrays

                    hf.close()
                    feature_arrays, action_arrays, state_arrays, relation_arrays, position_arrays, relation_annotated_arrays = [], [], [], [], [], []
        else:
            print("[Skip] stats.h5 file already exists")
    
    def extract_central_feature(self, t_images, bbox):
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            t_images = t_images.cuda()
            self.model.cuda()

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.model.layer1.register_forward_hook(get_activation('out'))
        self.model(t_images)
        feature = F.interpolate(activation['out'], size=t_images.shape[-2:], mode='bilinear', align_corners=False).data.cpu().numpy()

        v = np.array(((bbox[0, :, 0] + bbox[0, :, 2]) // 2) * (224 / 480), dtype=np.int32)
        u = np.array(((bbox[0, :, 1] + bbox[0, :, 3]) // 2) * (224 / 640), dtype=np.int32)
        output = feature[:, :, v, u]

        output = np.swapaxes(output, 1, 2)
        return output

if __name__=='__main__':
    args = gen_args()
    for phase in ['valid']:
        data_processor = DataProcessor(phase=phase, args=args)
        data_processor.add_feature_to_data()
        data_processor.load_data()
