import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from progressbar import ProgressBar

from data import FeatureValidateDataset

class NodeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return F.log_softmax(self.fc2(x), dim=1)

use_gpu = torch.cuda.is_available()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
exp_name = "data-final"

if __name__ == '__main__':
    # train a classifier for each category
    FeatureValidateDataset(exp_name, None, load_data=True)

    for obj_type in ['lamp', 'door', 'toy']:
        print("[{}] Validating {} features".format(exp_name, obj_type))
        datasets, dataloaders, data_n_batches = {}, {}, {}

        dataset = FeatureValidateDataset(exp_name, obj_type)
        train_set, valid_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])
        print("train size: {}, test size: {}".format(len(train_set), len(valid_set)))
        for phase in ['train', 'valid']:
            datasets[phase] = train_set if phase == 'train' else valid_set

            dataloaders[phase] = DataLoader(
                datasets[phase], batch_size=128,
                shuffle=True if phase == 'train' else False, num_workers=16)

            data_n_batches[phase] = len(dataloaders[phase])
        
        # classification model
        model = NodeDecoder().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 5
        for epoch in range(epochs):
            for phase in ['train', 'valid']:
                total_loss = 0.
                correct = 0.
                model.train(phase == 'train')
                loader = dataloaders[phase]
                bar = ProgressBar(data_n_batches[phase])

                for i, item in bar(enumerate(loader)):
                    data, target = item[0].to(device), item[1].flatten().long().to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    # print(loss)
                    # print(torch.argmax(output, dim=1)[torch.where(torch.argmax(output, dim=1) != target)])
                    # print(target[torch.where(torch.argmax(output, dim=1) != target)])

                    correct += (torch.argmax(output, dim=1) == target).sum().item()
                    total_loss += loss.item() * data.size(0)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # print training statistics
                # calculate average loss over an epoch
                print('[', phase, ']')
                print('Epoch: {} \tLoss: {:.6f}'.format(epoch+1, total_loss / len(datasets[phase])))
                print('Epoch: {} \tAccuracy: {:.6f}%'.format(epoch+1, 100 * correct / len(datasets[phase])))
    
        os.system('mkdir -p decoders/{}'.format('final-large'))
        torch.save(model.state_dict(), 'decoders/{}/{}_predictor.pth'.format('final-large', obj_type))
    