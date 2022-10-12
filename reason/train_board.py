import os
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import gen_args
from data import BoardDataset
from decoder import NodeDecoder
from models_dy import *
from utils import rand_int, count_parameters, AverageMeter, set_seed

args = gen_args()
set_seed(args.random_seed)

writer_dir = os.path.join('exp', args.exp)
os.system('mkdir -p {}'.format(writer_dir))
writer = SummaryWriter(writer_dir)

model_dir = os.path.join('model', args.exp)
os.system('mkdir -p {}'.format(model_dir))

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid', 'unseen']:
    datasets[phase] = BoardDataset(args, phase=phase)

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False, num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
get model for object state evluation
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
define model
'''
if args.dy_model == 'gnn':
    if args.version == 'wo-inference':
        model_dy = DynaNetGNN_wo_inference(args, use_gpu=use_gpu)
    else:
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)

    if args.preload_dy:
        # if resume from a pre-trained checkpoint
        model_dy_path = 'pre-trained/dynamics_best.pth'
        print("Loading saved ckp for dynamics net from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))
    print("model_dy trainable #params: %d" % count_parameters(model_dy))

# criterion
criterionMSE = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model_dy.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

if use_gpu:
    model_dy = model_dy.cuda()
    criterionMSE = criterionMSE.cuda()

prev_train_loss = np.inf
epochs = 200
for epoch in range(epochs):
    phases = ['train', 'valid', 'unseen'] if args.eval == 0 else ['valid', 'unseen']

    for phase in phases:
        model_dy.train(phase == 'train')
        # meter_loss = AverageMeter()
        meter_loss_mse = AverageMeter()

        if args.version != 'wo-infernece':
            meter_acc = AverageMeter()
            meter_recall = AverageMeter()
            meter_graph_acc = AverageMeter()

        loader = dataloaders[phase]
        num_samples = len(datasets[phase])

        num_correct = 0
        first_batch = True
        state_acc_list = []
        print("[Epoch {} || Phase {}]".format(epoch+1, phase))
        
        for idx, kps, actions, gt_edge_type, states, obj_types_cause_mask in loader:
            kps = kps.to(device)
            actions = actions.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                '''
                hyperparameter on the length of data
                '''
                n_his, n_kp = args.n_his, args.n_kp
                n_samples = args.n_identify + args.n_his + args.n_roll
                n_identify = args.n_identify

                '''
                load data
                '''
                B = kps.shape[0]
                kps = kps.view(B, n_samples, n_kp, args.state_dim)
                kps_id, kps_dy = kps[:, :n_identify], kps[:, n_identify:]
                states_id, states_dy = states[:, :n_identify], states[:, n_identify:]
                actions_id, actions_dy = actions[:, :n_identify], actions[:, n_identify:]

                '''
                step #1: identify the dynamics graph
                '''
                # randomize the observation length
                observe_length = rand_int(args.min_res, n_identify + 1)

                if args.version == 'wo-inference':
                    graph = None
                else:
                    graph = model_dy.graph_inference(
                        kps_id[:, :observe_length], actions_id[:, :observe_length], gt_graph=None)

                    # edge_attr: B x n_kp x n_kp x edge_attr_dim
                    # edge_type: B x n_kp x n_kp x edge_type_num
                    # edge_type_logits: B x n_kp x n_kp x edge_type_num
                    edge_attr, edge_type_logits = graph[1], graph[3]

                    idx_pred = torch.argmax(edge_type_logits, dim=3)
                    idx_pred = idx_pred.data.cpu().numpy()
                    edge_type_after_gumbel_softmax = np.round(graph[2][:, :, :, 1].data.cpu().numpy(), 2)

                    # print("[Edge attr]\n", edge_attr)
                    # print("[Edge type]\n", idx_pred)

                    # compare with ground truth edge type and compute precision and recall
                    edge_type_acc = []
                    edge_type_recall = []
                    graph_acc = 0.0
                    for i, et in enumerate(idx_pred):
                        cause_indices = np.where(obj_types_cause_mask[i] == 1)[0]
                        effect_indices = np.where(obj_types_cause_mask[i] == 0)[0]
                                
                        pred_graph = et[:, cause_indices][effect_indices, :]
                        gt_graph = gt_edge_type[i][:, cause_indices][effect_indices, :]
                        edge_type_acc.append(np.sum(np.array(np.equal(pred_graph, gt_graph))) / (gt_graph.size(0) * gt_graph.size(1)))

                        rows, cols = np.where(gt_graph == 1)
                        edge_type_recall.append(np.sum(pred_graph[rows, cols] == 1) / len(rows))

                        # if np.array_equal(pred_graph, gt_graph):
                        #     graph_acc += 1 / B

                    meter_acc.update(np.mean(edge_type_acc), B)
                    meter_recall.update(np.mean(edge_type_recall), B)
                    # meter_graph_acc.update(graph_acc, B)

                # step #2: dynamics prediction
                loss_mse, state_acc = 0., 0.
                kps_pred, kps_gt = [], []
                for j in range(args.n_roll):
                    kp_des = kps_dy[:, j + n_his].view(B, n_kp, args.state_dim)
                    state_des = states_dy[:, j + n_his].view(B, n_kp, 1)

                    kp_cur = kps_dy[:, j : j + n_his].view(B, n_his, n_kp, args.state_dim)
                    action_cur = actions_dy[:, j : j + n_his]
                    if args.version == 'wo-inference':
                        kp_pred = model_dy.dynam_prediction(kp_cur, action_cur)
                    else:
                        kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur)

                    if (epoch + 1) % 10 == 0:
                        batch_acc = 0.0
                        for ind, sample_idx in enumerate(idx):
                            data_dir = args.exp_train_data if phase == 'train' else args.exp_valid_data
                            sample_dir = os.path.join('../interact/data', phase, data_dir, str(sample_idx.data.cpu().numpy()+1))
                            sample_data = h5py.File(os.path.join(sample_dir, "data.h5"), "r")
                            obj_types = np.array([obj_type.decode('ascii') for obj_type in sample_data['object_type']])
                            
                            gt_states, pred_states = [], []
                            for i, pred_feature in enumerate(kp_pred[ind]):
                                if i < len(obj_types):
                                    if obj_types[i] == "Lamp":
                                        pred_state = torch.argmax(lamp_predictor(pred_feature[:args.feature_dim].reshape(1, args.feature_dim)), dim=1).data.cpu().numpy()
                                        pred_states.append(pred_state[0])
                                        gt_states.append(state_des[ind][i][0])
                                    elif obj_types[i] == "Door":
                                        pred_state = torch.argmax(door_predictor(pred_feature[:args.feature_dim].reshape(1, args.feature_dim)), dim=1).data.cpu().numpy()
                                        pred_states.append(pred_state[0])
                                        gt_states.append(state_des[ind][i][0])
                                    elif obj_types[i] == "Toy":
                                        pred_state = torch.argmax(toy_predictor(pred_feature[:args.feature_dim].reshape(1, args.feature_dim)), dim=1).data.cpu().numpy()
                                        pred_states.append(pred_state[0])
                                        gt_states.append(state_des[ind][i][0])
                                    
                            if np.array_equal(pred_states, gt_states):
                                batch_acc += 1 / B
                            # else:
                            #     if phase == 'valid':
                            #         print("Pred state => ", pred_states, "GT state => ", gt_states)
                        
                        state_acc += batch_acc / args.n_roll

                    # zero out trigger features
                    for i in range(kp_pred.size()[0]):
                        cause_indices = np.where(obj_types_cause_mask[i] == 1)
                        kp_pred[i, cause_indices, :] = 0

                    loss_mse_cur = criterionMSE(kp_pred, kp_des[:, :, :args.feature_dim])
                    loss_mse += loss_mse_cur / args.n_roll

                print("Epoch {} ==> Model loss: {}  ".format(epoch + 1, loss_mse.item()))
                if (epoch + 1) % 10 == 0:
                    print("Epoch {} ==> State loss: {}  ".format(epoch + 1, state_acc))

                # update meter
                meter_loss_mse.update(loss_mse.item(), B)
                # meter_loss.update(loss.item(), B)
                        
            first_batch = False
            state_acc_list.append(state_acc)

            # clip out sudden explosion in loss
            if phase == 'train' and loss_mse < prev_train_loss * 2:
                optimizer.zero_grad()
                loss_mse.backward()
                optimizer.step()
                prev_train_loss = loss_mse

        writer.add_scalar('{}/MSE'.format(phase), meter_loss_mse.avg, epoch + 1)
        if args.version != 'wo-inference':
            writer.add_scalar('{}/Edge type acc'.format(phase), meter_acc.avg, epoch + 1)
            writer.add_scalar('{}/Edge type recall'.format(phase), meter_recall.avg, epoch + 1)
            # writer.add_scalar('{}/Graph acc'.format(phase), meter_graph_acc.avg, epoch + 1)
        if (epoch + 1) % 10 == 0:
            writer.add_scalar('{}/State Acc'.format(phase), np.mean(state_acc_list) * 100, epoch + 1)
       
       # save model every 20 epochs
        if phase == 'train' and (epoch + 1) % 20 == 0:
            torch.save(model_dy.state_dict(), '{}/dy_net_train_epoch_{}.pth'.format(model_dir, epoch + 1))

writer.close()
