import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--nf_hidden_dy', type=int, default=16)
parser.add_argument('--n_kp', type=int, default=0, help="the number of keypoint")

parser.add_argument('--version', default='OURS', help='version of model (OURS, wo-inference')

'''
train
'''
parser.add_argument('--exp', default='exp', type=str, help='name of experiment. The directory to save data is exp/[exp]')
parser.add_argument('--exp_train_data', default='2-door-original', type=str, help='path to the train data for exp')
parser.add_argument('--exp_valid_data', default='2-door-original', type=str, help='path to the test data for exp')
parser.add_argument('--exp_predictor', default='2-door-original', type=str, help='path to the predictor for exp')
parser.add_argument('--random_seed', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)

parser.add_argument('--num_workers', type=int, default=10)

parser.add_argument('--log_per_iter', type=int, default=100, help="print log every x iterations")
parser.add_argument('--ckp_per_iter', type=int, default=5000, help="save checkpoint every x iterations")

parser.add_argument('--eval', type=int, default=0)

# for dynamics prediction
parser.add_argument('--min_res', type=int, default=0, help="minimal observation for the inference module")

parser.add_argument('--dy_model', default='gnn', help='the model for dynamics prediction - gnn|mlp')
parser.add_argument('--en_model', default='cnn', help='the model for encoding - gru|cnn|tra')
parser.add_argument('--n_his', type=int, default=5, help='number of frames used as input')
parser.add_argument('--n_identify', type=int, default=0, help='number of frames used for graph identification')
parser.add_argument('--n_roll', type=int, default=5, help='number of rollout steps for training')

parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--node_attr_dim', type=int, default=0)
parser.add_argument('--edge_attr_dim', type=int, default=0)
parser.add_argument('--edge_type_num', type=int, default=0)
parser.add_argument('--edge_st_idx', type=int, default=0, help="whether to exclude the first edge type")
parser.add_argument('--edge_share', type=int, default=0,
                    help="whether forcing the info being the same for both directions")

parser.add_argument('--preload_dy', type=int, default=1, help="whether to load saved dynamics model")
parser.add_argument('--edge_encoding_dim', type=int, default=0, help="dimension of edge feature")
parser.add_argument('--encode_action', type=int, default=0, help="whether to encode action")

'''
model
'''
# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)

def gen_args():
    args = parser.parse_args()
    args.data_names = ['states', 'actions', 'rels']

    # action encoding || pos + dir
    args.action_dim = 256 if args.encode_action else 6
    # object feature + obj pos
    args.state_dim = 256 + 3
    # object featur dim
    args.feature_dim = 256
    # none, causal
    args.relation_dim = 2

    # size of the latent causal graph
    args.node_attr_dim = 0
    args.edge_attr_dim = args.edge_encoding_dim
    args.edge_type_num = 2

    args.prior = torch.FloatTensor(np.array([0.5, 0.5])).cuda()

    return args
