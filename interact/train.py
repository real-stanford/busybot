import argparse
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

import utils
from model import Model
from replay_buffer import ReplayBuffer
from sim import PybulletSim

parser = argparse.ArgumentParser()

# global
parser.add_argument('--exp', default='exp', type=str,
                    help='name of experiment. The directory to save data is exp/[exp]')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed of pytorch and numpy')
parser.add_argument('--snapshot_gap', default=10, type=int,
                    help='Frequence of saving the snapshot (e.g. visualization, model, optimizer)')
parser.add_argument('--num_visualization', default=None, type=int,
                    help='numer of visualization sequences, None means num_envs')

# environment
parser.add_argument('--num_envs', default=16, type=int,
                    help='number of envs, each env has a process')
parser.add_argument('--num_frames', default=10, type=int,
                    help='number of frames to collect per env')
parser.add_argument('--max_seq_len', default=1, type=int,
                    help='number of steps for each sequence')
parser.add_argument('--num_direction', default=18,
                    type=int, help='number of directions')
parser.add_argument('--action_distance', default=0.1,
                    type=float, help='dragging distance in each interaction')

# model
parser.add_argument('--model_type', default='mag', type=str,
                    choices=['mag'], help='model_type')

# training
parser.add_argument('--load_checkpoint', default=None, type=str,
                    help='exp name or a directpry of ckpt (suffix is .pth). Load the the checkpoint (model, optimizer) from another training exp')
parser.add_argument('--load_model_type', default=None,
                    type=str, nargs='+', help='pos or dir')
parser.add_argument('--pos_learning_rate', default=5e-4,
                    type=float, help='learning rate of the position optimizer')
parser.add_argument('--dir_learning_rate', default=1e-4,
                    type=float, help='learning rate of the direction optimizer')
parser.add_argument('--pos_learning_rate_decay', default=90,
                    type=int, help='learning rate decay for position')
parser.add_argument('--dir_learning_rate_decay', default=75,
                    type=int, help='learning rate decay for direction')
parser.add_argument('--epoch', default=800, type=int,
                    help='How many training epochs')
parser.add_argument('--pos_iter_per_epoch', default=8, type=int,
                    help='number of training iterations per epoch (pos)')
parser.add_argument('--dir_iter_per_epoch', default=8, type=int,
                    help='number of training iterations per epoch (dir)')
parser.add_argument('--pos_batch_size', default=16, type=int,
                    help='batch size for position training')
parser.add_argument('--dir_batch_size', default=32, type=int,
                    help='batch size for direction training')

# replay buffer
parser.add_argument('--load_replay_buffer', default=None, type=str,
                    help='exp name. Load the replay buffer from another training exp')
parser.add_argument('--replay_buffer_size', default=6400,
                    type=int, help='maximum size of replay buffer')

# policy
parser.add_argument('--position_min_epsilon', default=0.1, type=float,
                    help='(position selection) minimal epsilon in data collection')
parser.add_argument('--position_decay_epoch', default=40, type=int,
                    help='(position selection) how many epoches to decay from 1 to min_epsilon')
parser.add_argument('--position_start_epoch', default=10,
                    type=int, help='(position selection) start epoch of training')

parser.add_argument('--direction_min_epsilon', default=0.1, type=float,
                    help='(direction selection) minimal epsilon in data collection')
parser.add_argument('--direction_decay_epoch', default=80, type=int,
                    help='(direction selection) how many epoches to decay from 1 to min_epsilon')
parser.add_argument('--direction_start_epoch', default=120,
                    type=int, help='(direction selection) start epoch of training')


def main():
    args = parser.parse_args()

    # Set exp directory and tensorboard writer
    writer_dir = os.path.join('exp', args.exp)
    utils.mkdir(writer_dir)
    writer = SummaryWriter(writer_dir)

    # Save arguments
    str_list = []
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
        str_list.append('--{0}={1} \\'.format(key, getattr(args, key)))
    with open(os.path.join('exp', args.exp, 'args.txt'), 'w+') as f:
        f.write('\n'.join(str_list))

    # Set directory. e.g. replay buffer, visualization, model snapshot
    args.replay_buffer_dir = os.path.join('exp', args.exp, 'replay_buffer')
    args.visualization_dir = os.path.join('exp', args.exp, 'visualization')
    utils.mkdir(args.visualization_dir)
    args.model_dir = os.path.join('exp', args.exp, 'models')
    utils.mkdir(args.model_dir)

    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialization of model, optimizer, replay buffer
    model = Model(num_directions=args.num_direction,
                  model_type=args.model_type)
    pos_optimizer = torch.optim.Adam(
        model.pos_model.parameters(), lr=args.pos_learning_rate, betas=(0.9, 0.95))
    dir_optimizer = torch.optim.Adam(
        model.dir_model.parameters(), lr=args.dir_learning_rate, betas=(0.9, 0.95))
    pos_scheduler = torch.optim.lr_scheduler.StepLR(
        pos_optimizer, step_size=args.pos_learning_rate_decay, gamma=0.5)
    dir_scheduler = torch.optim.lr_scheduler.StepLR(
        dir_optimizer, step_size=args.dir_learning_rate_decay, gamma=0.7)
    replay_buffer = ReplayBuffer(
        args.replay_buffer_dir, args.replay_buffer_size)

    # Set device
    device_pos = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_dir = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_pos, device_dir)

    if args.load_replay_buffer is not None:
        print(f'==> Loading replay buffer from {args.load_replay_buffer}')
        replay_buffer.load(os.path.join(
            'exp', args.load_replay_buffer, 'replay_buffer'))
        print(
            f'==> Loaded replay buffer from {args.load_replay_buffer} [size = {replay_buffer.length}]')

    if args.load_checkpoint is not None:
        print(f'==> Loading checkpoint from {args.load_checkpoint}')
        if args.load_checkpoint.endswith('.pth'):
            checkpoint = torch.load(
                args.load_checkpoint, map_location=device_pos)
        else:
            checkpoint = torch.load(os.path.join('pre-trained', 'latest.pth'), map_location=device_pos)
        if 'pos' in args.load_model_type:
            model.pos_model.load_state_dict(checkpoint['pos_state_dict'])
            pos_optimizer.load_state_dict(checkpoint['pos_optimizer'])
            print('==> pos model loaded')
        if 'dir' in args.load_model_type:
            model.dir_model.load_state_dict(checkpoint['dir_state_dict'])
            dir_optimizer.load_state_dict(checkpoint['dir_optimizer'])
            print('==> dir model loaded')
        start_epoch = 0
        del checkpoint
        print(f'==> Loaded checkpoint from {args.load_checkpoint}')
    else:
        start_epoch = 0

    for g in pos_optimizer.param_groups:
        g['lr'] = args.pos_learning_rate
    for g in dir_optimizer.param_groups:
        g['lr'] = args.dir_learning_rate

    # launch processes for each env
    processes, conns = [], []
    ctx = mp.get_context('spawn')
    env_arguments = {
        'action_distance': args.action_distance,
    }
    for rank in range(args.num_envs):
        conn_main, conn_env = ctx.Pipe()
        p = ctx.Process(target=env_process, args=(
            rank, start_epoch + args.seed + rank, conn_env, env_arguments))
        p.daemon = True
        p.start()
        processes.append(p)
        conns.append(conn_main)

    # Initialize exit signal handler (for graceful exits)
    def save_and_exit(signal, frame):
        print('Warning: keyboard interrupt! Cleaning up...')
        for p in processes:
            p.terminate()
        replay_buffer.dump()
        writer.close()
        print('Finished. Now exiting gracefully.')
        sys.exit(0)
    signal.signal(signal.SIGINT, save_and_exit)

    for epoch in range(start_epoch, args.epoch):
        print(f'---------- epoch-{epoch + 1} ----------')
        timestamp = time.time()

        print('==> max_seq_len = ', args.max_seq_len)

        if (epoch + 1) < args.position_start_epoch:
            interval = 1
        else:
            interval = 2

        if (epoch + 1) % interval == 0:
            # Data collection
            data = collect_train_data(
                conns, model, args.model_type,
                num_frames=args.num_frames,
                max_seq_len=args.max_seq_len,
                position_epsilon=args.position_min_epsilon +
                max(0, (1 - (epoch - args.position_start_epoch) /
                    args.position_decay_epoch) * (1 - args.position_min_epsilon)),
                direction_epsilon=args.direction_min_epsilon +
                max(0, (1 - (epoch - args.direction_start_epoch) /
                    args.direction_decay_epoch) * (1 - args.direction_min_epsilon)),
                mode='UCB',
                epoch=epoch,
                board_type='Interact',
                instance_type='train'
            )

            for d in data.values():
                replay_buffer.save_data(d)

            pos_move_list, pos_reward_list = list(), list()
            dir_move_list, dir_reward_list = list(), list()
            pos_recall_list, act_recall_list = list(), list()

            for key, val in data.items():
                if val['type'] == 0:
                    pos_reward_list.append(val['reward'])
                    pos_move_list.append(val['move_flag'])
                    if 'pos_recall' in val.keys():
                        pos_recall_list.append(val['pos_recall'])
                    if 'act_recall' in val.keys():
                        act_recall_list.append(val['act_recall'])
                else:
                    dir_move_list.append(int(val['move_flag']))
                    dir_reward_list.append(abs(val['reward']))

            mean_pos_recall = np.mean(pos_recall_list) if len(
                pos_recall_list) != 0 else 0
            mean_pos_move = np.mean(pos_move_list) if len(
                pos_move_list) != 0 else 0
            mean_pos_reward = np.mean(pos_reward_list) if len(
                pos_reward_list) != 0 else 0
            mean_dir_move = np.mean(dir_move_list) if len(
                dir_move_list) != 0 else 0
            mean_dir_reward = np.mean(dir_reward_list) if len(
                dir_reward_list) != 0 else 0
            mean_act_recall = np.mean(act_recall_list) if len(
                act_recall_list) != 0 else 0

            print(
                f'Train Data Collection. mean_pos_reward = {mean_pos_reward}, mean_dir_move = {mean_dir_move}, mean_dir_reward = {mean_dir_reward}')

            writer.add_scalar('Train/Position precision',
                              mean_pos_reward, epoch + 1)
            writer.add_scalar('Train/Position recall',
                              mean_pos_recall, epoch + 1)
            writer.add_scalar('Train/Direction Accuracy',
                              mean_dir_reward, epoch + 1)
            writer.add_scalar('Train/Action precision',
                              mean_pos_move, epoch + 1)
            writer.add_scalar('Train/Action recall',
                              mean_act_recall, epoch + 1)

            time_data_collection = time.time() - timestamp

            # Replay buffer statistic
            type_data = np.array(replay_buffer.scalar_data['type'])
            move_flag_data = np.array(replay_buffer.scalar_data['move_flag'])
            pos_positive_num = np.sum(np.logical_and(
                type_data == 0, move_flag_data == True))
            pos_negative_num = np.sum(np.logical_and(
                type_data == 0, move_flag_data == False))
            dir_positive_num = np.sum(np.logical_and(
                type_data == 1, move_flag_data == True))
            dir_negative_num = np.sum(np.logical_and(
                type_data == 1, move_flag_data == False))
            print(
                f'Replay buffer size = {len(type_data)}, pos(p+n) = {pos_positive_num}+{pos_negative_num}, dir(p+n) = {dir_positive_num}+{dir_negative_num}')

            writer.add_scalar('Replay Buffer/Position-positive',
                              pos_positive_num, epoch + 1)
            writer.add_scalar('Replay Buffer/Position-negative',
                              pos_negative_num, epoch + 1)
            writer.add_scalar('Replay Buffer/Direction-positive',
                              dir_positive_num, epoch + 1)
            writer.add_scalar('Replay Buffer/Direction-negative',
                              dir_negative_num, epoch + 1)

        # Policy training
        iter_info = list()
        if (epoch + 1) >= args.position_start_epoch: # and (epoch + 1) <= 100:
            iter_info.append(('pos', args.pos_iter_per_epoch))
            # decrease the learning rate
            if (epoch + 1) > 100:
                for g in pos_optimizer.param_groups:
                    g['lr'] = 1e-6
        if (epoch + 1) >= args.direction_start_epoch:
            iter_info.append(('dir', args.dir_iter_per_epoch))

        if len(iter_info) == 0:
            print('skip training')
            continue

        loss_summary = dict()
        for train_model_type, num_iters in iter_info:
            for _ in range(num_iters):
                loss_info = train(model, replay_buffer, pos_optimizer, dir_optimizer, args.pos_batch_size,
                                  args.dir_batch_size, args.model_type, device_pos, device_dir, [train_model_type])
                for k in loss_info:
                    if not k in loss_summary:
                        loss_summary[k] = list()
                    loss_summary[k].append(loss_info[k])
        print_str = 'Training loss: '
        for k in loss_summary:
            loss_avg = np.mean(loss_summary[k])
            print_str += f' {k} = {loss_avg:.4f}'
            writer.add_scalar(f'Policy Training/Loss-{k}', loss_avg, epoch + 1)
        print(print_str)

        # Step scheduler
        pos_scheduler.step()
        dir_scheduler.step()

        if (epoch + 1) % args.snapshot_gap == 0:
            # Save model and optimizer
            save_state = {
                'pos_state_dict': model.pos_model.state_dict(),
                'dir_state_dict': model.dir_model.state_dict(),
                'pos_optimizer': pos_optimizer.state_dict(),
                'dir_optimizer': dir_optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(save_state, os.path.join(args.model_dir, 'latest.pth'))
            shutil.copyfile(
                os.path.join(args.model_dir, 'latest.pth'),
                os.path.join(args.model_dir, 'epoch_%06d.pth' % (epoch + 1))
            )

            # Save replay buffer
            replay_buffer.dump()

        if (epoch + 1) % args.snapshot_gap == 0:
            # Visualization
            for board_type, instance_type in [('Interact', 'train'), ('Interact', 'test')]:
                data = collect_train_data(
                    conns, model, args.model_type,
                    max_seq_len=args.max_seq_len,
                    num_frames=args.num_frames,
                    position_epsilon=0,
                    direction_epsilon=0,
                    mode='UCB',
                    epoch=epoch,
                    board_type=board_type,
                    instance_type=instance_type
                )

                pos_move_list, pos_reward_list = list(), list()
                dir_move_list, dir_reward_list = list(), list()
                pos_recall_list, act_recall_list = list(), list()

                for key, val in data.items():
                    if val['type'] == 0:
                        pos_reward_list.append(val['reward'])
                        pos_move_list.append(val['move_flag'])
                        if 'pos_recall' in val.keys():
                            pos_recall_list.append(val['pos_recall'])
                        if 'act_recall' in val.keys():
                            act_recall_list.append(val['act_recall'])
                    else:
                        dir_move_list.append(int(val['move_flag']))
                        dir_reward_list.append(abs(val['reward']))

                mean_pos_recall = np.mean(pos_recall_list) if len(
                    pos_recall_list) != 0 else 0
                mean_pos_move = np.mean(pos_move_list) if len(
                    pos_move_list) != 0 else 0
                mean_pos_reward = np.mean(pos_reward_list) if len(
                    pos_reward_list) != 0 else 0
                mean_dir_move = np.mean(dir_move_list) if len(
                    dir_move_list) != 0 else 0
                mean_dir_reward = np.mean(dir_reward_list) if len(
                    dir_reward_list) != 0 else 0
                mean_act_recall = np.mean(act_recall_list) if len(
                    act_recall_list) != 0 else 0

                print(
                    f'Evaluation on {instance_type} instances. mean_pos_reward = {mean_pos_reward}, mean_dir_move = {mean_dir_move}, mean_dir_reward = {mean_dir_reward}')

                writer.add_scalar(
                    'Validation-{}/Position precision'.format(instance_type), mean_pos_reward, epoch + 1)
                writer.add_scalar(
                    'Validation-{}/Position recall'.format(instance_type), mean_pos_recall, epoch + 1)
                writer.add_scalar(
                    'Validation-{}/Direction accuracy'.format(instance_type), mean_dir_reward, epoch + 1)
                writer.add_scalar(
                    'Validation-{}/Action precision'.format(instance_type), mean_pos_move, epoch + 1)
                writer.add_scalar(
                    'Validation-{}/Action recall'.format(instance_type), mean_act_recall, epoch + 1)

                vis_path = os.path.join(
                    args.visualization_dir, 'epoch_%06d-board_ins_%s' % (epoch+1, instance_type))
                visualization(data, args.num_envs, args.num_frames, args.max_seq_len,
                              args.num_visualization, vis_path, f'{epoch+1}_{args.exp}')

    save_and_exit(None, None)


def get_position_action(affordance_map, epsilon, depth_image, prev_actions, mode, n_clusters):
    """Get position action based on affordance maps. (remove background if rand() < 0.05)

    Returns:
        action: [w, h]s
        score: float
    """
    if mode == 'UCB':
        N = np.ones_like(affordance_map)
        c = 0.5
        for prev_action in prev_actions:
            N[prev_action[0]-5:prev_action[0]+5,
                prev_action[1]-5:prev_action[1]+5] += 1

        # Get UCB score of all pixels
        affordance_map = affordance_map + c * \
            np.sqrt(2 * np.log(len(prev_actions)+1) / N)

    if mode == 'KMean':
        kmean_threshold = (np.max(affordance_map) +
                           np.min(affordance_map)) * 0.5
        pixels = np.argwhere(affordance_map > kmean_threshold)
        if len(pixels) != 0:
            kmeans = KMeans(n_clusters=n_clusters).fit(pixels)
            # Choose a random cluster
            random_label = np.random.choice(n_clusters)
            selected_pixels = pixels[np.argwhere(
                kmeans.labels_ == random_label).flatten()]
            kmean_mask = np.zeros_like(affordance_map)
            kmean_mask[selected_pixels[:, 0], selected_pixels[:, 1]] = 1
            affordance_map = np.multiply(affordance_map, kmean_mask)

    if np.random.rand() < epsilon or np.max(affordance_map) == 0:
        while True:
            idx = np.random.choice(affordance_map.size)
            action = np.array(np.unravel_index(idx, affordance_map.shape))
            z_value = depth_image[action[0], action[1]]
            if z_value > 0.01 or np.random.rand() < 0.1:
                break
    else:
        idx = np.argmax(affordance_map)

    action = np.array(np.unravel_index(idx, affordance_map.shape))
    action = action.tolist()
    score = affordance_map[action[0], action[1]]

    return action, score


def get_direction_action(affordance_map, epsilon):
    """Get direction action based on affordance maps.

    Returns:
        action: int (index)
        score: float
    """
    if np.random.rand() < epsilon:
        idx = np.random.choice(affordance_map.size)
    else:
        idx = np.argmax(affordance_map)
    action = idx
    score = affordance_map[idx]
    return action, score


def env_process(rank, seed, conn, env_arguments):
    # set random
    np.random.seed(seed)

    env = PybulletSim(gui_enabled=False, **env_arguments)

    while True:
        kwargs = conn.recv()
        if 'message' not in kwargs:
            raise ValueError(f'can not find \'message\'')

        if kwargs['message'] == 'reset':
            observation = env.reset(**kwargs)
            # keep generating the scene until it is valid
            while observation is None:
                observation = env.reset(**kwargs)
            prev_position = list()
            conn.send(observation)
        elif kwargs['message'] == 'step-position':
            affordance_map = kwargs['affordance_map']
            scene_state = env.get_scene_state()
            action, score = get_position_action(affordance_map, kwargs['epsilon'], kwargs['depth_image'],
                                                prev_position, kwargs['mode'], len(scene_state['joint_states'].keys()))

            prev_position.append(action)
            observation, reward, done, info = env.step(
                [0, action[0], action[1]])
            conn.send((action, score, reward, observation,
                      done, info, scene_state))
        elif kwargs['message'] == 'step-direction':
            affordance_map = kwargs['affordance_map']
            epoch = kwargs['epoch']

            if epoch < 100:
                # feed in gt direction, equivalent to exhaustive search over all possible directions
                action, score = [0, 0, 0], 1
            else:
                action_id, score = get_direction_action(
                    affordance_map, kwargs['epsilon'])
                action = kwargs['directions'][action_id]
                print("[selected action] ", action)

            observation, reward, info, (body_id, link_id), actual_action = env.step([
                1, action[0], action[1], action[2]])
            conn.send((actual_action, score, reward,
                      observation, info, (body_id, link_id)))
        else:
            raise ValueError


def collect_train_data(conns, model, model_type, num_frames, max_seq_len, position_epsilon, direction_epsilon, mode, epoch, **kwargs):
    num_envs = len(conns)

    model.eval()
    torch.set_grad_enabled(False)

    data = dict()
    kwargs['message'] = 'reset'
    for conn in conns:
        conn.send(kwargs)
    observations = [conn.recv() for conn in conns]
    done_record = [False for _ in range(num_envs)]
    action_direction = ['both' for _ in range(num_envs)]

    for frame in range(num_frames):
        print("[frame] ", frame + 1)
        # position selection
        position_affordances = model.get_position_affordance(observations)
        for rank in range(num_envs):
            step = 2 * frame
            position_affordance = position_affordances[rank]
            conns[rank].send({
                'message': 'step-position',
                'affordance_map': position_affordance,
                'epsilon': position_epsilon,
                'depth_image': observations[rank]['depth_image'],
                'mode': mode,
            })
            data[(rank, step)] = observations[rank]
            data[(rank, step)]['affordance_map'] = position_affordance

        observations = list()
        positions = list()
        for rank in range(num_envs):
            step = 2 * frame
            (action, score, (reward, move_flag), observation,
             done, info, scene_state) = conns[rank].recv()
            observations.append(observation)
            positions.append(action)
            data[(rank, step)]['type'] = 0
            data[(rank, step)]['action'] = action
            data[(rank, step)]['pos_action'] = action
            data[(rank, step)]['score'] = score
            data[(rank, step)]['reward'] = reward
            data[(rank, step)]['move_flag'] = move_flag
            data[(rank, step)]['next_image'] = observation['image']
            data[(rank, step)]['segmentation_mask'] = observation['segmentation_mask']
            data[(rank, step)]['scene_state'] = scene_state
            data[(rank, step)]['action_link'] = (
                info['body_id'], info['link_id']) if reward == 1 else -1
            done_record[rank] = done

        print("Done position selection")

        # direction selection
        direction_affordance_maps, directions, position_masks = model.get_direction_affordance(
            observations, positions, model_type)
        for rank in range(num_envs):
            if done_record[rank]:
                continue
            step = 2 * frame + max_seq_len
            direction_affordance = direction_affordance_maps[rank]
            position_mask = position_masks[rank]
            conns[rank].send({
                'message': 'step-direction',
                'affordance_map': direction_affordance,
                'epsilon': direction_epsilon,
                'action_direction': action_direction[rank],
                'directions': directions[rank],
                'epoch': epoch
            })
            data[(rank, step)] = observations[rank]
            data[(rank, step)]['affordance_map'] = direction_affordance
            data[(rank, step)]['position_mask'] = position_mask
            data[(rank, step)]['directions'] = directions[rank]

        dir_observations = list()
        for rank in range(num_envs):
            if done_record[rank]:
                dir_observations.append(observations[rank])
                continue
            step = 2 * frame + max_seq_len
            (action, score, (reward, move_flag), observation, info,
             (action_body_id, action_link_id)) = conns[rank].recv()

            dir_observations.append(observation)
            data[(rank, step)]['type'] = 1
            data[(rank, step)]['action'] = action
            data[(rank, step)]['pos_action'] = data[(
                rank, 2*frame)]['pos_action']
            data[(rank, step)]['score'] = score
            data[(rank, step)]['reward'] = reward
            data[(rank, step)]['move_flag'] = move_flag
            data[(rank, step)]['next_image'] = observation['image']
            data[(rank, step)]['action_direction'] = action_direction[rank]
            data[(rank, step)]['action_link'] = (
                action_body_id, action_link_id)
            for k in info:
                data[(rank, step)][k] = info[k]

            if move_flag:
                data[(rank, step-1)]['move_flag'] = True

        print("Done direction selection")
        observations = dir_observations

    # Compute position recall and action recall for each board env
    for rank in range(num_envs):
        scene_state = data[(rank, 0)]['scene_state']
        num_movable_joints = len(scene_state['joint_states'].keys())
        successful_actions = set()
        successful_positions = set()
        for frame in range(num_frames):
            if data[(rank, 2 * frame)]['action_link'] != -1:
                successful_positions.add(
                    data[(rank, 2 * frame)]['action_link'])
                if data[(rank, 2 * frame + max_seq_len)]['move_flag']:
                    action_body_id, action_link_id = data[(
                        rank, 2 * frame + max_seq_len)]['action_link']
                    successful_actions.add((action_body_id, action_link_id))

        position_recall = len(successful_positions) / num_movable_joints
        action_recall = len(successful_actions) / num_movable_joints
        data[(rank, 0)]['pos_recall'] = position_recall
        data[(rank, 0)]['act_recall'] = action_recall

    return data


def train(model, replay_buffer, pos_optimizer, dir_optimizer, pos_batch_size, dir_batch_size, model_type, device_pos, device_dir, train_model_type):
    type_data = np.array(replay_buffer.scalar_data['type'])
    move_flag_data = np.array(replay_buffer.scalar_data['move_flag'])

    # add data randomly
    sample_inds = dict()
    if 'pos' in train_model_type:
        sample_inds['position'] = {
            'index': np.argwhere(type_data == 0)[:, 0],
            'positive_index': np.argwhere(np.logical_and(type_data == 0, move_flag_data == True))[:, 0],
            'negative_index': np.argwhere(np.logical_and(type_data == 0, move_flag_data == False))[:, 0],
            'iter': 1,
        }
    if 'dir' in train_model_type:
        sample_inds['direction'] = {
            'index': np.argwhere(type_data == 1)[:, 0],
            'positive_index': np.argwhere(np.logical_and(type_data == 1, move_flag_data == True))[:, 0],
            'static_index': np.argwhere(np.logical_and(type_data == 1, move_flag_data == False))[:, 0],
            'iter': 3,
        }
    loss_dict = {'pos': [], 'sgn': [], 'mag': []}
    for sample_type, sample_info in sample_inds.items():
        if len(sample_info['index']) == 0:
            print('[Warning] Data is not balanced')
            continue
        for _ in range(sample_info['iter']):
            if sample_type == 'position':
                replay_iter = list()
                replay_iter.append(np.random.choice(
                    sample_info['positive_index'],
                    min(len(sample_info['positive_index']),
                        pos_batch_size // 2),
                    replace=False
                ))
                replay_iter.append(np.random.choice(
                    sample_info['negative_index'],
                    min(len(sample_info['negative_index']),
                        pos_batch_size // 2),
                    replace=False
                ))
                replay_iter = np.concatenate(replay_iter, 0)
            else:
                replay_iter = list()
                replay_iter.append(np.random.choice(
                    sample_info['positive_index'],
                    min(len(sample_info['positive_index']),
                        dir_batch_size // 2),
                    replace=False
                ))
                replay_iter.append(np.random.choice(
                    sample_info['static_index'],
                    min(len(sample_info['static_index']), dir_batch_size // 2),
                    replace=False
                ))
                replay_iter = np.concatenate(replay_iter, 0)

            # fetch data from replay buffer
            observations, scalars = replay_buffer.fetch_data(replay_iter)
            actions = scalars['action']

            model.train()
            torch.set_grad_enabled(True)
            if sample_type == 'position':
                output_tensor = model.get_position_affordance(
                    observations, torch_tensor=True)
                # Compute loss and gradients
                pos_optimizer.zero_grad()
                criterion = nn.CrossEntropyLoss()
                loss = criterion(
                    output_tensor[np.arange(
                        actions.shape[0]), :, actions[:, 0], actions[:, 1]],
                    torch.from_numpy(
                        np.array(scalars['move_flag'], dtype=int)).to(device_pos)
                )
                loss.backward()
                pos_optimizer.step()
                loss_dict['pos'].append(loss.item())
            elif sample_type == 'direction':
                positions = scalars['pos_action']
                mag_output, _ = model.get_direction_affordance(
                    observations, positions, model_type, torch_tensor=True, directions=actions)
                mag_target = scalars['reward']  # scalars['move_flag']

                # Compute loss and gradients
                dir_optimizer.zero_grad()
                loss = 0
                if 'mag' in model_type:
                    criterion = nn.BCEWithLogitsLoss(reduction='sum')
                    loss_mag = criterion(
                        mag_output[:, 0],
                        torch.from_numpy(mag_target.astype(
                            np.float32)).to(device_dir)
                    )
                    loss += loss_mag
                    loss_dict['mag'].append(loss_mag.item())

                loss.backward()
                dir_optimizer.step()

    loss_info = {}
    for k in loss_dict:
        if len(loss_dict[k]) > 0:
            loss_info[k] = np.mean(loss_dict[k])

    return loss_info


def visualization(vis_data, num_envs, num_frames, max_seq_len, num_visualization, vis_path, title='visualization'):
    num_visualization = num_envs if num_visualization is None else min(
        num_visualization, num_envs)
    data = {}
    ids = list()
    cols = ['compare', 'color_image', 'next_image',
            'affordance', 'position-mask', 'pred', 'info']
    for rank in range(num_visualization):
        for frame in range(num_frames):
            for step in range(2*frame, max_seq_len + 2*frame + 1):
                if (rank, step) in vis_data:
                    ids.append(f'{rank}_{step}')

    for (rank, step), sample_data in vis_data.items():
        if rank >= num_visualization:
            continue
        color_image = sample_data['image'][:, :, :3]
        next_color_image = sample_data['next_image'][:, :, :3]
        data[f'{rank}_{step}_color_image'] = color_image
        data[f'{rank}_{step}_next_image'] = next_color_image
        action = sample_data['action']
        affordance_map = sample_data['affordance_map']

        if sample_data['type'] == 0:
            data[f'{rank}_{step}_pred'] = [f"score: {sample_data['score']:.3f}",
                                           f"reward: {sample_data['move_flag']}, {sample_data['reward']:.3f}"]
            data[f'{rank}_{step}_info'] = [f"action: {action}"]
            data[f'{rank}_{step}_compare'] = color_image

            affordance_map -= np.min(affordance_map)
            affordance_map /= np.max(affordance_map)
            cmap = plt.get_cmap('jet')
            affordance_map = cmap(affordance_map)[..., :3]
            data[f'{rank}_{step}_affordance'] = affordance_map * \
                0.8 + color_image * 0.2
            data[f'{rank}_{step}_position-mask'] = np.zeros_like(
                affordance_map)
        else:
            data[f'{rank}_{step}_pred'] = [f"action_dir: {sample_data['action_direction']}",
                                           f"score: {sample_data['score']:.3f}", f"reward: {sample_data['move_flag']}, {sample_data['reward']:.3f}"]
            data[f'{rank}_{step}_info'] = [
                f"action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"]
            compare_image = color_image / 2 + next_color_image / 2
            compare_image = utils.draw_action(
                image=compare_image,
                position_start=sample_data['position_start'],
                position_end=sample_data['position_end'],
                cam_intrinsics=sample_data['cam_intrinsics'],
                cam_view_matrix=sample_data['cam_view_matrix'],
                thickness=2, tipLength=0.8, color=(0, 255, 0)
            )
            data[f'{rank}_{step}_compare'] = compare_image

            # affordance_map /= np.max(np.abs(affordance_map))
            affordance_map /= np.max(affordance_map)
            # affordance_map = (affordance_map + 1) / 2
            cmap = plt.get_cmap('jet')
            affordance_map = cmap(affordance_map)[..., :3]
            affordance_image = color_image.copy()
            affordance_map = (
                affordance_map * 255).astype(np.uint8).astype(np.float)
            num_direction = len(sample_data['directions'])
            for direction_id in range(num_direction):
                affordance_image = utils.draw_action(
                    image=affordance_image,
                    position_start=sample_data['position_start'],
                    position_end=sample_data['position_start'] +
                    sample_data['directions'][direction_id] * 0.4,
                    cam_intrinsics=sample_data['cam_intrinsics'],
                    cam_view_matrix=sample_data['cam_view_matrix'],
                    thickness=2,
                    tipLength=0.1,
                    color=tuple(affordance_map[direction_id])
                )
            data[f'{rank}_{step}_affordance'] = affordance_image
            position_mask = sample_data['position_mask']
            data[f'{rank}_{step}_position-mask'] = position_mask

    utils.html_visualize(vis_path, data, ids, cols, title=title)


if __name__ == '__main__':
    main()
    