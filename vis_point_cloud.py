import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist
from PIL import Image

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.realworld import RealWorldDataset
from policy import RISE, ForceRISE5
# from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform


default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "num_action": 20,
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": True,
    "discretize_rotation": True,
    "ensemble_mode": "new"
})

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def unnormalize_offset_action(offset_action):
    trans_min = np.array([-0.15, -0.15, -0.10])
    trans_max = np.array([0.15, 0.15, 0.10])
    max_gripper_width = 0.11 # meter
    offset_action[..., :3] = (offset_action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    offset_action[..., -1] = (offset_action[..., -1] + 1) / 2.0 * max_gripper_width
    return offset_action

def eval(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # policy
    print("Loading RISE policy ...")
    policy = RISE(
        num_action = args.num_action,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout
    ).to(device)

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # offset policy
    print("Loading ForceRISE policy ...")
    if args.policy == 'ForceRISE5':
        force_policy = ForceRISE5(
            num_action = args.num_action,
            input_dim = 6,
            obs_feature_dim = args.obs_feature_dim,
            action_dim = 10,
            hidden_dim = args.hidden_dim,
            nheads = args.nheads,
            num_encoder_layers = args.num_encoder_layers,
            num_decoder_layers = args.num_decoder_layers,
            dropout = args.dropout,
            num_obs_force= args.num_obs_force
        ).to(device)
    else:
        raise NotImplementedError("Policy {} not implemented.".format(args.policy))

    # load offset checkpoint
    assert args.offset_ckpt is not None, "Please provide the offset checkpoint to evaluate."
    force_policy.load_state_dict(torch.load(args.offset_ckpt, map_location = device), strict = False)
    print("Offset checkpoint {} loaded.".format(args.offset_ckpt))

    # projector = Projector(os.path.join(args.calib, str(calib_timestamp)))
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)

    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'val',
        num_obs = 1,
        num_obs_force= args.num_obs_force,
        num_action = args.num_action,
        voxel_size = args.voxel_size,
        with_cloud = True,
    )
    print(len(dataset))
    with torch.inference_mode():
        policy.eval()
        force_policy.eval()
        actions = []
        start_step = 100
        for i in range(start_step, start_step+1):
            ret_dict = dataset[i]
            if i % args.num_inference_step == 0:
                feats = torch.tensor(ret_dict['input_feats_list'][0])
                coords = torch.tensor(ret_dict['input_coords_list'][0])
                cloud = ret_dict['clouds_list'][0]
                feats, coords = feats.to(device), coords.to(device)
                coords, feats = create_batch(coords, feats)
                cloud_data = ME.SparseTensor(feats, coords)
                pred_raw_action = policy(cloud_data, actions = None, batch_size = 1).squeeze(0).cpu().numpy()
                rise_action = unnormalize_action(pred_raw_action) # cam coordinate 
                # load offset action
                force_torque = ret_dict['input_force_list'].unsqueeze(0)
                force_torque = force_torque.to(device)
                color_list = ret_dict['input_frame_list_normalized'].unsqueeze(0)
                color_list = color_list.to(device)
                # forcerise action
                if args.policy == 'ForceRISE5':
                    prop, pred_raw_force_action = force_policy(force_torque, color_list, cloud_data, actions = None, batch_size = 1)
                    pred_raw_force_action = pred_raw_force_action.squeeze(0).cpu().numpy()
                    print(prop, ret_dict['is_cut'])
                else:
                    raise NotImplementedError("Policy {} not implemented.".format(args.policy))
                force_action = unnormalize_action(pred_raw_force_action)
                force_action[:, 2] -= 0.02
                # final action
                gt_action = ret_dict['action'].squeeze(0).cpu().numpy()

                if args.vis:
                    print("Show cloud ...")
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in force_action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                        tcp_vis.paint_uniform_color([1.0, 165.0/255.0, 0.0])  # set color to green
                        tcp_vis_list.append(tcp_vis)
                    # tcp_vis_rise_list = []
                    # for raw_tcp in rise_action:
                    #     tcp_vis_rise = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                    #     tcp_vis_rise.paint_uniform_color([1.0, 0.0, 0.0]) # set color to red
                    #     tcp_vis_rise_list.append(tcp_vis_rise) 
                    # tcp_vis_gt_list = []
                    # for raw_tcp in gt_action:
                    #     tcp_vis_gt = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                    #     tcp_vis_gt.paint_uniform_color([0.0, 0.0, 1.0]) # set color to blue
                    #     tcp_vis_gt_list.append(tcp_vis_gt)

                    # o3d.visualization.draw_geometries([pcd, *tcp_vis_list, *tcp_vis_rise_list, *tcp_vis_gt_list])
                    # o3d.visualization.draw_geometries([pcd, *tcp_vis_list])
                    # write to ply file
                    # o3d.io.write_point_cloud("pcd/cloud_{}_correction.ply".format(i), pcd)
                    combined_mesh = o3d.geometry.TriangleMesh() 
                    for tcp_vis in tcp_vis_list:
                        combined_mesh += tcp_vis
                    o3d.io.write_triangle_mesh("pcd/action_{}.ply".format(i), combined_mesh)
                    # combined_mesh = o3d.geometry.TriangleMesh()
                    # combined_mesh += pcd_mesh
                    

                # ensemble_buffer.add_action(force_action, i)

                # if prop < 0.9:
                #     ensemble_buffer.add_action(force_action, t)
                # else:
                #     if np.max(abs(agent.force())) < 8: # force is smaller than the force threshold
                #         # action[:, 2] = action[:, 2] - 0.01
                #         if np.mean(force_action[:5, 2]) < agent.get_tcp_pose()[2]:
                #             force_action[:, 2] = force_action[:, 2] - 0.01
                #         else:
                #             force_action[:, 2] = force_action[:, 2] + 0.01                        
                #         force_ensemble_buffer.add_action(force_action, t)

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                continue
            actions.append(step_action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--offset_ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--policy', action = 'store', type = str, help='type of policy', required=True)
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--num_obs_force', action = 'store', type = int, help = 'number of force observation steps', required = False, default = 100)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--force_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 64)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    eval(vars(parser.parse_args()))