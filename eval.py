import torch
import argparse
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T

from copy import deepcopy
from easydict import EasyDict as edict

from policy import FoAR
from eval_agent import Agent
from utils.constants import *
from dataset.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform


default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "crop_in_base": False,
    "num_action": 20,
    "num_inference_step": 20,
    "num_obs_force": 100,
    "num_motion_calc_steps": 5,
    "cls_threshold": 0.9,
    "force_threshold": 8.0,
    "torque_threshold": 5.0,
    "epsilon": 0.006,
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
    "vis": False,
    "discretize_rotation": False,
    "ensemble_mode": "new"
})

def make_policy(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = FoAR(
        num_action = args.num_action,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout,
        num_obs_force = args.num_obs_force
    ).to(device)
    return policy


def mask_point_cloud(points, colors, projector=None, cam_id="750612070851", crop_in_base=True):
    if not crop_in_base:
        x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
    else:
        points = projector.project_point_to_base_coord(points, cam = cam_id)
        x_mask = ((points[:, 0] >= WORKSPACE_BASE_MIN[0]) & (points[:, 0] <= WORKSPACE_BASE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_BASE_MIN[1]) & (points[:, 1] <= WORKSPACE_BASE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_BASE_MIN[2]) & (points[:, 2] <= WORKSPACE_BASE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
        points = projector.project_point_to_camera_coord(points, cam = cam_id)

    return points, colors


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005, projector = None, crop_in_base=True):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)
    # mask point cloud
    points, colors = mask_point_cloud(points, colors, projector = projector, crop_in_base = crop_in_base)
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, voxel_size = 0.005, projector = None, crop_in_base=True):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size, projector=projector, crop_in_base=crop_in_base)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def _normalize_force(force_list):
    ''' force_list: [T, 6]'''
    force_list = (force_list - FORCE_MIN) / (FORCE_MAX - FORCE_MIN) * 2 - 1
    return force_list

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading policy ...")
    force_policy = make_policy(args)

    n_parameters = sum(p.numel() for p in force_policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    force_policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # evaluation
    agent = Agent(
        robot_ip = "192.168.2.100",
        pc_ip = "192.168.2.35",
        gripper_port = "/dev/ttyUSB0",
        camera_serial = "750612070851",
        num_obs_force= args.num_obs_force
    )
    projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)
    force_ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)
    
    img_process = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias = True),
            T.Normalize(mean = IMG_MEAN, std = IMG_STD)
        ])

    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    with torch.inference_mode():
        force_policy.eval()
        prev_width = None
        for t in range(args.max_steps):
            if t % args.num_inference_step == 0 :
                # pre-process inputs
                colors, depths = agent.get_observation()
                coords, feats, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics = agent.intrinsics,
                    voxel_size = args.voxel_size,
                    projector = projector,
                    crop_in_base = args.crop_in_base
                )
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)
                tcp = agent.get_tcp_pose()
                force_torque_base = agent.get_force_torque_history()
                force_torque_cam = []
                for i in range(args.num_obs_force):
                    force_torque_cam.append(projector.project_force_to_camera_coord(tcp, force_torque_base[i], cam="750612070851"))
                force_torque_cam = np.array(force_torque_cam)
                force_torque_normalized = _normalize_force(force_torque_cam.copy())
                force_torque_normalized = torch.from_numpy(force_torque_normalized).float()
                force_torque_normalized = force_torque_normalized.unsqueeze(0).to(device)
                color_list = img_process(colors).unsqueeze(0).to(device)
                # FoAR
                prop, pred_raw_action = force_policy(force_torque_normalized, color_list, cloud_data, actions = None, contact=None, batch_size = 1)
                pred_raw_action = pred_raw_action.squeeze(0).cpu().numpy()

                # unnormalize predicted actions
                action = unnormalize_action(pred_raw_action)
                # visualization
                if args.vis:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                    o3d.visualization.draw_geometries([pcd, *tcp_vis_list])
                # project action to base coordinate
                action_tcp = projector.project_tcp_to_base_coord(action[..., :-1], cam = agent.camera_serial, rotation_rep = "rotation_6d")
                action_width = action[..., -1]
                # safety insurance
                action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                # full actions
                action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis = -1)
                # add to ensemble buffer
                if prop < args.cls_threshold:
                    ensemble_buffer.add_action(action, t)
                else:
                    cur_force_value, cur_torque_value = agent.get_force_torque_value()
                    if cur_force_value < args.force_threshold and cur_torque_value < args.torque_threshold: # reactive control
                        distance = np.mean(action[:args.num_motion_calc_steps, :3], axis=0) - agent.get_tcp_pose()[:3]
                        unit_distance = distance / np.linalg.norm(distance)
                        # action[:, :3] = agent.get_tcp_pose()[:3] + unit_distance * args.epsilon
                        action[:, :3] = action[:, :3] + unit_distance * args.epsilon
                        force_ensemble_buffer.add_action(action, t)
            
            if prop < args.cls_threshold:
                step_action = ensemble_buffer.get_action()
                force_ensemble_buffer.get_action()
            else: 
                ensemble_buffer.get_action()
                step_action = force_ensemble_buffer.get_action()
                
            if step_action is None:   # no action in the buffer => no movement.
                continue
            
            step_tcp = step_action[:-1]
            step_width = step_action[-1]

            # send tcp pose to robot
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                last_rot = step_tcp[3:]
                for rot in rot_steps:
                    step_tcp[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp, 
                        rotation_rep = "rotation_6d",
                        blocking = True
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp,
                    rotation_rep = "rotation_6d",
                    blocking = True
                )
            
            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                agent.set_gripper_width(step_width, blocking = True)
                prev_width = step_width
    
    agent.stop()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'policy checkpoint path', required = True)
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--crop_in_base', action = 'store_true', help = 'whether to crop point cloud in base coordinate')
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--num_obs_force', action = 'store', type = int, help = 'width of force window', required = False, default = 100)
    parser.add_argument('--num_motion_calc_steps', action = 'store', type = int, help = 'number of motion calculation steps', required = False, default = 5)
    parser.add_argument('--cls_threshold', action = 'store', type = float, help = 'threshold for future contact probability', required = False, default = 0.9)
    parser.add_argument('--force_threshold', action = 'store', type = float, help = 'force threshold', required = False, default = 8.0)
    parser.add_argument('--torque_threshold', action = 'store', type = float, help = 'torque threshold', required = False, default = 5.0)
    parser.add_argument('--epsilon', action = 'store', type = int, help = 'epsilon for reactive control', required = False, default = 0.006)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_timesteps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    evaluate(vars(parser.parse_args()))