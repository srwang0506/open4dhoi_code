import json

import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import scipy
from scipy.spatial.transform import Rotation as R
from rembg import remove
import open3d as o3d
# import torchvision.transforms as transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def reconstruct3D_from_depth(pred_depth):
    # cam_u0 = rgb.shape[1] / 2.0
    # cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5
    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax
    depth_scale_1 = pred_depth_norm
    return depth_scale_1


def reconstruct_3D(masks, depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        # print('Infinit focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
        # print(depth.max())
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth
    z[masks == 0] = -1
    pcd_new = []
    x = np.reshape(x, (width * height, 1)).astype(np.float32)
    y = np.reshape(y, (width * height, 1)).astype(np.float32)
    z = np.reshape(z, (width * height, 1)).astype(np.float32)
    # pcd = np.concatenate((x, y, z), axis=1)
    pcd = np.concatenate((x, y, z), axis=1)
    # print(pcd.shape)
    index = np.asarray(pcd[:, 2] >= 0)
    # Filter out rows where the z value is negative
    pcd_new = pcd[pcd[:, 2] >= 0]

    pcd_new[:,2]*=-1

    # index=index.reshape(width,height)

    # pcd_new already has the shape (-1, 3), no need to reshape

    return pcd_new, index


def reconstruct_3D_lucid(masks, depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    W = depth.shape[1]
    H = depth.shape[0]
    focal = (1.8269e+02, 1.8269e+02)
    fov = (2 * np.arctan(W / (2 * focal[0])), 2 * np.arctan(H / (2 * focal[1])))
    K = np.array([
        [focal[0], 0., W / 2],
        [0., focal[1], H / 2],
        [0., 0., 1.],
    ]).astype(np.float32)
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pts_coord_cam = np.matmul(np.linalg.inv(K),
                              np.stack((x * depth, y * depth, 1 * depth), axis=0).reshape(3, -1)).transpose(1, 0)
    # new_pts_colors2 = (np.array(image_curr).reshape(-1, 3).astype(np.float32) / 255.)  ## new_pts_colors2
    z[masks==0]=-1
    # pcd_new=[]
    # x = np.reshape(x, (width * height, 1)).astype(np.float32)
    # y = np.reshape(y, (width * height, 1)).astype(np.float32)
    # z = np.reshape(z, (width * height, 1)).astype(np.float32)
    # # pcd = np.concatenate((x, y, z), axis=1)
    # pcd = np.concatenate((x, y, z), axis=1)
    # # print(pcd.shape)
    index = np.asarray(pcd[:, 2] >= 0)
    # Filter out rows where the z value is negative
    pcd_new = pcd[pcd[:, 2] >= 0]

    # index=index.reshape(width,height)

    # pcd_new already has the shape (-1, 3), no need to reshape
    index = masks.reshape(-1)
    pcd_new = pts_coord_cam[index == 1]

    return pcd_new, index == 1


def reconstruct_depth(masks, depth, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    depth = np.squeeze(depth)
    # print(depth.shape)
    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd, index = reconstruct_3D(masks, depth, f=focal)
    # pcd,index = reconstruct_3D_lucid(masks,depth, f=focal)
    # save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))
    # print(pcd.shape)
    return pcd, index


def get_obj_pcd(masks, depth):
    pred_depth = depth
    # pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # recover focal length, shift, and scale-invariant depth
    depth_scaleinv = reconstruct3D_from_depth(pred_depth)
    # mask_obj_all=np.zeros((masks.shape[-2],masks.shape[-1]), dtype=bool)
    # # for mask in masks:
    # # for obj in masks:
    # mask_obj_all[]=True
    pcd, index = reconstruct_depth(masks, depth_scaleinv, focal=1e6)
    # findex=get_front_index(pcd)
    # index[~findex]=False
    # pcd=pcd[findex]
    return pcd, index

def project(xyz, K):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    # xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]

    return xy
    
    
def get_scene_pcd(depth):
    pred_depth = depth
    # pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
    # recover focal length, shift, and scale-invariant depth
    depth_scaleinv = reconstruct3D_from_depth(pred_depth)

    mask_obj_all = np.ones((depth.shape[0], depth.shape[1]), dtype=bool)
    pcd, index = reconstruct_depth(mask_obj_all, depth_scaleinv, focal=1e6)
    return pcd, index


def point_align_vis(body, h, w, K):


    vert = np.array(body)

    # homog_coord = np.ones(list(vert.shape[:-1]) + [1])
    # points_h = np.concatenate([vert, homog_coord], axis=-1)
    # projected = points_h

    img_point=project(vert,np.asarray(K))

    img_point = img_point.astype(np.int32)

    # if h>w:
    #     img_point[:, 0] = img_point[:, 0]- (h-w)/2
    # else:
    #     img_point[:, 1] = img_point[:, 1]- (w-h)/2

    # find verts projected on the image
    flag1 = np.logical_and(img_point[:, 0] < w, img_point[:, 0] >= 0)
    flag2 = np.logical_and(img_point[:, 1] < h, img_point[:, 1] >= 0)
    filter = np.where(np.logical_and(flag1, flag2))[0]
    rest = vert[filter, :] #vert是human的点
    img_point = img_point[filter, :]
    index = np.argsort(rest[:, 2])  # sort by depth
    align = img_point[:, 1] * w + img_point[:, 0]  # pixel corresponding
    u_all = align
    u, indices = np.unique(align[index], return_index=True)  # find the first appearance（按照深度，实际上不做这一步也没有影响，公式可以不体现）
    # print('indices', indices.shape)
    front = rest[index[indices], :]
    pid = filter[index[indices]]
    l = len(front[:, 2])
    final = np.argsort(front[:, 2]) #取了前一半的点，认为这些点是人的前半身
    return u[final], front[final], pid[
        final]  # pixel index, corresponding point location obtained from human, corresponding pixel index


def get_front_index(points):
    fz = np.percentile(points[:, 2], 80)
    mean_z = np.mean(fz)
    # print(mean_z)
    # mean_z=2300
    return np.where(points[:, 2] < mean_z)[0]


def align(scene, body, h, w, K):


    # avgz = np.mean(body[:, 2])
    #
    # body[:, :2] += np.array([[avgz * center[0] / focal_len, avgz * center[1] / focal_len]])
    num_vert = body.shape[0]

    pidx, front_h, vidx = point_align_vis(body, h, w, K)


    front_s = scene[pidx, :]
    # f_index = get_front_index(front_s)
    # print(len(f_index))
    # front_s= front_s[f_index, :]

    # front_s = scene_vertices
    # find scale
    # cur_sel = np.where(np.logical_and(vidx >= 0, vidx < 1 * num_vert))[0]
    b = front_h
    s = front_s
    dis_b = np.mean(scipy.spatial.distance.cdist(b, b))
    dis_s = np.mean(scipy.spatial.distance.cdist(s, s))
    scale = dis_s / dis_b
    b *= scale
    displace = np.mean(s, axis=0) - np.mean(b, axis=0)
    b += displace
    return scale, displace, front_s, b, pidx


def load_transformation_matrix(t_dir):
    T=json.load(open(t_dir+'transform.json'))
    T = np.array(T)
    rotate=json.load(open(t_dir+'rotate90.json'))
    Rx, Ry, Rz = rotate

    return T, Rx, Ry, Rz


def replace_depth(depth,bbox, inp_depth,inp_img):
    inp_img = np.asarray(inp_img)
    inp_img = cv2.resize(inp_img, (bbox[2] - bbox[0], bbox[3] - bbox[1]), interpolation=cv2.INTER_NEAREST)
    inp_depth = cv2.resize(inp_depth, (bbox[2] - bbox[0], bbox[3] - bbox[1]), interpolation=cv2.INTER_NEAREST)
    mask = remove(inp_img, only_mask=True)
    mask = mask.astype(bool)

    left, top, right, bottom = bbox

    # depth = src_depth[0]

    target_region = depth[top:bottom, left:right]

    # print(target_region.shape, inp_depth.shape, mask.shape)
    mean_depth1 = np.mean(inp_depth[mask])
    mean_depth2 = np.mean(target_region[mask])

    target_region[mask] = inp_depth[mask] - mean_depth1 + mean_depth2

    return depth
scene = o3d.t.geometry.RaycastingScene()
light_dir = np.array([0, 0, -1], dtype=float)
light_dir /= np.linalg.norm(light_dir)
def get_front_points(vertices,mesh):
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    visible_idx = []
    for i, v in enumerate(vertices):
        # 射线起点稍微偏离表面一点，避免自穿透
        orig = v + light_dir * 1e-6
        # 射线方向同光线方向
        ans = scene.cast_rays(
            o3d.core.Tensor([[*orig, *light_dir]], dtype=o3d.core.Dtype.Float32)
        )
        # 如果打到的第一个三角形与当前顶点所属三角形不是同一个（即没自遮挡），则可见
        if ans['geometry_ids'].numpy()[0] != mesh_id:
            visible_idx.append(i)
    return vertices[visible_idx]

# def process_depth2square(frame):
#     # 获取帧的尺寸
#     (h, w) = frame.shape[:2]
#
#     # 创建一个白色背景的正方形画布
#     square_size = max(w, h)
#     square = np.zeros((square_size,square_size), dtype=np.float32)
#
#     # 将原图放置在正方形画布的中心
#     start_y = (square_size - h) // 2
#     start_x = (square_size - w) // 2
#     square[start_y:start_y + h, start_x:start_x + w] = frame
#
#     return square
# def process_frame2square(frame):
#     # 获取帧的尺寸
#     (h, w) = frame.shape[:2]
#
#     # 创建一个白色背景的正方形画布
#     square_size = max(w, h)
#     square = np.ones((square_size,square_size, 3), dtype=np.uint8) * 255
#
#     # 将原图放置在正方形画布的中心
#     start_y = (square_size - h) // 2
#     start_x = (square_size - w) // 2
#     square[start_y:start_y + h, start_x:start_x + w] = frame
#
#     return square

def compute_global_rotation(pose_axis_anges, joint_idx):
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation
M = np.diag([-1, 1, 1])
def update_hand_pose(hand_poses,global_orient,body_params,frame_idx):

    body_pose = body_params[frame_idx].detach().cpu().numpy().reshape(1, -1)
    global_orient = global_orient[frame_idx].detach().cpu().numpy().reshape(1, 3)
    try:
        handpose=hand_poses[str(frame_idx)]
    except:
        return torch.from_numpy(body_pose), np.zeros(45), np.zeros(45)
    full_body_pose = np.concatenate(
        [global_orient.reshape(1, 3), body_pose.reshape(21, 3)], axis=0)
    left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)
    right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)

    if 'left_hand' in handpose:
        global_orient_hand_left = np.asarray(handpose["left_global_orient"]).reshape(3, 3)
        left_wrist_global_rot = M @ global_orient_hand_left @ M  # mirror switch
        left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
        left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()
        body_pose[:, 57:60] = left_wrist_pose_vec
    if 'right_hand' in handpose:
        global_orient_hand_right = np.asarray(handpose["right_global_orient"]).reshape(3, 3)  
        right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ global_orient_hand_right
        right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()
        body_pose[:, 60:63] = right_wrist_pose_vec


    left_hand_pose = np.zeros(45)
    right_hand_pose = np.zeros(45)
    for i in range(15):
        if 'left_hand' in handpose:
            left_finger_pose = M @ np.asarray(hand_poses[str(frame_idx)]["left_hand"])[
                i] @ M
            left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
            left_hand_pose[i * 3: i * 3 + 3] = left_finger_pose_vec
        if 'right_hand' in handpose:
            right_finger_pose = np.asarray(hand_poses[str(frame_idx)]["right_hand"][i])
            right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
            right_hand_pose[i * 3: i * 3 + 3] = right_finger_pose_vec

    return torch.from_numpy(body_pose), left_hand_pose, right_hand_pose