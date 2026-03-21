import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # specify GPU id
import glob
import json
import numpy as np
import torch
import smplx
import argparse
from PIL import Image
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
# renderer & utilities from project

import trimesh
from video_optimizer.utils.parameter_transform import apply_transform_to_smpl_params
from pytorch3d.transforms import axis_angle_to_matrix

# Summary: read combined all_parameters_*.json and an object .obj, build SMPL human meshes,
# apply transform_to_global (using saved incam/global subsets), transform object by saved R/T,
# render frames with Renderer and write out a video.

def find_latest_file(dirpath, pattern):
    files = glob.glob(os.path.join(dirpath, pattern))
    return max(files, key=os.path.getmtime) if files else None

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def ensure_cuda(t):
    return t.cuda() if torch.cuda.is_available() else t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='folder containing final_optimized_parameters')
    parser.add_argument('--smpl_model', default='video_optimizer/smpl_models/SMPLX_NEUTRAL.npz')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--out', default='output_render.mp4')
    args = parser.parse_args()

    fop_dir = os.path.join(args.data_dir, 'final_optimized_parameters')
    if not os.path.isdir(fop_dir):
        raise FileNotFoundError(f"Not found: {fop_dir}")

    combined_json = find_latest_file(fop_dir, 'all_parameters_*.json') or find_latest_file(fop_dir, 'transformed_parameters_*.json')
    if not combined_json:
        raise FileNotFoundError("No combined or transformed JSON found in final_optimized_parameters")

    with open(combined_json, 'r', encoding='utf-8') as f:
        combined = json.load(f)

    # pull data (assume keys exist)
    human_raw = combined['human_params_raw']
    object_raw = combined['object_params_raw']
    incam_subset = combined['smpl_params_incam_subset']
    global_subset = combined['smpl_params_global_subset']
    transformed_section = combined['transformed']
    # check lengths
    print(f"Frames in human raw: {len(human_raw.get('body_pose', {}))}, object raw: {len(object_raw.get('poses', {}))}, incam subset: {len(incam_subset.get('global_orient', {}))}, global subset: {len(global_subset.get('global_orient', {}))}")
    # return
    # transformed object params (use original transformed parameters)
    # transformed_object_params = transformed_section['parameters']['object_params']
    # object mesh path (assume present)
    mesh_path = os.path.join(args.data_dir,'obj_init.obj')
 
    # SMPL model
    smpl_model = smplx.create(
        args.smpl_model,
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True
    )
    if torch.cuda.is_available():
        smpl_model = smpl_model.cuda()

    # frame keys
    frame_keys = sorted(human_raw.get('body_pose', {}).keys(), key=lambda k: int(k))
    if not frame_keys:
        raise ValueError("No frames found in human params")

    # load original object mesh (vertices used for per-frame transforms)
    obj_tm = trimesh.load(mesh_path, process=False)
    obj_vertices = np.asarray(obj_tm.vertices, dtype=np.float32)
    obj_faces = np.asarray(obj_tm.faces, dtype=np.int32)
    obj_colors = np.asarray(obj_tm.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0 if obj_tm.visual.vertex_colors is not None else None
    # obj_colors=torch.tensor(obj_colors).cuda() if obj_colors is not None else None

    num_frames = len(frame_keys)

    ## save transformed human and object parameters
    transformed_human_params = {}
    transformed_object_params = {}
    transformed_human_params['body_pose'] = {}
    transformed_human_params['betas'] = {}
    transformed_human_params['global_orient'] = {}
    transformed_human_params['transl'] = {}
    transformed_human_params['left_hand_pose'] = {}
    transformed_human_params['right_hand_pose'] = {}
    transformed_object_params['R_total'] = {}
    transformed_object_params['T_total'] = {}
    # Precompute human meshes (SMPL evaluation)
    human_vertices = []
    pelvises=[]
    org_pelvises=[]
    faces_human = None
    joints=[]

    motion_output=torch.load(os.path.join(args.data_dir, "motion", "result.pt"))
    global_body_params = motion_output["smpl_params_global"]
    incam_body_params = motion_output["smpl_params_incam"]

    # vis_body_dir='/data/boran/4dhoi/data_hoi/chair/20250825_172609/debug/visual/'
    # body_pose_list=os.listdir(vis_body_dir)
    # body_pose_list.sort()
    # print(len(body_pose_list), len(frame_keys))

    for idx,fk in tqdm(enumerate(frame_keys), desc='SMPL forward'):
        bp = np.asarray(human_raw['body_pose'][fk], dtype=np.float32)

        # bp=np.load(os.path.join(vis_body_dir, body_pose_list[idx]), allow_pickle=True)
        betas = np.asarray(human_raw['betas'][fk], dtype=np.float32)
        # # glob_orient = np.asarray(human_raw['global_orient'][fk], dtype=np.float32)
        # # transl = np.asarray(human_raw['transl'][fk], dtype=np.float32)
        left_hand = np.asarray(human_raw['left_hand_pose'][fk], dtype=np.float32)
        right_hand = np.asarray(human_raw['right_hand_pose'][fk], dtype=np.float32)
        glob_orient_org = np.asarray(human_raw['global_orient'][fk], dtype=np.float32)
        transl_org = np.asarray(human_raw['transl'][fk], dtype=np.float32)

        glob_o = torch.tensor(global_subset['global_orient'][idx], dtype=torch.float32)
        glob_t = torch.tensor(global_subset['transl'][idx], dtype=torch.float32)

        # print(bp.shape)

        # bp[:3]=np.asarray(global_body_params['body_pose'][idx+66], dtype=np.float32)[:3]

        # bp=np.asarray(global_body_params['body_pose'][idx+66], dtype=np.float32)
        # betas=np.asarray(global_body_params['betas'][0], dtype=np.float32)
        # bp=np.asarray(incam_body_params['body_pose'][idx+66], dtype=np.float32)
        # betas=np.asarray(incam_body_params['betas'][idx+66], dtype=np.float32)

        # glob_o=np.asarray(global_body_params['global_orient'][idx+66], dtype=np.float32)
        # glob_t=np.asarray(global_body_params['transl'][idx+66], dtype=np.float32)
        # left_hand=np.zeros(45, dtype=np.float32)
        # right_hand=np.zeros(45, dtype=np.float32)          


        with torch.no_grad():
            bpt = ensure_cuda(torch.tensor(bp).view(1, -1))
            betat = ensure_cuda(torch.tensor(betas).view(1, -1))
            glt = ensure_cuda(torch.tensor(glob_o).view(1, 3))
            trt = ensure_cuda(torch.tensor(glob_t).view(1, 3))
            lht = ensure_cuda(torch.tensor(left_hand).view(1, -1))
            rht = ensure_cuda(torch.tensor(right_hand).view(1, -1))
            out = smpl_model(
                betas=betat,
                body_pose=bpt,
                left_hand_pose=lht,
                right_hand_pose=rht,
                jaw_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                leye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                reye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                global_orient=glt,
                expression=torch.zeros((1,10)).cuda() if torch.cuda.is_available() else torch.zeros((1,10)),
                transl=trt
            )
        verts = out.vertices[0].cpu().numpy()
        human_vertices.append(verts)
        pelvis=out.joints[:, 0, :]
        pelvises.append(pelvis)
        joints.append(out.joints[0])

        with torch.no_grad():
            bpt = ensure_cuda(torch.tensor(bp).view(1, -1))
            betat = ensure_cuda(torch.tensor(betas).view(1, -1))
            glt = ensure_cuda(torch.tensor(glob_orient_org).view(1, 3))
            trt = ensure_cuda(torch.tensor(transl_org).view(1, 3))
            lht = ensure_cuda(torch.tensor(left_hand).view(1, -1))
            rht = ensure_cuda(torch.tensor(right_hand).view(1, -1))
            out_org = smpl_model(
                betas=betat,
                body_pose=bpt,
                left_hand_pose=lht,
                right_hand_pose=rht,
                jaw_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                leye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                reye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                global_orient=glt,
                expression=torch.zeros((1,10)).cuda() if torch.cuda.is_available() else torch.zeros((1,10)),
                transl=trt
            )
        org_pelvis=out_org.joints[:, 0, :]
        org_pelvises.append(org_pelvis)
        
        # human_vertices.append(verts)
        if faces_human is None:
            faces_human = np.asarray(smpl_model.faces, dtype=np.int32)
        ## save transformed params
        transformed_human_params['body_pose'][fk] = bp.tolist()
        transformed_human_params['betas'][fk] = betas.tolist()
        transformed_human_params['global_orient'][fk] = glob_o.tolist()
        transformed_human_params['transl'][fk] = glob_t.tolist()
        transformed_human_params['left_hand_pose'][fk] = left_hand.tolist()
        transformed_human_params['right_hand_pose'][fk] = right_hand.tolist()

    # Apply transform_to_global per frame for human; use transformed R_total/T_total for object
    human_vertices_transformed = []
    object_vertices_per_frame = []
    for idx, fk in enumerate(frame_keys):
        hverts = human_vertices[idx]
        # object default verts
        overts = obj_vertices.copy()
        pelvis=pelvises[idx]
        org_pelvis=org_pelvises[idx]
        # print(pelvis,org_pelvis)

        R_ = np.asarray(object_raw['poses'][fk], dtype=np.float32)
        # Convert R to axis-angle representation
        R_axis_angle = torch.tensor(Rotation.from_matrix(R_).as_rotvec(), dtype=torch.float32)
        T = torch.tensor(object_raw['centers'][fk], dtype=torch.float32)
        T_np = np.asarray(object_raw['centers'][fk], dtype=np.float32)   # (3,)

        # transform human with transform_to_global (use incam/global subsets)
        incam_o = torch.tensor(incam_subset['global_orient'][idx], dtype=torch.float32)
        incam_t = torch.tensor(incam_subset['transl'][idx], dtype=torch.float32)
        glob_o = torch.tensor(global_subset['global_orient'][idx], dtype=torch.float32)
        glob_t = torch.tensor(global_subset['transl'][idx], dtype=torch.float32)
        incam_params = (incam_o, incam_t)
        global_params = (glob_o, glob_t)

        new_o, new_t = apply_transform_to_smpl_params(
            R_axis_angle, T,
            (incam_o, org_pelvis), (glob_o, pelvis)
        )
        # print('check now',incam_o, incam_t, glob_o, glob_t)
        # Convert new_o to matrix
        # Convert axis-angle representation to rotation matrix

        R_old = axis_angle_to_matrix(incam_o).squeeze(0)  # (3,3)
        R_new = axis_angle_to_matrix(glob_o).squeeze(0)  # (3,3)
        T_old = org_pelvis.detach().cpu().squeeze(0)
        T_new = pelvis.detach().cpu().squeeze(0)

        R_delta = R_new @ R_old.T
        t_delta = T_new - (T_old @ R_delta.T)

        R_ind = R_delta
        t_ind = t_delta
        R_total = R_ind @ torch.from_numpy(R_).float()
        T_total = torch.from_numpy(T_np).float() @ R_ind.T + t_ind
        overts = (obj_vertices @ R_total.cpu().numpy().T) + T_total.cpu().numpy()

        # new_o_matrix = Rotation.from_rotvec(new_o).as_matrix()
        # overts = (obj_vertices @ new_o_matrix.T) + new_t

        ## save transformed object params
        transformed_object_params['R_total'][fk] = R_total.tolist()
        # transformed_object_params['R_total'][fk] = new_o.tolist()
        transformed_object_params['T_total'][fk] = T_total.tolist()
        # transformed_object_params['T_total'][fk] = new_t.tolist()

        human_vertices_transformed.append(hverts.astype(np.float32))
        object_vertices_per_frame.append(overts.astype(np.float32))
        # return 

    # save transformed parameters
    save_dict={
        'human_params_transformed': transformed_human_params,
        'object_params_transformed': transformed_object_params
    }
    save_path=os.path.join(args.data_dir, 'transformed_parameters_final.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=4)

    # Convert to torch and prepare renderer
    # verts_human_t = torch.tensor(np.stack(human_vertices_transformed, axis=0), dtype=torch.float32).cuda()
    # verts_object_t = torch.tensor(np.stack(object_vertices_per_frame, axis=0), dtype=torch.float32).cuda()
    # faces_human_t = faces_human
    # faces_obj = obj_faces

    # # camera intrinsics K
    # K = torch.tensor([
    #     [1200.0, 0.0, args.width / 2],
    #     [0.0, 1200.0, args.height / 2],
    #     [0.0, 0.0, 1.0]
    # ], dtype=torch.float32)
    # if torch.cuda.is_available():
    #     K = K.cuda()

    # renderer = Renderer(args.width, args.height, device="cuda" if torch.cuda.is_available() else "cpu",
    #                     faces_human=faces_human_t, faces_obj=faces_obj, K=K)

    # # compute combined verts for camera placement
    # combined_verts = torch.cat([verts_human_t, verts_object_t], dim=1)  # (F, N_h + N_o, 3)
    # # combined_verts = verts_human_t # (F, N_h + N_o, 3)
    # # For camera calc we move to CPU

    # # def move_to_start_point_face_z(verts,joints):
    # #     "XZ to origin, Start from the ground, Face-Z"
    # #     # position
    # #     verts = verts.clone()  # (L, V, 3)
    # #     offset = joints[0][0]  # (3)
    # #     offset[1] = verts[:, :, [1]].min()
    # #     verts = verts - offset
    # #     # face direction
    # #     T_ay2ayfz = compute_T_ayfz2ay(joints[[0]], inverse=True)
    # #     verts = apply_T_on_points(verts, T_ay2ayfz)
    # #     return verts
    # # joints_glob=torch.stack(joints, dim=0)
    # # # print(joints_glob.shape)
    # # verts_glob = move_to_start_point_face_z(combined_verts, joints_glob)
    # verts_glob = combined_verts
    

    # global_R, global_T, global_lights = get_global_cameras_static(
    #     verts_glob.cpu(),
    #     beta=2.5,
    #     cam_height_degree=20,
    #     target_center_height=1.0,
    #     vec_rot=180
    # )

    # # set ground from joints estimate (approx using mean of human verts)
    # # build a fake joints tensor as (F, J, 3) where J=1 center
    # joints_glob = verts_human_t.mean(dim=1, keepdim=True)  # (F,1,3)
    # scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob.cpu())
    # renderer.set_ground(scale * 1.5, cx, cz)

    # # render frames and save images
    # temp_dir = os.path.join(args.data_dir, 'final_optimized_parameters', 'temp_frames_render')
    # os.makedirs(temp_dir, exist_ok=True)
    # frame_paths = []
    # color = torch.ones(3).float().cuda() * 0.8
    # for i in tqdm(range(num_frames), desc='Rendering frames'):
    # # for i in tqdm(range(2), desc='Rendering frames'):
    #     cams = renderer.create_camera(global_R[i], global_T[i])
    #     img = renderer.render_with_ground_hoi(verts_human_t[i], verts_object_t[i], cams, global_lights, [0.8, 0.8, 0.8],
    #                                          obj_colors)
    #     # img = renderer.render_with_ground(verts_glob[i].unsqueeze(0), color[None],cams, global_lights)
    #     img = np.clip(img, 0, 255).astype(np.uint8)
    #     path = os.path.join(temp_dir, f'frame_{i:04d}.png')
    #     Image.fromarray(img).save(path, optimize=False)
    #     frame_paths.append(path)

    # # write video using imageio
    # import imageio
    # writer = imageio.get_writer(os.path.join(args.data_dir, args.out), fps=args.fps, codec='libx264', quality=9)
    # for p in tqdm(frame_paths, desc='Writing video'):
    #     img = imageio.imread(p)
    #     writer.append_data(img)
    # writer.close()

    # print("Render complete. Video saved to:", os.path.join(args.data_dir, args.out))


if __name__ == "__main__":
    main()
