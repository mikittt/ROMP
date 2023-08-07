import argparse
import numpy as np
import pickle
import os
import torch
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from romp.utils import create_OneEuroFilter, smooth_results
from bev.post_parser import SMPLA_parser, denormalize_cam_params_to_trans


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=int, help="Number of samples to generate", default=1
    )
    parser.add_argument(
        "--output_path", type=str, help='None'
    )
    return parser.parse_args()

def smooth(vecs,weight):
    last=vecs[0]
    smoothed=[]
    for vec in vecs:
        smoothed_vec=last*weight+(1-weight)*vec
        smoothed.append(smoothed_vec)
        last=smoothed_vec

    return np.array(smoothed)

def track(trans):
    prev_trans1 = trans[0][0]
    track1 = [0]
    track2 = [1]
    for i in range(1, len(trans)):
        rank = np.argsort(np.linalg.norm(trans[i]-prev_trans1[None, :], ord=2, axis=1))
        track1.append(rank[0])
        track2.append(rank[1])
        prev_trans1 = trans[i][rank[0]]

    return track1, track2

args = get_args()
root_dir = args.output_path
smpl_parser = SMPLA_parser(
    os.path.join(os.path.expanduser("~"),'.romp','smpla_packed_info.pth'), 
    os.path.join(os.path.expanduser("~"),'.romp','smil_packed_info.pth')).to(0)
smooth_coeff = 0.4
results = []
for one_file in tqdm(glob(os.path.join(root_dir, '*/*.npz'))):
    bev_result = np.load(one_file, allow_pickle=True)['results'].item()
    file_list = natsorted(bev_result.keys())
    num_per_frame = [len(bev_result[one_file]['cam_trans']) for one_file in file_list]
    intervals = []
    start = None
    for i, one in enumerate(num_per_frame):
        if one==2:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append([start, i-1])
                start = None
    if start is not None:
        intervals.append([start, i])
    if len(intervals) == 0:
        continue
    max_interval = intervals[np.argmax([interval[1]-interval[0] for interval in intervals])]
    interval_num_ratio = (max_interval[1]-max_interval[0])/len(num_per_frame)
    if interval_num_ratio<0.8:
        continue

    cam_trans = np.array([bev_result[file_list[i]]['cam_trans'] for i in range(max_interval[0], max_interval[1]+1)])
    smpls = np.array([bev_result[file_list[i]]['smpl_thetas'] for i in range(max_interval[0], max_interval[1]+1)])
    joints = np.array([bev_result[file_list[i]]['joints'] for i in range(max_interval[0], max_interval[1]+1)])

    center_coord = (joints[0, :, 0, :]+cam_trans[0]).mean(axis=0)
    trans = np.array([joints[i,:,0]+cam_trans[i]-center_coord for i in range(len(cam_trans))])
    joints = np.array([joints[i]+cam_trans[i][:,None]-center_coord[None, None, :] for i in range(len(cam_trans))])

    #trans, smpls, joints = track(trans, smpls, joints)
    track1, track2 = track(trans)

    cams = np.array([bev_result[file_list[i]]['cam'] for i in range(max_interval[0], max_interval[1]+1)])
    thetas = np.array([bev_result[file_list[i]]['smpl_thetas'] for i in range(max_interval[0], max_interval[1]+1)])
    betas = np.array([bev_result[file_list[i]]['smpl_betas'] for i in range(max_interval[0], max_interval[1]+1)])

    cams1, cams2 = torch.from_numpy(cams[np.arange(len(cams)),track1]), torch.from_numpy(cams[np.arange(len(cams)),track2])
    thetas1, thetas2 = torch.from_numpy(thetas[np.arange(len(cams)),track1]), torch.from_numpy(thetas[np.arange(len(cams)),track2])
    betas1, betas2 = torch.from_numpy(betas[np.arange(len(cams)),track1]), torch.from_numpy(betas[np.arange(len(cams)),track2])

    OE_filters = create_OneEuroFilter(smooth_coeff)
    thetas1_ = []
    betas1_ = []
    cams1_ = []
    for ii in range(len(thetas1)):
        thetas1_1, betas1_1, cams1_1 = smooth_results(
            OE_filters, thetas1[ii], betas1[ii], cams1[ii])
        thetas1_.append(thetas1_1[None,:])
        betas1_.append(betas1_1[None,:])
        cams1_.append(cams1_1)
    trans1_ = denormalize_cam_params_to_trans(cams1).cpu().numpy()
    trans1_ = smooth(trans1_, 0.7)
    _, joints1_, _ = smpl_parser(torch.cat(betas1_), torch.cat(thetas1_))
    OE_filters = create_OneEuroFilter(smooth_coeff)
    thetas2_ = []
    betas2_ = []
    cams2_ = []
    for ii in range(len(thetas1)):
        thetas2_1, betas2_1, cams2_1 = smooth_results(
            OE_filters, thetas2[ii], betas2[ii], cams2[ii])
        thetas2_.append(thetas2_1[None,:])
        betas2_.append(betas2_1[None,:])
        cams2_.append(cams2_1)
    trans2_ = denormalize_cam_params_to_trans(cams2).cpu().numpy()
    trans2_ = smooth(trans2_, 0.5)
    _, joints2_, _ = smpl_parser(torch.cat(betas2_), torch.cat(thetas2_))

    trans = np.concatenate([trans1_[None, :], trans2_[None, :]], axis=0)# (2, seq_len, 3)
    thetas = np.concatenate([torch.cat(thetas1_).cpu().numpy()[None, :], torch.cat(thetas2_).cpu().numpy()[None, :]], axis=0)
    joints = np.concatenate([joints1_.cpu().numpy()[None, :], joints2_.cpu().numpy()[None, :]], axis=0)# (2,seq_len, pose_dim, 3)

    joints = joints+trans[:,:,None]
    mean_init_pos = joints[:,0,0].mean(axis=0)
    joints = joints-mean_init_pos[None,None,None,:]
    results.append(
        {
            'smpl':thetas,
            'joints':joints,
            'file_name':one_file.split('/')[-1].split('.')[0]
        }
    )
with open('merged.pkl', 'wb') as f:
    pickle.dump(results, f)