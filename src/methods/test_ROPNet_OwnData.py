"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test ROPNet OwnData

Run ROPNet, with given parameters, for given .h5 file

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - .hdf5 file with estimated transformation

Credits: 
    ROPNet Code by zhulf0804
    Link: https://github.com/zhulf0804/ROPNet

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import numpy as np
import os
import random
import time
import torch

BASE_DIR = os.getcwd()


# ROPNet toolbox imports
from src.toolboxes.ROPNet.src.configs import eval_config_params
from src.toolboxes.ROPNet.src.models import ROPNet
from src.toolboxes.ROPNet.src.utils import npy2pcd, pcd2npy, inv_R_t, batch_transform, square_dists, \
    format_lines, vis_pcds


"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# Function to save the estimated transformation in the correct format
def extract_transformation(R, t):
    T = np.zeros((4,4))
    T[3,3] = 1
    T[0:3,0:3] = np.array(R.tolist()[0])
    T[:3,3] = np.array(t.tolist()[0])
    return T

def evaluate_ROPNet(src_cloud, tgt_cloud, args):
    model = ROPNet(args)
    
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        
        if args.cuda:
            tgt_cloud, src_cloud = tgt_cloud.cuda(), src_cloud.cuda()
                                             
        tic = time.time()
        B, N, _ = src_cloud.size()
        results = model(src=src_cloud,
                        tgt=tgt_cloud,
                        num_iter=2)
        toc = time.time()
        registration_time = toc - tic
        pred_Ts = results['pred_Ts']
        R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
        transformation = extract_transformation(R, t)

    return transformation, registration_time


def main(source_tensor, template_tensor):
    
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = eval_config_params()
    
    args.cuda = True
    args.normal = True
    args.p_keep = [1, 1] #If only one value => source partial, template complete
    
    #Change directory of pretrained
    # checkpoint_dir = "ROPNet/src/work_dirs/models/checkpoints/min_rot_error.pth"
    checkpoint_dir = "ROPNet/src/work_dirs/partial_0.7_noisy_0.01_floor/models/checkpoints/min_rot_error.pth"
    args.checkpoint = os.path.join(os.getcwd(),'src/toolboxes/' + checkpoint_dir)
    
    #Input root directory
    PRE_DIR = "src/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048"
    args.root = os.path.join(BASE_DIR, PRE_DIR)
    

    transformation, reg_time = \
        evaluate_ROPNet(source_tensor, template_tensor, args)
    
    return transformation, reg_time

if __name__ == '__main__':
    main()