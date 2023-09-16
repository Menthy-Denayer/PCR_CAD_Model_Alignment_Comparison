"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test PointNetLK OwnData

Run PointNetLK, with given parameters, for given inputs

Inputs:
    - source tensor
    - template tensor

Output:
    - found transformation
    - registration time

Credits: 
    PointNetLK Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# General imports
import os
import argparse

# Array related operations
import torch.utils.data

# Testing imports
from src.methods.tester import test_one_epoch

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# print(BASE_DIR)

#learning3d toolbox imports
from src.toolboxes.learning3d.models import PointNet, PointNetLK
# from toolboxes.learning3d.data_utils import UserData,RegistrationData,ModelNet40Data

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_pnlk_v1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--pretrained', default='learning3d/pretrained/exp_pnlk/models/best_model.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

def test(source_tensor, template_tensor, args, model):
    transformation, reg_time = test_one_epoch(source_tensor, template_tensor, 
                                              args.device, model, algo="PointNetLK")
    return transformation, reg_time

def main(source_tensor, template_tensor):
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Load arguments
    args = options()
    
    #Change directory of pretrained
    # PRE_DIR = "src/toolboxes/learning3d/checkpoints/exp_pnlk_noisy/models/best_model.t7"
    PRE_DIR = "src/toolboxes/learning3d/checkpoints/exp_pnlk/models/best_model.t7"
    args.pretrained = os.path.join(BASE_DIR, PRE_DIR)
    print(args.pretrained)
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNet Model.
    ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
    model = PointNetLK(feature_model=ptnet,pooling="max",
                       p0_zero_mean = True, p1_zero_mean = True) #avg much better than #max for noisy data
    model = model.to(args.device)

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)
    
    # Solve with PointNetLK and return registration time
    transformation, reg_time = test(source_tensor, template_tensor, args, model)
    return transformation, reg_time
    
if __name__ == '__main__':
    main()
    
"""
other way around for PRNet?
source =   igt   * template =   igt   * template
tranfo =   Tes   * source
source = (Tes)-1 * transfo ~= (Tes)-1 * template

"""