"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



ground truth accuracy

Assess the performance of the ground truth transformation by applying ICP
refinement with the ground truth estimate as a first guess

Inputs:
    - .json file

Output:
    - mean errors
    - variance of errors
    
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
import copy
from .data_manager import preprocess_data, save_result, invert_transformation
from .methods import test_ICP_OwnData as ICP
from .compute_errors import compute_errors, compute_mean_variance

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def show_open3d(source_pointcloud, template_pointcloud, name = "Open3D Window"):
    
    """
    Apply transformation to point cloud
    
    Parameters
    ----------
    pointcloud                      : Open3D Point Cloud        : point cloud to be transformed
    transformation_matrix           : 4x4 list                  : transformation to be applied
    
    Returns
    ----------
    transformed_pointcloud          : Open3D Point CLoud        : transformed point cloud
    """
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.scale(1/50,center=(0,0,0))
    
    # Show results in Open3d
    source_pointcloud.paint_uniform_color([1,0,0])
    template_pointcloud.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([template_pointcloud,source_pointcloud, coordinate_frame], 
                                      window_name = name)


def apply_transformation(pointcloud, transformation_matrix):
    """
    Apply transformation to point cloud
    
    Parameters
    ----------
    pointcloud                      : Open3D Point Cloud        : point cloud to be transformed
    transformation_matrix           : 4x4 list                  : transformation to be applied
    
    Returns
    ----------
    transformed_pointcloud          : Open3D Point CLoud        : transformed point cloud
    """
    
    # Turn one ground truth transformation into array
    transformation = np.asarray(transformation_matrix)
    
    transformed_pointcloud = copy.deepcopy(pointcloud)
    transformed_pointcloud.transform(transformation)
    
    return transformed_pointcloud

def evaluate_ground_truth(gt_json_file, voxel_size = 0.01, recall_lim = 0.01):
    """
    Evaluate the ground truth for a given ground truth .json file
    
    Parameters
    ----------
    gt_json_file                : string                        : path to .json file
    voxel_size                  : 4x4 list                      : voxel size used for ICP
    recall_lim                  : float                         : limit for computing recall metric
    
    Returns
    ----------
    errors                      : 8x1 numpy array               : list of computed metrics
    """
    
    """ Load ground truth files"""
    template_pointcloud, source_pointcloud, transformation, json_info = \
        preprocess_data(gt_json_file, voxel_size = 0)
        
    """ Apply ground truth transformation"""
    
    transformed_template_pointcloud = apply_transformation(template_pointcloud, transformation[0])
    
    show_open3d(source_pointcloud, transformed_template_pointcloud, name = "Ground Truth")

    """ Apply registration method """
    
    initial_transformation = invert_transformation(transformation[0])
    
    registration_transformation,reg_time = \
        ICP.main(source_pointcloud, template_pointcloud, voxel_size, initial_transformation)
    
    transformed_source_pointcloud = apply_transformation(source_pointcloud, registration_transformation)

    show_open3d(transformed_source_pointcloud, template_pointcloud, name = "Registration Result")
    
    """ Save refined ground truth estimation """
    
    # Add specific registration parameters
    registration_parameters = {"method" : "ICP Refinement",
                               "voxel_size" : voxel_size,
                               "centered" : False}
    
    # Save result
    file_name = gt_json_file.split('/')[-1].split('.json')[0]
    output_folder = "ICP/ground_truth_refinement"
    save_result(file_name, output_folder, transformation, registration_transformation, 
                      reg_time, json_info, registration_parameters)
    
    """ Compute errors compared to ground truth transformation """
     
    BASE_DIR = os.getcwd() + "/results/"
    result_json_file = BASE_DIR + output_folder + "/" + file_name + "_result.json"
    
    errors = compute_errors(result_json_file, recall_lim)
    
    return errors

def evaluate_ground_truth_batch(json_dir, voxel_size = 0.01, recall_lim = 0.01):
    """
    Evaluate the ground truth for a series of ground truth .json files
    
    Parameters
    ----------
    gt_json_file                : string                        : path to folder of .json files
    voxel_size                  : 4x4 list                      : voxel size used for ICP
    recall_lim                  : float                         : limit for computing recall metric
    
    Returns
    ----------
    mean_error                  : 8x1 numpy array               : list of mean errors 
    variance                    : 8x1 numpy array               : list of computed variances
    """
    
    error_list = np.zeros((8,1))
    
    for _,json_file in enumerate(tqdm(os.listdir(json_dir))):
        
        errors = evaluate_ground_truth(json_dir + "/" + json_file, voxel_size, recall_lim)
        error_list = np.append(error_list, errors, 1)
    
    mean_error, variance = compute_mean_variance(error_list[:,1:])
    return  mean_error, variance