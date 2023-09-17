"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



visualizer

Visualizes ground truth or registration found transformation

Inputs:
    - .json file
    - voxel size
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import open3d as o3d
import numpy as np
import copy
import os
from .data_manager import preprocess_data, read_json


"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

BASE_DIR = os.getcwd()

cad_path = BASE_DIR + "/datasets/CAD"
src_path = BASE_DIR + "/datasets/point_clouds"
gt_path = BASE_DIR + "/datasets/ground_truth"
result_path = BASE_DIR + "/datasets/results"

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def apply_transformation(template_pointcloud, source_pointcloud, transformation_matrix):
    """
    Visualises transformation on source
    
    Parameters
    ----------
    source_pointcloud       : Open3D Point Cloud    : source point cloud
    template_pointcloud     : Open3D Point Cloud    : template point cloud
    transformation_matrix   : 4x4 numpy array       : 4x4 ground truth transformation matrix
    """
    
    # Turn one ground truth transformation into array
    transformation = np.asarray(transformation_matrix)
    
    # Apply ground truth transformation on template
    source_pointcloud_transformed = copy.deepcopy(source_pointcloud)
    source_pointcloud_transformed.transform(transformation)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.scale(1/50,center=(0,0,0))
    
    # Show results in Open3d
    source_pointcloud.paint_uniform_color([1,0,0])
    source_pointcloud_transformed.paint_uniform_color([0,1,0])
    template_pointcloud.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([template_pointcloud,source_pointcloud, coordinate_frame], 
                                      window_name = "Original Point Clouds")
    o3d.visualization.draw_geometries([template_pointcloud,source_pointcloud_transformed,coordinate_frame], 
                                      window_name = "Transformation")
    
    return


def visualise_ground_truth(gt_json_file, voxel_size = 0, zero_mean = False):
    """
    Visualize ground truth transformation from .json file
    
    Parameters
    ----------
    gt_json_file            : string                : path to result .json file
    voxel_size              : float                 : down sample point cloud
    """
    
    template_pointcloud, source_pointcloud, transformation,_ = \
        preprocess_data(gt_json_file, voxel_size, zero_mean)

    apply_transformation(source_pointcloud, template_pointcloud, transformation[0])
    
    return

def visualise_result(json_file, voxel_size = 0):
    """
    Visualize registration result from .json file
    
    Parameters
    ----------
    json_file               : string                : path to result .json file
    voxel_size              : float                 : down sample point cloud
    """
       
    # Load data as point clouds
    template_pointcloud, source_pointcloud,_,json_info = \
        preprocess_data(json_file, voxel_size)
        
    transformation = json_info["estimated_transformation"]

    apply_transformation(template_pointcloud, source_pointcloud, transformation)
    
    return