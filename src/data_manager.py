"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import json
import torch
import numpy as np
import open3d as o3d

"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

BASE_DIR = os.getcwd()

cad_path = BASE_DIR + "/datasets/CAD"
src_path = BASE_DIR + "/datasets/point_clouds"
gt_path = BASE_DIR+ "/datasets/ground_truth"
out_path = BASE_DIR + "/results"

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def read_json(path,key = None):
    """
    Read key of json file, located in path

    Parameters
    ----------
    path        : string            : path to .json file
    key         : string            : key of .json file to read
       
    Returns
    -------
    data        : data              : data loaded from .json file
    
    Link: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
    """
    
    # Opening JSON file
    with open(path, 'r') as openfile:
 
        # Reading from json file
        json_object = json.load(openfile)
    
    if(key):
        return json_object[key]
    else:
        return json_object

def write_result_json(experiment_name, ground_truth, estimated_transfo, registration_time, 
                      dictionary):
    
    """
    Save result file with all experiment info into .json

    Parameters
    ----------
    experiment_name             : string                : name of experiment folder
    ground_truth                : Nx4x4 numpy array     : ground truth transformations
    estimated_transfo           : 1x4x4 numpy array     : estimated transformation
    registration_time           : float                 : computation time for registration
    dictionary                  : dictionary            : parameters to save
    
    Link: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
    """
    
    # Naming data
    object_name = dictionary["name"]
    scan_nmb = dictionary["scan"]
    
    # Data to be written
    dictionary["transformation"] = ground_truth
    dictionary["estimated_transformation"] = estimated_transfo.tolist()
    dictionary["frame"] = "template to source",
    dictionary["result frame"] = "source to template"
    dictionary["registration time [s]"] = registration_time

    # Serializing json
    json_object = json.dumps(dictionary, indent=3)
    
    out_folder = out_path + "/" + experiment_name
    out_file = out_folder + "/" + object_name + "_" + str(scan_nmb) + "_result.json"
    
    if not os.path.exists(out_path + "/" + experiment_name):
        os.makedirs(out_path + "/" + experiment_name)
 
    # Writing to sample.json
    with open(out_file, "w") as outfile:
        outfile.write(json_object)


def prepare_stl(template_stl, nmb_source_points, multiple, scale = 1,
                Normals_radius = 0.01, Normals_Neighbours = 30):
    """
    Prepare .stl file to .ply file
    
    Parameters
    ----------
    template_stl        : Open3d TriangleMesh Object                        : .stl file of template
    nmb_source_points   : int                                               : number of points in source PC
    multiple            : float                                             : multiple of nmb_source_points
    scale               : float                                             : scale factor for template
    Normals_radius      : float (to estimate normal vectors on template)    : estimate normal vectors
    Normals_Neighbours  : float (to estimate normal vectors on template)    : estimate normal vectors
    
    Returns
    -------
    template_pointcloud : Open3D Point Cloud                                : template point cloud
    """
    
    nmb_template_points = nmb_source_points*multiple
    template_pointcloud = template_stl.sample_points_uniformly(number_of_points=nmb_template_points)  
    
    template_pointcloud.scale(1/scale,center=template_pointcloud.get_center())
    template_pointcloud.translate(-template_pointcloud.get_center())
    
    template_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=Normals_radius, max_nn=Normals_Neighbours))

    return template_pointcloud

def preprocess_data(json_file, voxel_size, zero_mean = False):
    """
    Load correct files from directory and preprocess

    Parameters
    ----------
    json_file           : string                : location of .json file
    voxel_size          : float                 : down sample point clouds
    zero_mean           : boolean               : center point clouds
       
    Returns
    -------
    template_pointcloud : Open3D Point Cloud    : template point cloud
    source_pointcloud   : Open3D Point Cloud    : source point cloud
    """
    
    # Read data
    json_info = read_json(json_file)
    object_name = json_info["name"]
    scan_nmb = json_info["scan"]
    BB = json_info["bounding_box"]

    
    if(BB == 0):
        src_ply_file = src_path + "/filtered" 
    else:
        src_ply_file = src_path + "/bounding_box/BB_" + str(BB) 
    
    src_ply_file = src_ply_file + "/" + object_name + "_" + str(scan_nmb)
    
    if(BB != 0):
        src_ply_file = src_ply_file + "_BB_" + str(BB) 
        
    src_ply_file = src_ply_file + "_Source.ply"
    tmp_stl_file = cad_path + "/" + object_name + ".stl"
    
    # print(src_ply_file)
    # print(gt_json_file)
    # print(tmp_stl_file)
    
    if(os.path.isfile(src_ply_file) and os.path.isfile(json_file) and os.path.isfile(tmp_stl_file)):

        """
        Initializations
        ---------------
        Turn files into Open3D PointCloud Object
        """

        sampling_ratio = json_info["sampling_density"]
        scale = json_info["scale"]

        source_pointcloud = o3d.io.read_point_cloud(src_ply_file)
        template_stl = o3d.io.read_triangle_mesh(tmp_stl_file)  

        nmb_source_points = len(np.asarray(source_pointcloud.points))
        template_pointcloud = prepare_stl(template_stl, nmb_source_points, sampling_ratio, scale)
        
        if(voxel_size > 0):
            source_pointcloud = source_pointcloud.voxel_down_sample(voxel_size)
            template_pointcloud = template_pointcloud.voxel_down_sample(voxel_size)
            
        if(zero_mean):
            source_pointcloud.translate(-source_pointcloud.get_center())
            template_pointcloud.translate(-template_pointcloud.get_center())
        

        """
        Ground Truth Estimation
        ---------------
        Estimate ground truth transformation, applied on template --> source
        """

        transformation = read_json(json_file, "transformation")
        
        """
        Information
        -----------
        Save information from .json file
        """

    else:
        print("This combination does not exist. Check for spelling errors.")
    
    return template_pointcloud, source_pointcloud, transformation, json_info

def pointcloud_to_torch(pointcloud):
    """
    Turn point cloud object into torch tensor

    Parameters
    ----------
    pointcloud          : Open3D Point Cloud    : point cloud
       
    Returns
    -------
    tensor              : 1xNx6 Torch Tensor    : tensor
    """
    
    points_array = np.asarray(pointcloud.points)
    normals_array = np.asarray(pointcloud.normals)
    array = np.concatenate((points_array,normals_array),1)
    
    tensor = torch.tensor(array)
    tensor = tensor.expand(1,tensor.size(0),6)
    
    return tensor.float()