"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



ground truth visualisation

Visualizes ground truth transformation

Inputs:
    - Object name
    - Scan number
    - Bounding box size
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import open3d as o3d
import numpy as np
import json
import copy
import os


"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

BASE_DIR = os.getcwd()

cad_path = BASE_DIR + "/datasets/CAD"
src_path = BASE_DIR + "/datasets/point_clouds"
gt_path = BASE_DIR+ "/datasets/ground_truth"

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def read_json(path,key):
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
 
    return json_object[key]

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



def ground_truth_estimation(transformation_matrix, source_pointcloud, template_pointcloud):
    """
    Visualises ground truth transformation on template
    
    Parameters
    ----------
    transformation_matrix   : Nx4x4 numpy array     : 4x4 ground truth transformation matrix
    source_pointcloud       : Open3D Point Cloud    : source point cloud
    template_pointcloud     : Open3D Point Cloud    : template point cloud
    """
    
    # Turn one ground truth transformation into array
    ground_truth = np.asarray(transformation_matrix)[0,:,:]
    
    # Apply ground truth transformation on template
    template_pointcloud_transformed = copy.deepcopy(template_pointcloud)
    template_pointcloud_transformed.transform(ground_truth)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.scale(1/50,center=(0,0,0))
    
    # Show results in Open3d
    source_pointcloud.paint_uniform_color([1,0,0])
    template_pointcloud_transformed.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([template_pointcloud,source_pointcloud, coordinate_frame])
    o3d.visualization.draw_geometries([template_pointcloud_transformed,source_pointcloud,coordinate_frame])
    
    return


def visualise_ground_truth(object_name, scan_nmb, BB = 0):
    """
    Load correct files from directory
    
    Parameters
    ----------
    object_name     : string        : name of objects in dataset
    scan_nmb        : int           : number of scan
    BB              : float         : size of bounding box
    """
    
    if(BB == 0):
        src_ply_file = src_path + "/filtered" 
        gt_json_file = gt_path + "/filtered" 
    else:
        src_ply_file = src_path + "/bounding_box/BB_" + str(BB) 
        gt_json_file = gt_path + "/bounding_box/BB_" + str(BB)  
    
    src_ply_file = src_ply_file + "/" + object_name + "_" + str(scan_nmb)
    gt_json_file = gt_json_file + "/" + object_name + "_" + str(scan_nmb)
    
    if(BB != 0):
        src_ply_file = src_ply_file + "_BB_" + str(BB) 
        gt_json_file = gt_json_file + "_BB_" + str(BB) 
        
    src_ply_file = src_ply_file + "_Source.ply"
    gt_json_file = gt_json_file + ".json" 
    tmp_stl_file = cad_path + "/" + object_name + ".stl"
    
    # print(src_ply_file)
    # print(gt_json_file)
    # print(tmp_stl_file)
    
    if(os.path.isfile(src_ply_file) and os.path.isfile(gt_json_file) and os.path.isfile(tmp_stl_file)):
        
        """
        Initializations
        ---------------
        Turn files into Open3D PointCloud Object
        """

        sampling_ratio = read_json(gt_json_file, "sampling_density")
        scale = read_json(gt_json_file, "scale")

        source_pointcloud = o3d.io.read_point_cloud(src_ply_file)
        template_stl = o3d.io.read_triangle_mesh(tmp_stl_file)  

        nmb_source_points = len(np.asarray(source_pointcloud.points))
        template_pointcloud = prepare_stl(template_stl, nmb_source_points, sampling_ratio, scale)

        """
        Ground Truth Estimation
        ---------------
        Estimate ground truth transformation, applied on template --> source
        """

        transformation = read_json(gt_json_file, "transformation")

        ground_truth_estimation(transformation, source_pointcloud, template_pointcloud)
        
    else:
        print("This combination does not exist. Check for spelling errors.")
    
    return

"""
=============================================================================
-----------------------------EXECUTE CODE HERE-------------------------------
=============================================================================
"""

visualise_ground_truth("Pendulum",2,BB=1.4)