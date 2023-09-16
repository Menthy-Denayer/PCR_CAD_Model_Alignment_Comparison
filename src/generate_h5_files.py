"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



generate h5 files

Creates .hdf5 files from datasets

Inputs:
    - CAD models (.stl)
    - point cloud scans (.ply)
    - ground truths (.json)
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import open3d as o3d
from tqdm import tqdm
import numpy as np
import json
import h5py

"""
=============================================================================
----------------------------------VARIABLES----------------------------------
=============================================================================
"""

BASE_DIR = os.getcwd()
cad_path = BASE_DIR + "/datasets/CAD"

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

def write_h5(file_name, output_path, template, source, gt): 
    """
    Read key of json file, located in path

    Parameters
    ----------
    file_name           : string            : name of the created .hdf5 file
    output_path         : string            : path to the output folder
    template            : 1xNx6 numpy array : array of template points
    source              : 1xMx6 numpy array : array of source points
    gt                  : Px4x4 numpy array : array of all ground truth solutions
    """
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_file = output_path + "/" + file_name

    with h5py.File(output_file,"w") as data_file:
        data_file.create_dataset("template",data=template)
        data_file.create_dataset("source",data=source)
        data_file.create_dataset("transformation",data=np.expand_dims(gt[0,:,:],0))
        data_file.create_dataset("transformation_all",data=gt,chunks=True, maxshape=(None,4,4))
        
    return

def pointcloud_to_array(pointcloud):
    """
    Turn point cloud into array

    Parameters
    ----------
    pointcloud              : Open3D Point Cloud            : point cloud file
    
    Returns
    -------
    array                   : 1xNx6 numpy array             : point cloud array
    """
    
    points_array = np.asarray(pointcloud.points)
    normals_array = np.asarray(pointcloud.normals)
    
    array = np.concatenate((points_array,normals_array),1)
    array = np.expand_dims(array,0)
    return array

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


def prepare_data(src_loc, cad_loc, gt_loc):
    """
    Loads source, CAD and ground truth data from path
    
    Parameters
    ----------
    src_loc             : string                        : source .ply file location
    cad_loc             : string                        : CAD .stl file location
    gt_loc              : string                        : ground truth .json location
    
    Returns
    -------
    source_array        : 1xNx6 numpy array             : source array
    template_array      : 1xMx6 numpy array             : template array
    gt_array            : Px4x4 numpy array             : ground truth array
    """
    
    sampling_ratio = read_json(gt_loc, "sampling_density")
    scale = read_json(gt_loc, "scale")
    
    source_pointcloud = o3d.io.read_point_cloud(src_loc)
    template_stl = o3d.io.read_triangle_mesh(cad_loc)  

    nmb_source_points = len(np.asarray(source_pointcloud.points))
    template_pointcloud = prepare_stl(template_stl, nmb_source_points, sampling_ratio, scale)
    
    transformation = read_json(gt_loc, "transformation")
    
    source_array = pointcloud_to_array(source_pointcloud)
    template_array = pointcloud_to_array(template_pointcloud)
    gt_array = np.asarray(transformation)
    
    return source_array, template_array, gt_array


"""
=============================================================================
----------------------------EXECUTE CODE-------------------------------------
=============================================================================
"""

""" Turn filtered scans into .hdf5 """

gt_path = BASE_DIR +   "/datasets/ground_truth/filtered"
input_path = BASE_DIR + "/datasets/point_clouds/filtered"
output_path = BASE_DIR + "/input_files/filtered"


print(" ----------------------------------------------\n \
Creating .hdf5 files for filtered point clouds\n \
-----------------------------------------------")

for _,file_name in enumerate(tqdm(os.listdir(input_path))):
    
    if('_1_' in file_name):
        object_name = file_name.split('_1_')[0]
    
    cad_file = cad_path + "/" + object_name + ".stl"
    gt_file = gt_path + "/" + file_name.split('_Source')[0] + ".json"
    src_file = input_path + "/" + file_name

    source, template, gt = prepare_data(src_file, cad_file, gt_file)
    
    output_file = file_name.split('_Source')[0] + ".hdf5"
    write_h5(output_file, output_path, template, source, gt)

""" Turn BB scans into .hdf5 """

input_path = BASE_DIR + "/datasets/point_clouds/bounding_box"
output_path = BASE_DIR + "/input_files/bounding_box"
gt_path = BASE_DIR +   "/datasets/ground_truth/bounding_box"

print(" --------------------------------------------------\n \
Creating .hdf5 files for bounding box point clouds\n \
--------------------------------------------------")

for _,bounding_box in enumerate(tqdm(os.listdir(input_path))):
    
    subfolder_path = input_path + "/" + bounding_box

    for _,file_name in enumerate(tqdm(os.listdir(subfolder_path))):
        
        if('_1_' in file_name):
            object_name = file_name.split('_1_')[0]
        
        cad_file = cad_path + "/" + object_name + ".stl"
        gt_file = gt_path + "/" + bounding_box + "/" + file_name.split('_Source')[0] + ".json"
        src_file = subfolder_path + "/" + file_name

        source, template, gt = prepare_data(src_file, cad_file, gt_file)
        
        output_file = file_name.split('_Source')[0] + ".hdf5"
        write_h5(output_file, output_path + "/" + bounding_box, template, source, gt)