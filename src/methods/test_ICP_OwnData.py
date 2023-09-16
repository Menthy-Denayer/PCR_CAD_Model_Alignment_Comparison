"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test ICP OwnData

Run ICP, with given parameters, for given inputs

Inputs:
    - source point cloud
    - template point cloud
    - voxel size
    - initial transformation estimate

Output:
    - found transformation
    - registration time

Credits: 
    ICP as part of the Open3D library

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General Imports
import open3d as o3d
import numpy as np
import time
        

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""
  
def preprocess_point_cloud(pcd, voxel_size):

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def prepare_dataset(template, source, voxel_size):

    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(template, voxel_size)
    
    return source_fpfh, target_fpfh

def refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, transformation):
    distance_threshold = voxel_size * 0.4
    Max_it = 100000

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=Max_it)) #o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    return result

def test_one_epoch(source, template, voxel_size, transformation):

    if(not transformation):
        transformation = np.identity(4)
    
    source_fpfh, target_fpfh = prepare_dataset(template, source, voxel_size)
    
    start = time.time()
    result_icp = refine_registration(source, template, source_fpfh, 
                                     target_fpfh, voxel_size, transformation)
    
    registration_time = (time.time() - start)
    transformation = result_icp.transformation
        
    return transformation, registration_time

def main(source, template, voxel_size, transformation = None):
    
    # Execute registration
    transformation, reg_time = test_one_epoch(source, template, voxel_size, transformation)

    return transformation, reg_time

if __name__ == '__main__':
    main()