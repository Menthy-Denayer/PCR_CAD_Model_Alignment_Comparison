"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test FGR OwnData

Run FGR, with given parameters, for given inputs

Inputs:
    - source point cloud
    - template point cloud
    - voxel size

Output:
    - found transformation
    - registration time

Credits: 
    FGR as part of the Open3D library

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General Imports
import open3d as o3d
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
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def prepare_dataset(template, source, voxel_size):
    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(template, voxel_size)
    
    return source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
            # % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def test_one_epoch(source, template, voxel_size):
    
    source_fpfh, target_fpfh = prepare_dataset(template, source, voxel_size)
    
    start = time.time()
    result_fast = execute_fast_global_registration(source, template,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    
    reg_time = time.time() - start
    transformation = result_fast.transformation     
    
    return transformation, reg_time

def main(source, template, voxel_size):
    
    # Execute registration
    transformation, reg_time = test_one_epoch(source, template, voxel_size)
    
    return transformation, reg_time

if __name__ == '__main__':
    main()