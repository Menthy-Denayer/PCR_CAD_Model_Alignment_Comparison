"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test RANSAC OwnData

Run RANSAC, with given parameters, for given inputs

Inputs:
    - source point cloud
    - template point cloud

Output:
    - found transformation
    - registration time

Credits: 
    RANSAC as part of the Open3D library

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
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def prepare_dataset(template, source, voxel_size):
    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(template, voxel_size)
    
    return source_fpfh, target_fpfh

#Info on how this works: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    Max_it = 1000000
    #Info: http://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(Max_it, 0.999))
    return result

def test_one_epoch(source, template, voxel_size):
    # Parameters based on "Choi_Robust_Reconstruction_of_2015_CVPR_paper"
        
    source_fpfh, target_fpfh = prepare_dataset(template, source, voxel_size)
    

    start = time.time()
    result_ransac = execute_global_registration(source, template,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    registration_time = time.time() - start
    transformation = result_ransac.transformation
    
    return transformation, registration_time
        
def main(source, template, voxel_size):
    
    transformation, reg_time = test_one_epoch(source, template, voxel_size)
    
    return transformation, reg_time
if __name__ == '__main__':
    main()