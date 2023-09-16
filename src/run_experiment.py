"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



run experiment

Runs single experiment

Inputs:
    - Object name
    - Scan nmb
    - BB
    - PCR method
    - VS
    - Zero Mean
    
Output:
    - .json with results
"""

"""
=============================================================================
----------------------------------IMPORTS------------------------------------
=============================================================================
"""

from .data_manager import preprocess_data, pointcloud_to_torch, write_result_json, read_json

"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

torch_methods ={"GO-ICP", "PointNetLK", "RPMNet", "ROPNet"}

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""


def run_method(template, source, pcr_method, MSEThresh, trimFraction, voxel_size):
    """
    Run registration with chosen method
    
    Parameters
    ----------
    template                : Open3D Point Cloud        : template point cloud
    source                  : Open3D Point Cloud        : source point cloud
    pcr_metod               : string                    : registration method to use
    MSEThresh               : float                     : GO-ICP parameter
    trimFraction            : float                     : GO-ICP parameter
    voxel_size              : float                     : down sample point cloud
    
    Returns
    -------
    transformation          : 4x4 numpy array           : found transformation
    registration_time       : float                     : computation time for registration
    """
    
    registration_parameters = {"method": pcr_method}
    
    if pcr_method in torch_methods:
        source_tensor = pointcloud_to_torch(source)
        template_tensor = pointcloud_to_torch(template)
    
    if pcr_method == "ICP":
        
        # Import ICP
        import src.methods.test_ICP_OwnData as ICP
        
        transformation, registration_time = ICP.main(source, template, voxel_size)
        
    elif pcr_method == "GO-ICP":
        
        # Import GO-ICP 
        import src.methods.test_GOICP_OwnData as GO_ICP
        
        # GO-ICP specific parameters
        registration_parameters["MSE Threshold"] = MSEThresh
        registration_parameters["Trim Factor"] = trimFraction
        
        transformation, registration_time = GO_ICP.main(source_tensor[:,:,0:3], template_tensor[:,:,0:3])

    elif pcr_method == "RPMNet":
        
        # Import RPMNet
        import src.methods.test_RPMNet_OwnData as RPMNet
        
        transformation, registration_time = RPMNet.main(source_tensor, template_tensor)
        
    elif pcr_method == "FGR":
        
        # Import FGR
        import src.methods.test_FGR_OwnData as FGR
        
        transformation, registration_time = FGR.main(source, template, voxel_size)
        
    elif pcr_method == "RANSAC":
        
        # Import RANSAC
        import src.methods.test_RANSAC_OwnData as RANSAC
        
        transformation, registration_time = RANSAC.main(source, template, voxel_size)
        
    elif pcr_method == "PointNetLK": 
        
        # Import PointNetLK
        import src.methods.test_PointNetLK_OwnData as PointNetLK
        
        transformation, registration_time = PointNetLK.main(source_tensor[:,:,0:3], template_tensor[:,:,0:3])
        
    elif pcr_method == "ROPNet":
        
        # Import ROPNet
        import src.methods.test_ROPNet_OwnData as ROPNet
        
        transformation, registration_time = ROPNet.main(source_tensor, template_tensor)
    
    return transformation, registration_time, registration_parameters

def run_one_experiment(gt_json_file, pcr_method, voxel_size = 0, zero_mean = False,
                       MSEThresh=0.00001, trimFraction=0.0001, 
                       experiment_name = "experiment"):
    """
    Run a single registration experiment
    
    Parameters
    ----------
    gt_json_file            : string                : path to ground truth .json file
    pcr_method              : string                : registration method to use
    voxel_size              : float                 : down sample point cloud
    zero_mean               : boolean               : center point cloud
    MSEThresh               : float                 : GO-ICP parameter
    trimFraction            : float                 : GO-ICP parameter
    experiment_name         : string                : result folder name
    """
    
    # Load point clouds
    template_pointcloud, source_pointcloud, transformation, json_info = \
        preprocess_data(gt_json_file, voxel_size, zero_mean)
    
    # Run method
    estimated_transformation, registration_time, registration_parameters = \
        run_method(template_pointcloud, source_pointcloud, pcr_method,
                   MSEThresh, trimFraction, voxel_size)
    
    # Add specific registration parameters
    registration_parameters["voxel_size"] = voxel_size
    registration_parameters["centered"] = zero_mean
    json_info["registration_parameters"] = registration_parameters
    
    # Save result
    output_folder = pcr_method + "/" + experiment_name
    write_result_json(output_folder, transformation, estimated_transformation, 
                      registration_time, json_info)
    return

def run_refinement(json_file, voxel_size = 0, experiment_name = "refinement"):
    """
    Apply refinement to result from .json file
    
    Parameters
    ----------
    json_file               : string                : path to result .json file
    voxel_size              : float                 : parameter used for refinement (not down sampling)
    experiment_name         : string                : result folder name
    """
    
    import src.methods.test_ICP_OwnData as ICP
    
    # Read data from previous registration step
    json_info = read_json(json_file)
    registration_parameters = json_info["registration_parameters"]
    zero_mean = registration_parameters["centered"]
    ground_truth = json_info["transformation"]
    transformation = json_info["estimated_transformation"]
    initial_pcr_method = registration_parameters["method"]
    
    # Load data as point clouds
    # Voxel size at 0 since refinement works on original point clouds
    template_pointcloud, source_pointcloud,_,_ = \
        preprocess_data(json_file, voxel_size = 0, zero_mean = zero_mean)
    
    # Run method
    estimated_transformation, registration_time = \
        ICP.main(source_pointcloud, template_pointcloud, voxel_size, transformation)
    
    # Add specific registration parameters
    registration_parameters = {"method" : "ICP refinement"}
    registration_parameters["voxel_size"] = voxel_size
    registration_parameters["centered"] = zero_mean
    json_info["registration_parameters"] = registration_parameters
    
    # Save result
    output_folder = initial_pcr_method + "_refined/" + experiment_name
    write_result_json(output_folder, ground_truth, estimated_transformation, 
                      registration_time, json_info)
    return