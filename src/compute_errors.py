"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



compute errors

Compute the different metrics based on the template, source, transformed source,
ground truth & estimated transformation.

Inputs:
    - .json result file

Output:
    - Relative Errors
    - Root Mean Square Errors
    - Mean Absolute Errors
    - Recall
    - R2 
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
import os
from tqdm import tqdm
import time
import torch
import numpy as np
from .errors import Errors
from .data_manager import preprocess_data, pointcloud_to_torch, read_json

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""

def compute_mean_variance(error_list):
    """
    Compute mean and variance for given list of errors
    
    Parameters
    ----------
    error_list          : 8xN numpy array           : list of computed errors
    
    Returns
    ----------
    mean                : 8x1 numpy array           : means for every error
    variance            : 8x1 numpy array           : variance for every error             
    """
    
    mean = np.mean(error_list,1)
    variance = np.var(error_list,1)
    return mean, variance

def symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, transfo_tensor, recall_lim):
    """
    Compute smallest error in the case of symmetric solutions
    
    Parameters
    ----------
    templ_tensor        : Nx6 torch tensor          : tensor of template points
    src_tensor          : Nx6 torch tensor          : tensor of source points
    gt_symm_tensor      : Mx4x4 torch tensor        : tensor of all ground truth solutions
    transfo_tensor      : Nx6 torch tensor          : tensor of transformed template points
    recall_lim          : float                     : limit for recall metric
    
    Returns
    ----------
    errors_min          : 8x1 numpy array           : list of best errors (closest to symmetric solution)
    mean_error_time     : float                     : time to compute errors         
    """
    
    # Go over all possible ground truth solutions and find best one
    # Initialize error with max possible value (180°)
    max_recall = 0
    nmb_sol = gt_symm_tensor.shape[0]
    error_time = 0
    errors_min = np.ones((8,1))*-1
    
    for i in range(nmb_sol):
        # Compute errors (+ time)
        # Create error module
        
        gt_sol = gt_symm_tensor[i,:,:].expand(1,4,4)
        error_class = Errors(templ_tensor[:,:,0:3],src_tensor[:,:,0:3],gt_sol,transfo_tensor,recall_lim)
        
        # Compute errors
        start = time.time()
        errors = error_class()
        error_time = time.time()-start + error_time
        
        # Display errors
        # error_class.display(errors)
        
        # Save errors if smallest one
        if(errors[6,0] > max_recall):
            errors_min = errors
            max_recall = errors[6,0]
            
    mean_error_time = error_time/nmb_sol
    # Display errors
    # error_class.display(errors_min)
    return errors_min, mean_error_time 

def compute_errors(result_json_file, recall_lim = 0.01): 
    """
    Compute all errors and find best error in case of symmetric solutions
    
    Parameters
    ----------
    result_json_file    : string                    : location of result .json file
    recall_lim          : float                     : limit for recall metric
    
    Returns
    ----------
    errors_min          : 8x1 numpy array           : list of (best) errors (closest to symmetric solution)       
    """
    
    # Load data
    template_pointcloud, source_pointcloud, transformation, json_info = \
        preprocess_data(result_json_file, voxel_size = 0)
        
    transfo_tensor = torch.tensor(json_info["estimated_transformation"])
    transfo_tensor = transfo_tensor.expand(1,4,4)
    gt_symm_tensor = torch.tensor(transformation)    
    
    templ_tensor = pointcloud_to_torch(template_pointcloud)
    src_tensor = pointcloud_to_torch(source_pointcloud)
    
    errors, mean_error_time = symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, 
                                                       transfo_tensor, recall_lim)
    
    # print(":: Mean error time is: ", mean_error_time)
    
    return errors


def compute_errors_batch(result_json_dir, recall_lim, R2_lim, MRAE_lim):
    """
    Compute all errors for a batch of .json files. Results are averaged per object,
    over all scans. Failure cases are not counted in averaging but saved as failure cases.
    
    Parameters
    ----------
    result_json_file    : string                    : location of result .json file
    recall_lim          : float                     : limit for recall metric
    R2_lim              : float                     : R2 limit used to determine failure cases
    MRAE_lim            : float                     : MRAE limit used to determine failure cases
    
    Returns
    ----------
    objects_mean_dict   : dictionary                : dictionary of mean errors per object
    objects_var_dict    : dictionary                : dictionary of variances per object
    scan_failure_dict   : dictionary                : number failure cases per scan per object
    failure_cases_list  : dictionary                : percentage of total failure cases per object
    """
    
    scan_failure_dict = {}
    object_scan_dict = {}
    object_errors_dict = {}
    
    for _,result_json_file in enumerate(tqdm(os.listdir(result_json_dir))):
        
        result_json_path = result_json_dir + "/" + result_json_file
        
        errors = compute_errors(result_json_path, recall_lim)
        
        # Process errors
        object_name = read_json(result_json_path,"name")
        scan_nmb = read_json(result_json_path, "scan")
        object_scan_dict = update_dictionary(object_name, scan_nmb, object_scan_dict)
        
        if(error_criteria(errors[7,0], errors[0,0], R2_lim, MRAE_lim)):
            
            scan_failure_dict = update_dictionary(object_name, scan_nmb, scan_failure_dict)

        else:
            if(object_name in object_errors_dict):
                object_errors_dict[object_name] = np.append(object_errors_dict[object_name], errors, 1)
            else:
                object_errors_dict[object_name] = np.zeros((8,1))
        
    failure_cases_list = compute_failure_cases(object_scan_dict, scan_failure_dict)
    
    objects_mean_dict = {}
    objects_var_dict = {}
    
    for object_name in object_errors_dict:
        objects_mean_dict[object_name], objects_var_dict[object_name] = \
            compute_mean_variance(object_errors_dict[object_name])
    
    return objects_mean_dict, objects_var_dict, scan_failure_dict, failure_cases_list

def update_dictionary(object_name, scan_nmb, dictionary):
    """
    Define criteria to consider case a failure
    
    Parameters
    ----------
    object_name                 : string                : name of the object
    scan_nmb                    : int                   : scan number of object
    dictionary                  : dictionary            : dictionary of objects and scans
    
    Returns
    ----------
    new_dictionary              : dictionary            : updated dictionary 
    """
    
    new_dictionary = dictionary
    
    if(object_name in new_dictionary):
        if(scan_nmb in new_dictionary[object_name]):
            new_dictionary[object_name][scan_nmb] = new_dictionary[object_name][scan_nmb] + 1
        else:
            new_dictionary[object_name][scan_nmb] = 1
    else:
        new_dictionary[object_name] = {}
    
    return new_dictionary

def compute_failure_cases(object_scan_dict, scan_failure_dict):
    """
    Compute total percentage of failure cases per object
    
    Parameters
    ----------
    object_scan_dict            : dictionary                : number of scans per object
    scan_failure_dict           : dictionary                : number of failures per scan per object
    
    Returns
    ----------
    failure_cases_list          : dictionary                : total percentage of failure cases per object
    """
    
    failure_cases_dict = {}
    
    for object_name in object_scan_dict:
        scan_list = object_scan_dict[object_name]
        
        number_scans_total = 0
        number_failures_total = 0
        for scan in scan_list:
            number_scans_subtotal = scan_list[scan]
            number_scans_total = number_scans_total + number_scans_subtotal
            
            if(scan in scan_failure_dict[object_name]):
                number_failures_total = number_failures_total + scan_failure_dict[object_name][scan]
        
        failure_cases = number_failures_total/number_scans_total
        failure_cases_dict[object_name] = failure_cases*100
    
    return failure_cases_dict

def error_criteria(R2_value, MRAE_value, R2_lim, MRAE_lim):
    """
    Define criteria to consider case a failure
    
    Parameters
    ----------
    R2_value            : float             : computed value for R2 metric
    MRAE_value          : float             : computed value for MRAE metric
    R2_lim              : float             : R2 for identifying failure case
    MRAE_lim            : float             : MRAE for identifying failure case
    
    Returns
    ----------
    boolean to signal failure/not
    """
    
    if(R2_value < R2_lim or abs(MRAE_value) > MRAE_lim):
        return True
    else:
        return False