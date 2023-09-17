import os

from src.run_experiment import run_one_experiment, run_refinement, run_experiment_batch, run_refinement_batch
from src.visualizer import visualise_result, visualise_ground_truth
from src.ground_truth_accuracy import evaluate_ground_truth, evaluate_ground_truth_batch
from src.compute_errors import compute_errors, compute_errors_batch

BASE_DIR = os.getcwd()

"""
=============================================================================
--------------------VISUALIZE GROUND TRUTH TRANSFORMATION--------------------
=============================================================================
"""

# gt_json_file = BASE_DIR + "/datasets/ground_truth/bounding_box/BB_1.4/Base-Top_Plate_1_BB_1.4.json"

# visualise_ground_truth(gt_json_file, voxel_size = 0.002, zero_mean = True)


"""
=============================================================================
---------------------ASSESS GROUND TRUTH TRANSFORMATION----------------------
=============================================================================
"""

# gt_json_dir = BASE_DIR + "/datasets/ground_truth/filtered"
# gt_json_file = gt_json_dir + "/Shaft_New_4.json"

# errors = evaluate_ground_truth(gt_json_file, voxel_size=0.01)

# mean_errors, variance, error_list = evaluate_ground_truth_batch(gt_json_dir, voxel_size = 0.01)


"""
=============================================================================
-------------------------------RUN EXPERIMENT--------------------------------
=============================================================================
"""

gt_json_dir = BASE_DIR + "/datasets/ground_truth/filtered"
gt_json_file = gt_json_dir + "/Base-Top_Plate_2.json"

# run_one_experiment(gt_json_file, pcr_method = "GO-ICP", voxel_size = 0,
#                     MSEThresh=0.00001, zero_mean = True)

# run_experiment_batch(gt_json_dir, "RANSAC", voxel_size = 0.005, zero_mean = True, 
#                       nmb_it = 2, experiment_name = "VS_0.005")

"""
=============================================================================
------------------------------VISUALIZE RESULT-------------------------------
=============================================================================
"""
# result_json_file = os.getcwd() + "/results/RANSAC/experiment/Base-Top_Plate_2_result.json"

# visualise_result(result_json_file)

"""
=============================================================================
-------------------------------REFINE RESULT---------------------------------
=============================================================================
"""

# result_json_dir = BASE_DIR + "/results/RANSAC/VS_0.005"
# result_json_file = result_json_dir + "/Base-Top_Plate_1_result_it_1.json"

# run_refinement(result_json_file, voxel_size = 0.1)

# run_refinement_batch(result_json_dir, voxel_size = 0.1, experiment_name = "VS_0.1")

# refined_json_dir = BASE_DIR + "/results/RANSAC_refined/VS_0.1/"
# refined_json_file = refined_json_dir + "/Base-Top_Plate_2_it_1_result.json"

# visualise_result(refined_json_file, voxel_size = 0)

"""
=============================================================================
-------------------------------COMPUTE ERRORS---------------------------------
=============================================================================

errors are returned in the following order:
    - Angular MRE
    - Translational MRE
    - Angular RMSE
    - Translational RMSE
    - Angular MAE
    - Translational MAE
    - Recall
    - R2 

"""

result_json_dir = BASE_DIR + "/results/RANSAC/VS_0.005"
result_json_file = result_json_dir + "/Round-Peg_2_it_1_result.json"

visualise_result(result_json_file, voxel_size = 0.003)

# visualise_ground_truth(result_json_file, voxel_size = 0.003, zero_mean = True)

# errors = compute_errors(result_json_file, 0.01)

mean_errors, variance, failure_list, failure_cases_list = \
    compute_errors_batch(result_json_dir, 0.01, 0, 120)
