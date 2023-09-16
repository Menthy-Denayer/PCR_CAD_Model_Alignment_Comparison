import os


from src.run_experiment import run_one_experiment, run_refinement
from src.visualizer import visualise_result

BASE_DIR = os.getcwd()

"""
=============================================================================
--------------------VISUALIZE GROUND TRUTH TRANSFORMATION--------------------
=============================================================================
"""

# gt_json_file = BASE_DIR + "/datasets/ground_truth/bounding_box/BB_1.4/Base-Top_Plate_1_BB_1.4.json"

# visualise_ground_truth(gt_json_file, voxel_size = 0.005)

"""
=============================================================================
-------------------------------RUN EXPERIMENT--------------------------------
=============================================================================
"""

gt_json_file = BASE_DIR + "/datasets/ground_truth/filtered/Base-Top_Plate_2.json"

run_one_experiment(gt_json_file, pcr_method = "RANSAC", voxel_size = 0.005,
                    MSEThresh=0.1, zero_mean = True)

"""
=============================================================================
------------------------------VISUALIZE RESULT-------------------------------
=============================================================================
"""
result_json_file = os.getcwd() + "/results/RANSAC/experiment/Base-Top_Plate_2_result.json"

visualise_result(result_json_file)

"""
=============================================================================
-------------------------------REFINE RESULT---------------------------------
=============================================================================
"""

run_refinement(result_json_file, voxel_size = 0.1)

refined_result_json_file = os.getcwd() + "/results/RANSAC_refined/refinement/Base-Top_Plate_2_result.json"

visualise_result(refined_result_json_file, voxel_size = 0)