# PCR CAD Model Alignment Comparison

GitHub page aimed at comparing Point Cloud Registration (PCR) techniques when performing CAD model alignment. 

# Installation
1. Download the datasets folder, source code and main file
2. Install the required packages (to run the registration methods)
3. Use the main.py file to:
  -  Visualize the ground truth transformation
  -  Assess the quality of the ground truth estimation using ICP refinement
  -  Compute the result of applying any registration method included
  -  Visualize the result of the registration
  -  Refine the registration using ICP
  -  Compute metrics to assess the registration performance 

# Packages
A list of required Python packages is provided in the _packages.txt_ file. Packages to run the main code are: PyTorch, Open3D, tqdm (progress-bar), tabulate (errors summary), (Spyder-kernels)

For the registration methods (GO-ICP, RPMNet, PointNetLK & ROPNet), the packages can be found on the author's source code page.

# Acknowledgements 
Many thanks to the authors of the used registration methods!
- [Open3D](http://www.open3d.org/) (RANSAC, ICP, FGR)
- [GO-ICP](https://github.com/aalavandhaann/go-icp_cython.git)
- [Learning3D](https://github.com/vinits5/learning3d.git) (PointNetLK, RPMNet)
- [ROPNet](https://github.com/zhulf0804/ROPNet.git)

# Project Status
- [x] Added primary source code
- [x] Added used datasets
- [x] Add list of required packages to run created source code
