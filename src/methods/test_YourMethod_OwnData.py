"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test YourMethod OwnData

Run YourMethod, with given parameters, for given inputs

Inputs:
    - source point cloud
    - template point cloud
    - voxel size

Output:
    - found transformation
    - registration time

Credits: 
    YourMethod

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import time

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def test_one_epoch(source, template):
    """
    Main to run test of own registration method
    
    Parameters
    ----------
    source                  : Open3D Point Cloud        : source point cloud
    template                : Open3D Point Cloud        : template point cloud
    
    Returns
    ----------
    transformation          : 4x4 numpy array           : registration found transformation
    registration time       : float                     : registration time
    """

    start = time.time()
        
    """ ADD REGISTRATION CODE HERE """
        
    estimated_transformation = []; "Function of template_ and source_"
        
    """ ------------------------- """
        
    reg_time = time.time() - start
    
    return estimated_transformation, reg_time


def main(source, template):
    """
    Main to run test of own registration method
    
    Parameters
    ----------
    source                  : Open3D Point Cloud        : source point cloud
    template                : Open3D Point Cloud        : template point cloud
    
    Returns
    ----------
    transformation          : 4x4 numpy array           : registration found transformation
    registration time       : float                     : registration time
    """
    
    # Execute registration
    transformation, registration_time = test_one_epoch(source, template)
    
    return transformation, registration_time

if __name__ == '__main__':
    main()