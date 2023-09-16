"""
=============================================================================
-----------------------------------CREDITS-----------------------------------
=============================================================================

PointNetLK/RPMNet/PRNet Code by vinits5 as part of the Learning3D library 
Link: https://github.com/vinits5/learning3d#use-your-own-data

Changes/additions by Menthy Denayer (2023)

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#Array related operations
import torch
import time


"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""


def test_one_epoch(source, template, device, model, algo):
    """
    Extract rotation matrix and translation vector from tensor
    
    Parameters
    ----------
    tensor              : Nx4x4 torch tensor            // Transformation matrix T
    
    Returns
    ----------
    reg_time            : float                         // Time for the registration process
    """
    
    with torch.no_grad():
        model.eval()

        "--------------------assigning device--------------------"
        start = time.time()
        template = template.to(device)
        source = source.to(device)

        "--------------------performing registration--------------------"
        start = time.time()
        
        if(algo == "PointNetLK"):
            output = model(template, source)
        elif(algo == "RPMNet"):
            output = model(template, source)
            
        reg_time = time.time() - start

        transformation = output['est_T'].detach().cpu().numpy()[0]
        
    return transformation, reg_time