import torch
import math
def learning_rate_schedule(t,alpha_max:float,alpha_min:float,Tw:int,Tc:int):
    """
    Calculate the learning rate alpha_t according to the given scheduler.

    Args:
        t (int): The current step or time step.
        alpha_max (float): Maximum value of the learning rate.
        alpha_min (float): Minimum value of the learning rate.
        Tw (int): Warm-up duration in number of steps.
        Tc (int): Constant duration after warm-up.

    Returns:
        float: The calculated learning rate alphat_t.
    """ 
    alpha_t=0.0
    
    if t < Tw:
        alpha_t = alpha_max * (t / Tw)
    elif(Tw <= t and t<=Tc):
        alpha_t = alpha_min + (0.5*(1+math.cos((t-Tw)*math.pi/(Tc-Tw)))*(alpha_max-alpha_min))
    else:#t>tW
        alpha_t = alpha_min
    
    return alpha_t