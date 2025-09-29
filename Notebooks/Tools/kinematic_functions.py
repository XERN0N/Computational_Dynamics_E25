from jax import jit, vmap, jacfwd, grad
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import timeit

@jit
def A_jit(rotation_angle):
    """
    Parameters
    -------------
    Input: scalar

    Returns
    -------------
    Output: jnp.array (2,2)

    This jitted function takes in a scalar angle [rad] and returns the corresponding 
    2D rotation matrix A.
    """
    
    A_matrix = jnp.array(([jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
                         [jnp.sin(rotation_angle), jnp.cos(rotation_angle)]))
    
    return A_matrix

@jit
def B_jit(rotation_angle):
    """
    Parameters
    -------------
    Input: scalar

    Returns
    -------------
    Output: jnp.array (2,2)

    This jitted function takes in a scalar angle [rad] and returns the corresponding 
    2D rotation matrix B that is A differentiated.
    """
    B_matrix = jnp.array(([-jnp.sin(rotation_angle), -jnp.cos(rotation_angle)],
                         [jnp.cos(rotation_angle), -jnp.sin(rotation_angle)]))

    return B_matrix

@jit
def phi_abs(global_coord_1, local_point_1, constraint):
    """
    Parameters
    -------------
    global_coord_1: ndarray (3,)

    local_point_1: ndarray (2,)

    constraint: ndarray (2,)

    Returns
    -------------
    absolute_constraint: ndarray (2,)


    This function takes in coordinates and angles of a body.
    It returns the function value of the absolute constraint.
    If the absolute constraint is satisfied then this function should return 0.
    """
    r_vec_1 = global_coord_1[0:2]
    phi_1 = global_coord_1[2]
    absolute_constraint = (r_vec_1 + A_jit(phi_1) @ local_point_1) - constraint

    return absolute_constraint

@jit
def phi_relative(global_coord_1, local_point_1, global_coord_2, local_point_2, constraint):
    """
    Parameters
    -------------
    global_coord_1: ndarray (3,)

    local_point_1: ndarray (2,)

    global_coord_2: ndarray (3,)

    local_point_2: ndarray (2,)

    constraint: ndarray (2,)

    Returns
    -------------
    relative_constraint: ndarray (2,)


    This function takes in coordinates and angles of a body.
    It returns the function value of the relative constraint.
    If the relative constraint is satisfied then this function should return 0.
    """
    r_vec_1 = global_coord_1[0:2]
    phi_1 = global_coord_1[2]
    r_vec_2 = global_coord_2[0:2]
    phi_2 = global_coord_2[2]

    relative_constraint = (r_vec_1 + A_jit(phi_1) @ local_point_1) 
    - (r_vec_2 + A_jit(phi_2) @ local_point_2) - constraint

    return relative_constraint

@jit
def phi_driver(input_function, time_value):
    """
    Parameters
    -------------
    input_function: lambda-function (t).
        The function that defines the drivers behavior
    time_value: float
        The point of time for the function to be evaluated
    global_coord: 1D ndarray
        The global coordinates 

    Returns
    -------------
    driving_constraint: float
        The results of the constraint after being evaluated


    This function takes in an arbitrary lambda function as a function of either time and global coordinates or just time. 
    """
    
    driving_constraint = input_function(time_value)

    return driving_constraint

def newton_rhapson(function, initial_guess, jacobian, time, rtol=1e-3, atol=1e-3, max_iter=500):
    pass

if __name__ == "__main__":

    
    
    
    
    
    
    #omega = 1.5
    #driver = lambda t: 0 - omega*t
    """ 
    time_steps = np.arange(0, 10, 0.1)
    result = np.empty_like(time_steps)
    for index, time in enumerate(time_steps):
        #result[index] = driver(time)
        result[index] = np.linalg.norm(B_jit(time) - B_test(time))"""
    
    #print(result)
    #test = B_jit(jnp.pi/2).block_until_ready()
    #print(test)
    #test = A_jit(jnp.pi/2).block_until_ready()
    #print(timeit.timeit(lambda: A_jit(jnp.pi/2).block_until_ready(), number=1000000))
    #print(timeit.timeit(test))
    #print(timeit.timeit("A_jit(jnp.pi/2).blockuntil", globals=globals()))