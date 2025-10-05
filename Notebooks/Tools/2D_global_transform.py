import numpy as np
import numpy.typing as npt
from jax import jit
import jax.numpy as jnp
import jax.typing as jnpt
from functools import partial



class body_2D:

    def __init__(self, x_function=None, y_function=None, time_vector: jnpt.ArrayLike = None, local_points=None):
        self.x_func = x_function
        self.y_func = y_function
        self.time_vec = time_vector
        self.local_point_vec = local_points
        print("THIS CLASS IS DEPRECTED, PLEASE USE SOMETHING ELSE")

    def set_local_point(self, local_point, operation="add"):
        """ 
        This method adds, removes or resets and adds local points to the body.
        The default is add.
        """

        if operation.strip().casefold() == "add":
            print("test")
        elif operation.strip().casefold() == "remove":
            self.local_point_vec
        elif operation.strip().casefold() == "clear":
            if local_point is None:
                self.local_point_vec = jnp.array([[0,0]])
            else:
                self.local_point_vec = local_point
        else:
            print("No points have been set. Invalid operation input")


    def set_xy_function(self, new_x_function, new_y_function):
        """ 
        This method is meant to set or update the x and y-functions to be used for the 2D body.
        It will overwrite the existing function.
        The new functions provided should only be dependant on time.
        """
        if new_x_function and new_y_function is not None:
            self.x_func = new_x_function
            self.y_func = new_y_function
            print("The new x and y functions have been set")
        elif new_x_function or new_y_function is not None:
            if new_y_function is not None:
                self.y_func = new_y_function
                print("The new y function has been set")
            elif new_x_function is not None:
                self.x_func = new_x_function
                print("The new x function has been set")
        else:
            print("No input given and no functions have been set. run set_xy_function() with one or two arguments")

    @jit
    def get_orientation_2D(self, angle):
        """
        This method works for 2D planar dynamics models.
        It takes a scalar float and returns a 2x2 jnp.array with the corresponding rotation matrix (A)
        
        Input: Scalar float

        Return: jnp.array (2, 2) of floats
        """
        #Check for valid input
        assert isinstance(angle, (float, int))

        #Calculate using [[c(phi), -s(phi)], [s(phi), c(phi)]]
        orientation_matrix = jnp.squeeze(jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]))

        return orientation_matrix

    #@partial(jit, static_argnames=["x_func", "y_func"] )
    def get_translation_2D(self, timevalue: jnpt.DTypeLike) -> jnpt.ArrayLike:
        """
        This method works for 2D planar dynamics models
        It takes two functions of single variables as input and the corresponding variable value.
        it returns the translation vector for that instance.

        Input: (f(t), g(t), t) as scalar functions with scalar float value input.

        Return: jnp.array (2,1) of float
        """
        #Check for valid input
        #assert len(timevalue.shape) == 0
        #assert isinstance(timevalue, (float, int, np.ndarray)) deactivated because of tracers
        
        #ONLY CHECKS FOR NON-STANDARD CONSTRUCTOR VALUES - it still allows for illegal input
        #assert not isinstance(self.x_func, None)
        #assert not isinstance(self.y_func, None)
        
        x_value_function = self.x_func
        y_value_function = self.y_func

        #Call x function and y function to get coordinates to variable input (timestep)
        translation_vector = jnp.array([[x_value_function(timevalue), y_value_function(timevalue)]])

        return translation_vector

    
    
    """ 
    def global_transformation_2D(translation_matrix, rotational_angle, local_point_matrix):
        
        #2x1xN + 2x2xN x 2x1xN
        #orientation_matrix @ local_point_matrix
        pass
        
        orientation_matrix = jnp.squeeze(jnp.array([[jnp.cos(rotational_angle), -jnp.sin(rotational_angle)], [jnp.sin(rotational_angle), jnp.cos(rotational_angle)]]))
        local_point = jnp.atleast_1d(local_point)
        
        global_vector = jnp.sum(translation_vector, orientation_matrix @ local_point)
        #return global_vector


    #global_transformation_2D_jit = jit(global_transformation_2D, static_argnames=("rotational_angle", "local_point"))

    """

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    h = 2
    omega = np.pi/2
    t = np.arange(0, 6*np.pi, 0.01)
    body_orientation = t*omega
    s_p = np.array([1, 2]).T
    translation_array = np.array([h*np.cos(body_orientation), 2*h*np.sin(body_orientation)]).T

    kinematic_path_points = np.empty((len(t), 2))
""" 
    for index, timepoints in enumerate(t):
        kinematic_path_points[index] = global_transformation_2D(translation_array[index, :], body_orientation[index], s_p)


    #print(kinematic_path_points)
    combined = kinematic_path_points[:,0] + kinematic_path_points[:,1]
    
    plt.plot(kinematic_path_points[:,0], kinematic_path_points[:,1])
    plt.show()
 """