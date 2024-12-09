# Module of helper functions
import numpy as np
import inspect

def consume_fraction_of_error_budget(consumed_fraction, current_error_budget):
    # Takes in the current error budget and the amount to be consumed and outputs the amount consumed and the remaining error budget
    consumed_error_budget = consumed_fraction * current_error_budget
    remaining_error_budget = current_error_budget - consumed_error_budget
    return consumed_error_budget, remaining_error_budget

def calculate_max_of_solution_norm(fluid_nodes, uniform_density_deviation):

    phi_max_squared =0
    for k in range(1,4):
        phi_max_squared +=fluid_nodes**k*(1 + uniform_density_deviation)**(2*k)
    return np.sqrt(phi_max_squared)



def num_grid_nodes(Reynolds, L_xyz, L_char=0.01):
    """
    Calculate the number of grid nodes for a given Reynolds number, domain dimensions, and characteristic length

    Inputs:
    Reynolds: The Reynolds number (equal to ratio of characteristic length to Kolmogorov scale). Can be a scalar or a vector.
    L_xyz: A 3-vector giving the x, y, and z lengths of the computational domain.
    Lchar: A scalar giving the characteristic length.

    Outputs:
    n_grid_nodes: An integer giving the total number of grid nodes.
    n_xyz: A 3-vector or Nx3 array of integers (where N>1 Reynolds numbers were input). Each row consists of the number of grid nodes in the x, y, and z dimensions.
    Dx: A scalar or vector (if multiple Reynolds numbers are input) giving the spacing between grid nodes (equal to the Kolmogorov scale).
    """

    # Calculate the grid node spacing (Kolmogorov scale)
    Dx = np.tile(L_char,np.size(Reynolds)) / Reynolds

    # Calculate the number of grid nodes in each dimension
    n_xyz = np.ceil(np.outer((1/Dx),L_xyz))
    
    # Multiply nx*ny*nz to get total number of grid nodes
    n_grid_nodes = np.prod(n_xyz, axis=1)

    return n_grid_nodes, n_xyz, Dx

# Example usage:
Lxyz = [0.10, 0.08, 0.08]
Lchar = 0.01

# with a single Reynolds number:
Re = 20
try:
    result = num_grid_nodes(Re, Lxyz, Lchar)
    print(f"The ratio is: {result}")
except ValueError as e:
    print(e)
    
# with multiple Reynolds numbers:
Re = [1,20,500]
try:
    result = num_grid_nodes(Re, Lxyz, Lchar)
    print(f"The ratio is: {result}")
except ValueError as e:
    print(e)



def log_function_inputs(func):
    """Decorator to log function calls and their arguments."""
    def wrapper(*args, **kwargs):
        # Get the function name
        func_name = func.__name__
        
        # Log the function's arguments
        arg_names = inspect.getfullargspec(func).args
        inputs = {name: arg for name, arg in zip(arg_names, args)}
        inputs.update(kwargs)
        
        print(f"Function '{func_name}' called with inputs: {inputs}")
        
        return func(*args, **kwargs)  # Call the original function
    return wrapper
