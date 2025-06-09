import numpy as np
# from qsub.subroutine_model import SubroutineModel
from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    IterativeQuantumAmplitudeEstimationCircuit,
    IterativeQuantumAmplitudeEstimationAlgorithm,
)
from qsub.quantum_algorithms.differential_equation_solvers.linear_ode_solvers import (
    TaylorQuantumODESolver,
)

from qsub.quantum_algorithms.fluid_dynamics.lattice_boltzmann import (
    LBMDragEstimation,
    LBMDragCoefficientsReflection,
    LBMLinearTermBlockEncoding,
    LBMQuadraticTermBlockEncoding,
    LBMCubicTermBlockEncoding,
)

from qsub.quantum_algorithms.differential_equation_solvers.linearization_methods import (
    CarlemanBlockEncoding,
)

import matplotlib.pyplot as plt
from qsub.utils import calculate_max_of_solution_norm, num_grid_nodes
import pandas as pd
import json
from pprint import pprint


CONFIG = {
    "x_length_in_meters": 0.1,
    "y_length_in_meters": 0.08,
    "z_length_in_meters": 0.08,
    "sphere_radius_in_meters": 0.005,
    "number_of_velocity_grid_points": 27,
    "mu_P_A" : -0.00001,
    "kappa_P": 1,
    "A_stable": False
}

UTILITY_SCALE_CONFIG = {
    "x_length_in_meters": 0.1*100,
    "y_length_in_meters": 0.08*100,
    "z_length_in_meters": 0.08*100,
    "sphere_radius_in_meters": 0.005*100,
    "number_of_velocity_grid_points": 27,
    "mu_P_A" : -0.00001,
    "kappa_P": 1,
    "A_stable": False
}



new_instances = True
structured_block_encodings = True
df = pd.read_csv("problem_instance_parameters_and_results_20241203_newsphere.xlsx - one-sheet.csv")
df.set_index("Parameter", inplace=True)
delta_t = df.loc["delta_t"]
nf = df.loc["n_f"]
num_points = df.loc["num_points"]
evolution_times = df.loc["T lattice_evolution_time_Scale"]
matrix_norm_upperbound = float(df.loc["||A||_2", "Sphere Re=10^3"])


if new_instances:
    utility_time_discretizations = list(map(float, delta_t.loc["Sphere Re=10^1":].values.tolist()))
    utility_scale_evolution_times = list(map(float, evolution_times.loc["Sphere Re=10^1":].values.tolist()))
    reynolds_numbers = [10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    failure_tolerance_values = [0.05, 0.1, 0.15]
    n_fluid_nodes = list(map(float, nf.loc["Sphere Re=10^1":].values.tolist()))
    size = list(map(float, num_points.loc["Sphere Re=10^1":].values.tolist()))
    grid_points, _, _ = num_grid_nodes(reynolds_numbers,[UTILITY_SCALE_CONFIG['x_length_in_meters'], UTILITY_SCALE_CONFIG['y_length_in_meters'], UTILITY_SCALE_CONFIG['z_length_in_meters']])
else:
    time_discretizations = list(map(float, delta_t.loc[["Sphere Re=1", "Sphere Re=20", "Sphere Re=500"]].values.tolist()))
    evolution_times = list(map(float, evolution_times.loc[["Sphere Re=1", "Sphere Re=20", "Sphere Re=500"]].values.tolist()))
    reynolds_numbers = [1, 20, 500]
    failure_tolerance_values = [0.05]
    n_fluid_nodes = list(map(float, nf.loc[["Sphere Re=1", "Sphere Re=20", "Sphere Re=500"]].values.tolist()))
    size = list(map(float, num_points.loc[["Sphere Re=1", "Sphere Re=20", "Sphere Re=500"]].values.tolist()))
    grid_points, _, _ = num_grid_nodes(reynolds_numbers,[CONFIG['x_length_in_meters'], CONFIG['y_length_in_meters'], CONFIG['z_length_in_meters']])



def calculate_cell_volume(number_of_spatial_grid_points, config):
    """Calculate the volume of a cell."""
    return (
        config["x_length_in_meters"]
        * config["y_length_in_meters"]
        * config["z_length_in_meters"]
        / number_of_spatial_grid_points
    )

def calculate_drag_force(cell_volume, sphere_radius, time_discretization):
    """Estimate the drag force."""
    cell_face_area = cell_volume ** (2 / 3)
    number_of_cells_incident_on_face = (
        2 * np.pi * sphere_radius**2 / cell_face_area
    )
    return (cell_volume / time_discretization) * number_of_cells_incident_on_face

def setup_quantum_solver(number_of_spatial_grid_points, config):
    """Set up quantum encodings and solvers."""
    linear_encoding = LBMLinearTermBlockEncoding(structured=structured_block_encodings)
    linear_encoding.set_requirements(
        number_of_spatial_grid_points=number_of_spatial_grid_points, 
        number_of_velocity_grid_points= config["number_of_velocity_grid_points"])

    quadratic_encoding = LBMQuadraticTermBlockEncoding(structured=structured_block_encodings)
    quadratic_encoding.set_requirements(
        number_of_spatial_grid_points=number_of_spatial_grid_points, 
        number_of_velocity_grid_points=config["number_of_velocity_grid_points"])

    cubic_encoding = LBMCubicTermBlockEncoding(structured=structured_block_encodings)
    cubic_encoding.set_requirements(
            number_of_spatial_grid_points=number_of_spatial_grid_points, 
            number_of_velocity_grid_points=config["number_of_velocity_grid_points"])

    carleman_encoding = CarlemanBlockEncoding(
        block_encode_linear_term=linear_encoding,
        block_encode_quadratic_term=quadratic_encoding,
        block_encode_cubic_term=cubic_encoding,
    )
    carleman_encoding.set_requirements(
        kappa_P=config['kappa_P'], 
        mu_P_A=config['mu_P_A'], 
        A_stable=config['A_stable'],
        matrix_norm_upperbound=matrix_norm_upperbound

    )

    taylor_ode = TaylorQuantumODESolver(
                amplify_amplitude=ObliviousAmplitudeAmplification()
        )
    taylor_ode.set_requirements(
        solve_linear_system=TaylorQLSA(), 
        ode_matrix_block_encoding=carleman_encoding,
        matrix_norm_upperbound=matrix_norm_upperbound
    )
    return taylor_ode

def estimate_quantum_resources(evolution_time, 
                    failure_tolerance, 
                    relative_estimation_error, 
                    number_of_spatial_grid_points, 
                    uniform_density_deviation, 
                    fluid_nodes, 
                    config ,
                    time_discretization_in_seconds=None
                    ):
    """Generate graphs for drag estimation."""

    norm_x_t = calculate_max_of_solution_norm(fluid_nodes, uniform_density_deviation)

    cell_volume = calculate_cell_volume(number_of_spatial_grid_points, config)
    rough_drag_force = 0
    if "time_discretization_in_seconds" in config.keys():
        rough_drag_force = calculate_drag_force(
            cell_volume, config["sphere_radius_in_meters"], config["time_discretization_in_seconds"]
        )
    elif time_discretization_in_seconds is not None:
        rough_drag_force = calculate_drag_force(
            cell_volume, config["sphere_radius_in_meters"], time_discretization_in_seconds
        )
    else:
        print("Check time discretization parameter. It might not have been provided")
    

    # Set up the quantum solver
    taylor_ode_solver = setup_quantum_solver(
        number_of_spatial_grid_points, 
        config, 
    )
    # Set up amplitude estimation algorithm
    amplitude_estimation = IterativeQuantumAmplitudeEstimationAlgorithm(
            run_iterative_qae_circuit=IterativeQuantumAmplitudeEstimationCircuit()
        )
    # Set up drag estimation
    drag_estimation = LBMDragEstimation(
        estimate_amplitude= amplitude_estimation
    )
    drag_estimation.set_requirements(
        evolution_time=evolution_time,
        relative_estimation_error=relative_estimation_error,
        estimated_drag_force=rough_drag_force,
        failure_tolerance=failure_tolerance,
        norm_inhomogeneous_term_vector=0.0,  # homogeneous ODE
        norm_x_t=norm_x_t,
        solve_quantum_ode=taylor_ode_solver,
        number_of_spatial_grid_points=number_of_spatial_grid_points,
        **config,
        matrix_norm_upperbound=matrix_norm_upperbound,
        time_discretization_in_seconds= time_discretization_in_seconds,
        mark_drag_vector=LBMDragCoefficientsReflection(),
    )

    # Run the solver and collect results
    drag_estimation.run_profile(verbose=False)
    drag_estimation.print_profile()
    # drag_estimation.plot_graph()

    counts = drag_estimation.count_subroutines()
    n_qubits = drag_estimation.count_qubits()
    return counts["t_gate"], n_qubits


resources = {}
evol_times =None
t_discretizations = None
for error in failure_tolerance_values:
    resources[error] = []
    if new_instances:
        evol_times = utility_scale_evolution_times
        t_discretizations = utility_time_discretizations
        config_used = UTILITY_SCALE_CONFIG
    else:
        evol_times = evolution_times
        t_discretizations = time_discretizations
        config_used = CONFIG
    for grid, evol_time, fluid_node, dt in zip(grid_points, evol_times, n_fluid_nodes, t_discretizations):
        t_counts, n_qubits = estimate_quantum_resources(
            number_of_spatial_grid_points=grid,
            evolution_time=evol_time,
            failure_tolerance=error,
            fluid_nodes=fluid_node,
            uniform_density_deviation=0.001,
            relative_estimation_error=0.1,
            config=config_used,
            time_discretization_in_seconds= dt
        )
        resources[error].append((t_counts, n_qubits))

tolerances = list(resources.keys())

# pprint(resources)

# Function to set up common plot parameters
def configure_plot_settings():
    plt.rcParams.update({'font.size': 10})  # General font size
    plt.rcParams.update({'axes.titlesize': 10})  # Title font size
    plt.rcParams.update({'axes.labelsize': 10})  # Axis labels font size
    plt.rcParams.update({'xtick.labelsize': 8})  # X-tick labels font size
    plt.rcParams.update({'ytick.labelsize': 12})  # Y-tick labels font size
    plt.rcParams.update({'legend.fontsize': 10})  # Legend font size

# Function to plot T-gate Counts or Number of Qubits vs Failure Tolerance
# def plot_tolerance_vs_resources(resources, tolerances, reynolds_numbers, ylabel, title, resource_key):
#     num_reynolds = len(reynolds_numbers)
#     num_cols = 3  # e.g., 3 plots per row
#     num_rows = (num_reynolds + num_cols - 1) // num_cols  # ceiling division
#     fig, ax = plt.subplots(figsize=(10, 6),figsize=(3, 2), squeeze=False)
#     dark_colors = ['#000000', '#8B0000', '#00008B', '#006400', '#008B8B', '#8B008B']
#     markers = ['x', '+', 'v']

#     for tol_idx, tolerance in enumerate(tolerances):
#         row = tol_idx // num_cols
#         col = tol_idx % num_cols
#         ax = ax[row][col]
#         for reynolds_idx, reynolds in enumerate(reynolds_numbers):
#             t_counts, num_qubits = resources[tolerance][reynolds_idx]
#             value = t_counts if resource_key == "t_gate" else num_qubits
#             ax.scatter(
#                 tolerance,
#                 value,
#                 s=100,
#                 color=dark_colors[reynolds_idx % len(dark_colors)],
#                 marker=markers[reynolds_idx % len(markers)],
#                 label=f'Re {reynolds}' if tol_idx == 0 else ""
#             )

#     ax.set_yscale('log')
#     ax.set_xlabel('Failure Tolerance')
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(True)
#     ax.legend(title='Reynolds Number')
#     plt.tight_layout()
#     plt.show()

def plot_tolerance_vs_resources(resources, tolerances, reynolds_numbers, ylabel, title, resource_key):
    num_reynolds = len(reynolds_numbers)
    num_cols = 3  # e.g., 3 plots per row
    num_rows = (num_reynolds + num_cols - 1) // num_cols  # ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), squeeze=False)
    dark_colors = ['#000000', '#8B0000', '#00008B', '#006400', '#008B8B', '#8B008B']
    markers = ['x', '+', 'v']

    for idx, reynolds in enumerate(reynolds_numbers):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col]

        for tol_idx, tolerance in enumerate(tolerances):
            t_counts, num_qubits = resources[tolerance][idx]
            value = t_counts if resource_key == "t_gate" else num_qubits
            ax.scatter(
                tolerance,
                value,
                s=100,
                color=dark_colors[tol_idx % len(dark_colors)],
                marker=markers[tol_idx % len(markers)],
                label=f'Tolerance={tolerance}'
            )

        ax.set_yscale('log')
        ax.set_xlabel('Failure tolerance')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Re={reynolds}')
        ax.grid(True)
        ax.legend()

    # Remove empty plots if reynolds_numbers doesn't fill all grid
    for idx in range(num_reynolds, num_rows*num_cols):
        fig.delaxes(axes[idx//num_cols][idx%num_cols])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to plot T-gate Counts vs Number of Qubits for different Reynolds numbers
def plot_t_counts_vs_qubits(resources, tolerances, reynolds_numbers, block_encoding_type: str = None):
    # Determine grid size
    num_reynolds = len(reynolds_numbers)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_reynolds + num_cols - 1) // num_cols

    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), sharey=True, squeeze=False)
    dark_colors = ['#000000', '#8B0000', '#00008B', '#006400', '#008B8B', '#8B008B']
    markers = ['x', 'o', '^', 's', 'p', '*']

    for idx, reynolds in enumerate(reynolds_numbers):
        row, col = divmod(idx, num_cols)
        ax = axes[row][col]
        for tol_idx, tolerance in enumerate(tolerances):
            t_counts, num_qubits = resources[tolerance][idx]
            ax.scatter(
                num_qubits,
                t_counts,
                s=100,
                color=dark_colors[tol_idx % len(dark_colors)],
                marker=markers[tol_idx % len(markers)],
                label=f'Tolerance={tolerance}'
            )
        ax.set_yscale('log')
        ax.set_xlabel('Number of qubits')
        ax.set_title(f'Reynolds={reynolds} ' + (block_encoding_type or ''))
        ax.grid(True)
        ax.legend(title="Failure tolerance")

    # Set shared Y-axis label on first subplot
    axes[0][0].set_ylabel('$T$-gate Counts')

    # Remove any empty subplots
    for empty_idx in range(num_reynolds, num_rows * num_cols):
        r, c = divmod(empty_idx, num_cols)
        fig.delaxes(axes[r][c])

    # Global title and layout
    fig.suptitle('T-gate counts vs. number of qubits ' + (block_encoding_type or ''), fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to plot (T-gate counts × Qubits) vs Qubits for different Reynolds numbers
def plot_t_counts_times_qubits(resources, tolerances, reynolds_numbers, block_encoding_type: str = None):
    # Determine grid size
    num_reynolds = len(reynolds_numbers)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_reynolds + num_cols - 1) // num_cols

    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), sharey=True, squeeze=False)
    dark_colors = ['#000000', '#8B0000', '#00008B', '#006400', '#008B8B', '#8B008B']
    markers = ['x', 'o', '^', 's', 'p', '*']

    for idx, reynolds in enumerate(reynolds_numbers):
        row, col = divmod(idx, num_cols)
        ax = axes[row][col]
        for tol_idx, tolerance in enumerate(tolerances):
            t_counts, num_qubits = resources[tolerance][idx]
            product_t_counts_qubits = t_counts * num_qubits
            ax.scatter(
                num_qubits,
                product_t_counts_qubits,
                s=100,
                color=dark_colors[tol_idx % len(dark_colors)],
                marker=markers[tol_idx % len(markers)],
                label=f'Tolerance={tolerance}'
            )
        ax.set_yscale('log')
        ax.set_xlabel('Number of qubits')
        ax.set_title(f'Reynolds={reynolds} ' + (block_encoding_type or ''))
        ax.grid(True)
        ax.legend(title="Failure tolerance")

    # Set shared Y-axis label on first subplot
    axes[0][0].set_ylabel('$T$ counts × number of qubits')

    # Remove any empty subplots
    for empty_idx in range(num_reynolds, num_rows * num_cols):
        r, c = divmod(empty_idx, num_cols)
        fig.delaxes(axes[r][c])

    # Global title and layout
    fig.suptitle('T-gate counts × qubits vs. number of qubits ' + (block_encoding_type or ''), fontsize=16)
    plt.tight_layout()
    plt.show()

# Main function to generate all plots
def generate_plots(resources, tolerances, reynolds_numbers):
    configure_plot_settings()
    structure=''
    if structured_block_encodings:
        structure = '(bespoke blocking encodings)'
    else:
        structure = '(unstructured block encodings)'

    # Plot T-gate Counts vs Failure Tolerance
    plot_tolerance_vs_resources(
        resources, 
        tolerances, 
        reynolds_numbers,
        ylabel='$T$-gate counts',
        title='$T$-gate counts vs failure tolerance '+ structure,
        resource_key="t_gate"
    )
    # Plot Number of Qubits vs Failure Tolerance
    plot_tolerance_vs_resources(
        resources, 
        tolerances, 
        reynolds_numbers,
        ylabel='Number of qubits',
        title='Number of qubits vs failure tolerance '+ structure,
        resource_key="qubits"
    )
    # Plot T-gate Counts vs Number of Qubits
    plot_t_counts_vs_qubits(resources, 
                            tolerances, 
                            reynolds_numbers, 
                            block_encoding_type=structure
    )
    # Plot (T-gate Counts × Qubits) vs Qubits
    plot_t_counts_times_qubits(resources, 
                               tolerances, 
                               reynolds_numbers,
                               block_encoding_type= structure
    )



def generate_qb_estimates_json_files(resources, reynold_numbers, size, block_encoding_type):

    qb_estimates = {
        "id": "Sphere-Re-20",
        "name": "Incompressible-CFD",
        "category": "scientific",
        "size": 5.12e+6,
        "task": "Estimate drag force with 0.05 failure tolerance",
        "implementation": "CL-LBM + Berry2017 + PsiQuantum_QLSA + DragMeasurable",
        "value": 0.0,
        "value_ci": [
            1.06e+9,
            7.78e+9
            ],
        "repetitions": 1,
        "logical-abstract": {
        "num_qubits":  3097,
        "t_count": 5.750650973927722e+20
        },
        "timestamp":"2024-05-14T10:44:31.390257",
        "uuid": None,
        "comments":"Verification instance: flow around a sphere at different Reynolds Numbers (Re).  Utility estimate: $0. Size is number of LBM grid points n."
  }
    for tol_dix in resources:
        for re, s, est_tuple in zip(reynold_numbers, size, resources[tol_dix]):
            qb_estimates["id"] = f"Sphere-Re-{re}"
            qb_estimates["name"] = "Incompressible-CFD"
            qb_estimates["size"] = s
            qb_estimates["task"] = "Estimate drag force with 0.05 failure tolerance"
            qb_estimates["implementation"]=  "CL-LBM + Berry2017 + PsiQuantum_QLSA + DragMeasurable"
            qb_estimates["value"]=0
            qb_estimates["repetitions"]=1
            qb_estimates["logical-abstract"]={
                "num_qubits": float(round(est_tuple[1])), 
                "t_count":  est_tuple[0]
            }
            qb_estimates["comments"] = "Verification instance: flow around a sphere at different Reynolds Numbers (Re).  Utility estimate: $0. Size is number of LBM grid points n."

            with open(f"../../QRE/CFD/2025-06/CFD_sphere_{block_encoding_type}_logRe_{re}.json", "w") as json_file:
                json.dump(qb_estimates, json_file)
    
# Call the main plotting function
generate_plots(resources, failure_tolerance_values, reynolds_numbers)

# Generating json files for DARPA reporting
# generate_qb_estimates_json_files(resources, reynolds_numbers, size, "structured")
