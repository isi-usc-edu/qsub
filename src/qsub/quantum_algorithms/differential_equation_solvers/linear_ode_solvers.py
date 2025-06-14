from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget

from typing import Optional
import math
import warnings


class TaylorQuantumODESolver(SubroutineModel):
    """
    Subroutine for preparing the amplified final time state of the ODE solver according to https://arxiv.org/abs/2309.07881.

    Notes:

    - This subroutine sets requirements of and routes specific subroutines (e.g. ODEFinalTimePrep(), ODEHistoryBlockEncoding(),
    ODEHistoryBVector()) to service the tasks of generic subroutines (e.g. amplitude_amplification and qlsa).

    - The desired subroutines for solve_linear_system, ode_matrix_block_encoding, prepare_inhomogeneous_term_vector, and
    prepare_initial_vector are specified via the set_requirements() method. They are then

    Attributes:
        amplify_amplitude (SubroutineModel): Subroutine for the amplification of the final time state.
    """

    def __init__(
        self,
        task_name="solve_quantum_ode",
        requirements=None,
        amplify_amplitude: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if amplify_amplitude is not None:
            self.amplify_amplitude = amplify_amplitude
        else:
            self.amplify_amplitude = SubroutineModel("amplify_amplitude")

        # Initialize the sub-subtask requirements as generic subroutines with task names
        self.requirements["solve_linear_system"] = SubroutineModel(
            "solve_linear_system"
        )
        self.requirements["ode_matrix_block_encoding"] = GenericBlockEncoding(
            "ode_matrix_block_encoding"
        )
        self.requirements["prepare_inhomogeneous_term_vector"] = SubroutineModel(
            "prepare_inhomogeneous_term_vector"
        )
        self.requirements["prepare_initial_vector"] = SubroutineModel(
            "prepare_initial_vector"
        )

    def set_requirements(
        self,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        failure_tolerance: float = None,
        norm_inhomogeneous_term_vector: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        solve_linear_system: SubroutineModel = None,
        ode_matrix_block_encoding: SubroutineModel = None,
        prepare_inhomogeneous_term_vector: SubroutineModel = None,
        prepare_initial_vector: SubroutineModel = None,
        matrix_norm_upperbound: float=None
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot time discretization budget
        (
            time_discretization_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
        epsilon_td = time_discretization_failure_tolerance / 4

        # Compute state preparation probability
        state_preparation_probability = (
            get_state_preparation_overlap_of_ode_final_state(
                self.requirements["evolution_time"],
                epsilon_td,
                self.requirements["norm_inhomogeneous_term_vector"],
                self.requirements["norm_x_t"],
                self.requirements["A_stable"],
                self.requirements["matrix_norm_upperbound"]
            )
        )

        # Set number of calls to the amplify amplitude task to one
        # (Amara.K: Should be 1/state_preparation_probability 5/5/2025)
        self.amplify_amplitude.number_of_times_called = 1/state_preparation_probability
    

        # Set amp amp requirements
        self.amplify_amplitude.set_requirements(
            failure_tolerance=remaining_failure_tolerance,
            input_state_squared_overlap=state_preparation_probability,
        )

        # Set amp_amp st prep subroutine as final_state_prep
        self.amplify_amplitude.state_preparation_oracle = ODEFinalTimePrep()

        # Set final_state_prep probability requirement
        self.amplify_amplitude.state_preparation_oracle.set_requirements(
            state_preparation_probability=state_preparation_probability
        )
        # Set final_state_prep subroutine as qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state = (
            self.requirements["solve_linear_system"]
        )

        # Set qlsa b_vector_prep as ODEHistoryBVector() with prepare_inhomogeneous_term_vector
        # and prepare_initial_vector set as subroutines
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.prepare_b_vector = ODEHistoryBVector(
            prepare_inhomogeneous_term_vector=self.requirements[
                "prepare_inhomogeneous_term_vector"
            ],
            prepare_initial_vector=self.requirements["prepare_initial_vector"],
        )
        # Set qlsa block encoding subroutine as ODEHistoryBlockEncoding
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.linear_system_block_encoding = ODEHistoryBlockEncoding(
            block_encode_ode_matrix=self.requirements["ode_matrix_block_encoding"]
        )

        # Pass problem instance requirements to ODEHistoryBlockEncoding
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.linear_system_block_encoding.set_requirements(
            evolution_time=self.requirements["evolution_time"],
            epsilon_td=epsilon_td,
            norm_inhomogeneous_term_vector=self.requirements[
                "norm_inhomogeneous_term_vector"
            ],
            norm_x_t=self.requirements["norm_x_t"],
        )

    def count_qubits(self):
        return self.amplify_amplitude.count_qubits()

    def get_subnormalization(self):
        # Returns the normalization factor for the vector encoding the solution state
        NotImplementedError("THIS QUANTITY NOT NEEDED FOR lbm_drag_estimation.py !!")


def get_state_preparation_overlap_of_ode_history_state(
    norm_inhomogeneous_term_vector,
):
    """
    Compute the state preparation probability for the history state of the ODE solver according to Eq. 17 of https://arxiv.org/abs/2309.07881.

    Arguments:
        norm_inhomogeneous_term_vector (float): Norm of vector b.

    Returns:
        state_preparation_probability: (float) overlap squared of ODE history state before amplitude amplification.
    """
    # Constant
    I_0_2 = 2.2796  # Approximation of I_0(2)

    # Compute success probabilities Pr_H from Eq. 17 of https://arxiv.org/abs/2309.07881
    K = (3 - math.exp(1)) ** 2 if norm_inhomogeneous_term_vector != 0 else 1
    state_preparation_probability = K / (K - 1 + I_0_2)

    return state_preparation_probability


def get_state_preparation_overlap_of_ode_final_state(
    evolution_time,
    epsilon_td,
    norm_inhomogeneous_term_vector,
    norm_x_t,
    A_stable,
    matrix_norm_upperbound
):
    """
    Compute the state preparation probability for the final time state of the ODE solver according to Eq. 18 of https://arxiv.org/abs/2309.07881.

    Arguments:
        evolution_time (float): Total time for the ODE solver.
        epsilon_td (float): Time discretization error.
        mu_P_A (float): A parameter related to matrix A.
        norm_inhomogeneous_term_vector (float): Norm of vector b.
        kappa_P (float): Condition number of the preconditioner used in the QLSA.
        norm_x_t (float): Maximum norm of the vector x(t) over the interval [0, T].
        A_stable (bool): Indicates if matrix A is stable.

    Returns:
        state_preparation_probability: (float) overlap squared of ODE solution state before amplitude amplification.
    """
    # Constant
    I_0_2 = 2.2796  # Approximation of I_0(2)

    # Compute the Taylor truncation
    taylor_truncation = compute_ode_taylor_truncation(
        evolution_time, epsilon_td, 
        norm_inhomogeneous_term_vector, 
        norm_x_t,
        matrix_norm_upperbound
    )

    # Set the idling parameter p
    idling_parameter = set_ode_idling_parameter(
        evolution_time, 
        taylor_truncation, 
        A_stable,
        matrix_norm_upperbound
        
    )

    # Compute success probability Pr_F from Eq. 18 of https://arxiv.org/abs/2309.07881
    K = (3 - math.exp(1)) ** 2 if norm_inhomogeneous_term_vector != 0 else 1
    n_time_steps = evolution_time/matrix_norm_upperbound
    state_preparation_probability = 1 / (
        (1 - (I_0_2 - 1) / ((idling_parameter + 1) * K))
        + (n_time_steps+ 1)
        * (I_0_2 - 1)
        / ((idling_parameter + 1) * K)
        * ((1 + epsilon_td) / (1 - epsilon_td)) ** 2
        * math.exp(1) ** 2
    )

    return state_preparation_probability


def compute_ode_taylor_truncation(
    evolution_time, epsilon_td, norm_inhomogeneous_term_vector, norm_x_t,
    matrix_norm_upperbound
):
    n_time_steps= evolution_time/matrix_norm_upperbound
    x_star =  n_time_steps* math.exp(3)/epsilon_td*( 1 + evolution_time * math.exp(2) * 
                            norm_inhomogeneous_term_vector / norm_x_t
                            )
    return math.ceil(
        (3 * math.log(x_star) / 2 + 1) / math.log(1 + math.log(x_star) / 2) - 1
    )


def set_ode_idling_parameter(evolution_time, taylor_truncation, A_stable, matrix_norm_upperbound):
    n_time_steps= evolution_time/matrix_norm_upperbound
    if A_stable:
        return math.ceil(math.sqrt(n_time_steps) / (taylor_truncation + 1)) * (
            taylor_truncation + 1
        )
    else:
        return math.ceil(n_time_steps/ (taylor_truncation + 1)) * (
            taylor_truncation + 1
        )


class ODEFinalTimePrep(SubroutineModel):
    """
    Subroutine for the preparation of the final time state of the ODE solver according to https://arxiv.org/abs/2309.07881.

    Attributes:
        prepare_ode_history_state (SubroutineModel): Subroutine for the preparation of the ODE history state.
    """

    def __init__(
        self,
        task_name="prepare_ode_final_time_state",
        requirements=None,
        prepare_ode_history_state: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if prepare_ode_history_state is not None:
            self.prepare_ode_history_state = prepare_ode_history_state
        else:
            self.prepare_ode_history_state = SubroutineModel(
                "prepare_ode_history_state"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        state_preparation_probability: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Note: This subroutine consumes no failure probability.
        # Rather, it properly costs the error budget consumption of
        # the qlsa subroutine that prepares the history state on account
        # of the need to post-select on the final state

        state_preparation_probability = self.requirements[
            "state_preparation_probability"
        ]
        failure_tolerance = self.requirements["failure_tolerance"]
        # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
        epsilon_ls = (
            failure_tolerance * state_preparation_probability / (2 + failure_tolerance)
        )

        # The subroutine is only called once
        self.prepare_ode_history_state.number_of_times_called = 1

        # Set prepare ode history state requirements
        self.prepare_ode_history_state.set_requirements(
            failure_tolerance=epsilon_ls,
        )

    def count_qubits(self):
        return self.prepare_ode_history_state.count_qubits()


class ODEHistoryBlockEncoding(SubroutineModel):
    """
    Subroutine for the block encoding of the ODE history matrix according to https://arxiv.org/abs/2309.07881.

    TODO: Add in costs of operations (e.g. Toffoli gates) used to create this block encoding from the ODE matrix.

    Attributes:
        block_encode_ode_matrix (SubroutineModel): Subroutine for the block encoding of the ODE matrix.
    """

    def __init__(
        self,
        task_name="block_encode_ode_history_system",
        requirements=None,
        block_encode_ode_matrix: Optional[GenericBlockEncoding] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_ode_matrix is not None:
            self.block_encode_ode_matrix = block_encode_ode_matrix
        else:
            self.block_encode_ode_matrix = GenericBlockEncoding(
                "block_encode_ode_matrix"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        evolution_time: float = None,
        epsilon_td: float = None,
        norm_inhomogeneous_term_vector: float = None,
        norm_x_t: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Note: This subroutine consumes no failure probability.

        # Set number of calls to the linear term block encoding
        self.block_encode_ode_matrix.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.block_encode_ode_matrix.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        subnormalization_of_A = self.block_encode_ode_matrix.get_subnormalization()
        matrix_norm_upperbound = self.block_encode_ode_matrix.requirements["matrix_norm_upperbound"]
        taylor_truncation = compute_ode_taylor_truncation(
            self.requirements["evolution_time"],
            self.requirements["epsilon_td"],
            self.requirements["norm_inhomogeneous_term_vector"],
            self.requirements["norm_x_t"],
            matrix_norm_upperbound
        )
        # Calculation of history block encoding subnormalization
        # from Step 4 of Theorem 2 in https://arxiv.org/abs/2309.07881
        time_step = 1/matrix_norm_upperbound
        omega_L = (1 + math.sqrt(taylor_truncation + 1) + subnormalization_of_A*time_step) / (
            math.sqrt(taylor_truncation + 1) + 2
        )
        return omega_L

    def get_condition_number(self):
        evolution_time = self.requirements["evolution_time"]
        epsilon_td = self.requirements["epsilon_td"]
        norm_inhomogeneous_term_vector = self.requirements[
            "norm_inhomogeneous_term_vector"
        ]
        norm_x_t = self.requirements["norm_x_t"]

        kappa_P = self.block_encode_ode_matrix.requirements["kappa_P"]
        mu_P_A = self.block_encode_ode_matrix.requirements["mu_P_A"]
        A_stable = self.block_encode_ode_matrix.requirements["A_stable"]
        matrix_norm_upperbound = self.block_encode_ode_matrix.requirements["matrix_norm_upperbound"]

        taylor_truncation = compute_ode_taylor_truncation(
            evolution_time, 
            epsilon_td, 
            norm_inhomogeneous_term_vector, 
            norm_x_t,
            matrix_norm_upperbound
        )
        idling_parameter = set_ode_idling_parameter(
            evolution_time, 
            taylor_truncation, 
            A_stable,
            matrix_norm_upperbound
        )

        kappa_L = compute_history_matrix_condition_number(
            evolution_time,
            epsilon_td,
            taylor_truncation,
            idling_parameter,
            kappa_P,
            mu_P_A,
            matrix_norm_upperbound,
            A_stable
        )

        return kappa_L

    def count_qubits(self):
        evolution_time = self.requirements["evolution_time"]
        epsilon_td = self.requirements["epsilon_td"]
        norm_inhomogeneous_term_vector = self.requirements[
            "norm_inhomogeneous_term_vector"
        ]
        norm_x_t = self.requirements["norm_x_t"]
        matrix_norm_upperbound = self.block_encode_ode_matrix.requirements["matrix_norm_upperbound"]
        taylor_truncation = compute_ode_taylor_truncation(
            evolution_time, 
            epsilon_td, 
            norm_inhomogeneous_term_vector, 
            norm_x_t,
            matrix_norm_upperbound

        )
        A_stable = self.block_encode_ode_matrix.requirements["A_stable"]
        idling_parameter = set_ode_idling_parameter(
            evolution_time, 
            taylor_truncation, 
            A_stable,
            matrix_norm_upperbound
        )

        # From Step 10 in Theorem 2 of https://arxiv.org/abs/2309.07881
        number_of_history_and_truncation_ancilla_qubits = (evolution_time + 1) * (
            taylor_truncation + 1
        ) + idling_parameter

        # From Theorem 1 in Appendix of https://arxiv.org/abs/2309.07881
        number_of_qubits = (
            self.block_encode_ode_matrix.count_qubits()
            + number_of_history_and_truncation_ancilla_qubits
            + 6
        )

        return number_of_qubits


def compute_history_matrix_condition_number(
    evolution_time,
    epsilon_td,
    taylor_truncation,
    idling_parameter,
    kappa_P,
    mu_P_A,
    matrix_norm_upperbound,
    A_stable
):
    """
    Compute the condition number of the history matrix according
    to Step 5 of Theorem 2 in https://arxiv.org/abs/2309.07881.

    Arguments:
        evolution_time (float): Total time for the ODE solver.
        epsilon_td (float): Time discretization error.
        taylor_truncation (int): Taylor truncation.
        idling_parameter (int): Idling parameter.
        kappa_P (float): Condition number of the preconditioner used in the QLSA.
        mu_P_A (float): A parameter related to matrix A.

    Returns:
        kappa_L (float): The condition number bound.
    """

    I_0_2 = 2.2796  # Approximation of I_0(2)
    n_time_steps = evolution_time/matrix_norm_upperbound
    time_step = 1/matrix_norm_upperbound
    C_max = 1  # Example value for C_max
    n_time_steps = evolution_time/matrix_norm_upperbound
    p = idling_parameter  # Example value for p
    I_0_2 = 2.2796  # Approximation of I_0(2)
    c_plus =1
    kappa_L = 0

    g_k = sum(
        [
            (
                math.factorial(s)
                * sum([1 / math.factorial(j) for j in range(s, taylor_truncation + 1)])
                ** 2
            )
            for s in range(1, taylor_truncation + 1)
        ]
    )

    if A_stable:

        kappa_L = math.sqrt(
            (
                (1 + epsilon_td) ** 2
                * (1 + g_k)
                * kappa_P
                * (
                    idling_parameter
                    * (1 - math.exp(2 * evolution_time * mu_P_A + 2 * mu_P_A*time_step))
                    / (1 - math.exp(2 * mu_P_A*time_step))
                    + I_0_2
                    * (
                        math.exp(2 * mu_P_A * (evolution_time + 2*time_step))
                        + n_time_steps
                        + 1
                        - math.exp(2 * mu_P_A*time_step) * (2 + n_time_steps)
                    )
                    / ((1 - math.exp(2 * time_step*mu_P_A)) ** 2)
                    + idling_parameter * (idling_parameter + 1) / 2
                    + (idling_parameter + n_time_steps * taylor_truncation) * (I_0_2 - 1)
                )
            )
            * (math.sqrt(taylor_truncation + 1) + 2)
        )
    else:
        term1 = math.sqrt(taylor_truncation + 1) + 2
        term2 = (epsilon_td / c_plus)
        term3 = 1 + term2
        term4 = 1 + g_k * (n_time_steps + 1) * (p + I_0_2 * (n_time_steps / 2 + 1))
        term5 = (p * (p + 1)) / 2
        term6 = (p + n_time_steps * taylor_truncation) * (I_0_2 - 1)
        inner_expression = C_max**2*term3**2 * (term4 + term5 + term6)
        kappa_L = term1 * math.sqrt(inner_expression)

    return kappa_L


class ODEHistoryBVector(SubroutineModel):
    """
    Costing of the ODE History b vector according to https://arxiv.org/abs/2309.07881.
    """

    def __init__(
        self,
        task_name="prepare_ode_history_b_vector",
        requirements=None,
        prepare_inhomogeneous_term_vector: Optional[SubroutineModel] = None,
        prepare_initial_vector: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if prepare_inhomogeneous_term_vector is not None:
            self.prepare_inhomogeneous_term_vector = prepare_inhomogeneous_term_vector
        else:
            self.prepare_inhomogeneous_term_vector = SubroutineModel(
                "prepare_inhomogeneous_term_vector"
            )

        if prepare_initial_vector is not None:
            self.prepare_initial_vector = prepare_initial_vector
        else:
            self.prepare_initial_vector = SubroutineModel("prepare_initial_vector")

    def set_requirements(
        self,
        failure_tolerance: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot time discretization budget
        (
            prepare_inhomogeneous_term_vector_failure_tolerance,
            prepare_initial_vector_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set number of calls to the linear term block encoding
        self.prepare_inhomogeneous_term_vector.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.prepare_inhomogeneous_term_vector.set_requirements(
            failure_tolerance=prepare_inhomogeneous_term_vector_failure_tolerance,
        )

        # Set number of calls to the quadratic term block encoding
        self.prepare_initial_vector.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.prepare_initial_vector.set_requirements(
            failure_tolerance=prepare_initial_vector_failure_tolerance,
        )
