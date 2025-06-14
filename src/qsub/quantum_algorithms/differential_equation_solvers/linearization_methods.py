from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget

from typing import Optional
import warnings
from sympy import symbols, Max, ceiling, log, Basic
import numpy as np


class CarlemanBlockEncoding(GenericBlockEncoding):
    def __init__(
        self,
        task_name="block_encode_carleman_linearization",
        requirements=None,
        block_encode_linear_term: Optional[GenericBlockEncoding] = None,
        block_encode_quadratic_term: Optional[GenericBlockEncoding] = None,
        block_encode_cubic_term: Optional[GenericBlockEncoding] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_linear_term is not None:
            self.block_encode_linear_term = block_encode_linear_term
        else:
            self.block_encode_linear_term = GenericBlockEncoding(
                "block_encode_linear_term"
            )

        if block_encode_quadratic_term is not None:
            self.block_encode_quadratic_term = block_encode_quadratic_term
        else:
            self.block_encode_quadratic_term = GenericBlockEncoding(
                "block_encode_quadratic_term"
            )

        if block_encode_cubic_term is not None:
            self.block_encode_cubic_term = block_encode_cubic_term
        else:
            self.block_encode_cubic_term = GenericBlockEncoding(
                "block_encode_cubic_term"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        kappa_P: float = None,
        mu_P_A: float = None,
        A_stable: bool = None,
        matrix_norm_upperbound:float=None
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

        # Allot truncation level failure tolerance budget
        (
            truncation_error,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # TODO: may eventually want the Carleman block encoding to consume some
        # failure tolerance based on the degree that it uses (or at some point we may want
        # to allow this degree to be set by this failure rate consumption)

        (
            linear_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(1/3, remaining_failure_tolerance)

        (
            quadratic_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(1/3, remaining_failure_tolerance)

        (
            cubic_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(1/3, remaining_failure_tolerance)

        # Set number of calls to the linear term block encoding
        self.block_encode_linear_term.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.block_encode_linear_term.set_requirements(
            failure_tolerance=linear_block_encoding_failure_tolerance,
        )

        # Set number of calls to the quadratic term block encoding
        self.block_encode_quadratic_term.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.block_encode_quadratic_term.set_requirements(
            failure_tolerance=quadratic_block_encoding_failure_tolerance,
        )

        # Set number of calls to the cubic term block encoding
        self.block_encode_cubic_term.number_of_times_called = 1

        # Set cubic term block encoding requirements
        self.block_encode_cubic_term.set_requirements(
            failure_tolerance=cubic_block_encoding_failure_tolerance,
        )

    def get_subnormalization(self):
        carleman_truncation_level = 3 # Chosen for this paper https://arxiv.org/abs/2406.06323

        ode_degree = 3
        if (
            isinstance(self.block_encode_linear_term.get_subnormalization(), Basic)
            or isinstance(
                self.block_encode_quadratic_term.get_subnormalization(), Basic
            )
            or isinstance(self.block_encode_cubic_term.get_subnormalization(), Basic)
        ):
            max_block_encoding_subnormalization = Max(
                self.block_encode_linear_term.get_subnormalization(),
                self.block_encode_quadratic_term.get_subnormalization(),
                self.block_encode_cubic_term.get_subnormalization(),
            )
        else:
            max_block_encoding_subnormalization = max(
                self.block_encode_linear_term.get_subnormalization(),
                self.block_encode_quadratic_term.get_subnormalization(),
                self.block_encode_cubic_term.get_subnormalization(),
            )

        # Upper bound on subnormalization from paper (TODO: add reference)
        subnormalization = (
            ode_degree
            * carleman_truncation_level
            * (carleman_truncation_level + 1)
            * max_block_encoding_subnormalization
            / 2
        )

        return subnormalization

    def count_qubits(self):
        carleman_truncation_level = 3 # chosen for this paper https://arxiv.org/abs/2406.06323

        max_number_of_ancillas_used_to_block_encode_terms = Max(
            self.block_encode_linear_term.count_block_encoding_ancilla_qubits(), 
            self.block_encode_cubic_term.count_encoding_qubits(), 
            self.block_encode_quadratic_term.count_encoding_qubits()

        )
        number_of_qubits = max_number_of_ancillas_used_to_block_encode_terms + 2*np.log2(carleman_truncation_level)
        return number_of_qubits

    def count_encoding_qubits(self):
        carleman_truncation_level = 3 # chosen for this paper https://arxiv.org/abs/2406.06323
        ode_degree = 3
        max_number_of_ancillas_used_to_block_encode_terms = Max(
        self.block_encode_linear_term.count_block_encoding_ancilla_qubits(), 
        self.block_encode_cubic_term.count_encoding_qubits(), 
        self.block_encode_quadratic_term.count_encoding_qubits()
        )
        number_of_qubits_encoding_system = (
            self.block_encode_linear_term.count_encoding_qubits() +
            self.block_encode_quadratic_term.count_encoding_qubits()+
            self.block_encode_cubic_term.count_encoding_qubits()
        )

        number_of_qubits = (
            number_of_qubits_encoding_system * (carleman_truncation_level)
            + max_number_of_ancillas_used_to_block_encode_terms
            + 3 * ceiling(np.log2(carleman_truncation_level))
            + ceiling(np.log2(ode_degree))
        )
        return number_of_qubits
            

    
    def count_block_encoding_ancilla_qubits(self):
        carleman_truncation_level = 3 # chosen for this paper https://arxiv.org/abs/2406.06323
        max_number_of_ancillas_used_to_block_encode_terms = Max(
        self.block_encode_linear_term.count_block_encoding_ancilla_qubits(), 
        self.block_encode_cubic_term.count_encoding_qubits(), 
        self.block_encode_quadratic_term.count_encoding_qubits()
        )
        number_of_qubits = max_number_of_ancillas_used_to_block_encode_terms + 2*np.log2(carleman_truncation_level)
        return number_of_qubits

    def count_qubits(self):
        return self.count_encoding_qubits() + self.count_block_encoding_ancilla_qubits()
