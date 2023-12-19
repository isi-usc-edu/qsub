from qsub.subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget

from typing import Optional
import warnings


class CarlemanBlockEncoding(SubroutineModel):
    def __init__(
        self,
        task_name="block_encode_carleman_linearization",
        requirements=None,
        block_encode_linear_term: Optional[SubroutineModel] = None,
        block_encode_quadratic_term: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_linear_term is not None:
            self.block_encode_linear_term = block_encode_linear_term
        else:
            self.block_encode_linear_term = SubroutineModel("block_encode_linear_term")

        if block_encode_quadratic_term is not None:
            self.block_encode_quadratic_term = block_encode_quadratic_term
        else:
            self.block_encode_quadratic_term = SubroutineModel(
                "block_encode_quadratic_term"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        kappa_P: float = None,
        mu_P_A: float = None,
        A_stable: bool = None,
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
            truncation_error,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        be_costs = get_block_encoding_costs_from_carleman_requirements(truncation_error)

        # Set number of calls to the linear term block encoding
        self.block_encode_linear_term.number_of_times_called = be_costs

        # Set linear term block encoding requirements
        self.block_encode_linear_term.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

        # Set number of calls to the quadratic term block encoding
        self.block_encode_quadratic_term.number_of_times_called = be_costs

        # Set quadratic term block encoding requirements
        self.block_encode_quadratic_term.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        warnings.warn("This function is not fully implemented.", UserWarning)
        return 42


def get_block_encoding_costs_from_carleman_requirements(truncation_error):
    warnings.warn("This function is not fully implemented.", UserWarning)
    return 1