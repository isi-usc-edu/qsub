from typing import Optional, Union
from dataclasses import dataclass, asdict, is_dataclass
from abc import ABC, abstractmethod
import numpy as np
from graphviz import Digraph
from dataclasses import fields
import pprint
import json


@dataclass
class SubtaskRequirements(ABC):
    task_name: str = None
    failure_tolerance: float = None
    number_of_times_called: Optional[Union[float, int]] = None

    def requirements_to_string(self):
        formatted_string = json.dumps(asdict(self), indent=4)
        return formatted_string


class Subtask(ABC):
    """A Subtask is simply an object that holds SubtaskRequirements."""

    def __init__(
        self,
        requirements: SubtaskRequirements = None,
    ):
        self.requirements = requirements

    def set_subtask_requirements(self, requirements: SubtaskRequirements):
        self.requirements = requirements

    def allocate_aggregate_failure_tolerance_across_calls_to_self(
        self,
        aggregate_failure_tolerance,
    ):
        self.requirements.failure_tolerance = (
            aggregate_failure_tolerance / self.requirements.number_of_times_called
        )

    def generate_requirements_dag(self, graph=None, parent_name=None):
        # Create a new graph if one is not provided
        if graph is None:
            graph = Digraph(comment="Requirements DAG")

        # Create a node for the current subroutine
        requirements_node_name = (
            f"requirements_{type(self).__name__}_{self.requirements.task_name}"
        )
        graph.node(
            requirements_node_name,
            self.requirements.requirements_to_string(),
            shape="box",
        )

        # Create an edge from the parent node to the current node
        if parent_name:
            graph.edge(parent_name, requirements_node_name)

        return graph, requirements_node_name


class SubroutineModel(Subtask):
    """A SubroutineModel is a subtask endowed with the ability to model the servicing of
    that subtask by assigning requirements to its own (child) subtasks. These child subtasks
    are stored as attributes of the SubroutineModel. The key method of the SubroutineModel, which
    distinguishes it from a Subtask, is the assign_requirements_to_subtasks() method. This method
    uses the requirements of the SubroutineModel to assign requirements to its child subtasks.
    """

    def recursively_assign_requirements_to_all_subtasks(self):
        self.assign_requirements_to_subtasks()
        # If any of the subtasks are SubroutineModels, run their profiles as well.
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.recursively_assign_requirements_to_all_subtasks()

    def assign_requirements_to_subtasks(self):
        # Allocate failure tolerance
        failure_budget_allocation = self.allocate_failure_tolerance_budget()

        # Generate requirements for prepare_state and mark_subspace
        self.assign_requirements_from_failure_budget(failure_budget_allocation)

    def generate_requirements_dag(self, graph=None, parent_name=None):

        # Call parent class method to generate the requirements node
        graph, requirements_node_name = super().generate_requirements_dag(
            graph, parent_name
        )
        # graph = self.__super__.generate_requirements_dag(graph, parent_name)

        # Create subroutine node for the subtask
        subroutine_node_name = (
            f"subroutine_{type(self).__name__}_{self.requirements.task_name}"
        )
        graph.node(
            subroutine_node_name,
            shape="ellipse",
        )

        # Create an edge from the requirements node to the subroutine node
        graph.edge(requirements_node_name, subroutine_node_name)

        # Recursively call this method for all child subroutines
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, Subtask):
                child.generate_requirements_dag(graph, subroutine_node_name)

        return graph

    @abstractmethod
    def allocate_failure_tolerance_budget(self):
        pass

    @abstractmethod
    def assign_requirements_from_failure_budget(self, failure_budget_allocation):
        pass


@dataclass
class AmplifyAmplitudeRequirements(SubtaskRequirements):
    prepare_state_requirements: SubtaskRequirements = None
    mark_subspace_requirements: SubtaskRequirements = None
    initial_state_overlap: float = None
    task_name: str = "amplify_amplitude"


class FixedPointAmplitudeAmplification(SubroutineModel):
    def __init__(
        self,
        requirements: AmplifyAmplitudeRequirements = None,
        prepare_state: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
        mark_subspace: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__(requirements=requirements)
        self.prepare_state = prepare_state
        self.mark_subspace = mark_subspace

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.5,
            "fraction_to_prepare_state": 0.25,
            "fraction_to_mark_subspace": 0.25,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "prepare_state_aggregate_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_prepare_state"
            ]
            * self.requirements.failure_tolerance,
            "mark_subspace_aggregate_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_mark_subspace"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded
        assert (
            sum(failure_budget_allocation.values())
            <= self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def assign_requirements_from_failure_budget(self, failure_budget_allocation):

        # Generate number of calls to Grover iterate from consumed failure tolerance
        number_of_grover_iterates = (
            compute_number_of_grover_iterates_for_fixed_point_amplitude_amplification(
                failure_budget_allocation["consumed_failure_tolerance"],
                self.requirements.initial_state_overlap,
            )
        )

        # Pass input requirements to prepare_state
        self.prepare_state.requirements = self.requirements.prepare_state_requirements

        # Prepare state is called twice per Grover iterate
        self.prepare_state.requirements.number_of_times_called = (
            2 * number_of_grover_iterates
        )

        # Allocate aggregate failure tolerance across calls to prepare_state
        self.prepare_state.allocate_aggregate_failure_tolerance_across_calls_to_self(
            failure_budget_allocation["prepare_state_aggregate_failure_tolerance"]
        )

        # Pass input requirements to mark_subspace
        self.mark_subspace.requirements = self.requirements.mark_subspace_requirements

        # Mark subspace is called once per Grover iterate
        self.mark_subspace.requirements.number_of_times_called = (
            number_of_grover_iterates
        )

        # Allocate aggregate failure tolerance across calls to mark_subspace
        self.mark_subspace.allocate_aggregate_failure_tolerance_across_calls_to_self(
            failure_budget_allocation["mark_subspace_aggregate_failure_tolerance"]
        )


def compute_number_of_grover_iterates_for_fixed_point_amplitude_amplification(
    failure_tolerance, initial_state_squared_overlap
):
    # Compute number of Grover iterates needed for fixed point amplitude amplification
    # from https://arxiv.org/abs/1409.3305
    number_of_grover_iterates = np.log(2 / failure_tolerance) / np.sqrt(
        initial_state_squared_overlap
    )

    return number_of_grover_iterates


@dataclass
class BoringTaskRequirements(SubtaskRequirements):
    boring_requirement: float = None
    task_name: str = "boring_task"


class BoringSubroutineModelA(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.6,
            "fraction_to_more_boring_task": 0.4,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "more_boring_task_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_more_boring_task"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded
        assert (
            sum(failure_budget_allocation.values())
            <= self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def assign_requirements_from_failure_budget(self, failure_budget_allocation):
        # Generate requirements for more_boring_task
        more_boring_task_requirements = MoreBoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "more_boring_task_failure_tolerance"
            ]
        )
        self.more_boring_task.set_subtask_requirements(more_boring_task_requirements)


class BoringSubroutineModelB(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.8,
            "fraction_to_more_boring_task": 0.2,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "more_boring_task_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_more_boring_task"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded by more than a millionth of a percent
        assert (
            sum(failure_budget_allocation.values())
            <= 1.000001 * self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def assign_requirements_from_failure_budget(self, failure_budget_allocation):
        # Generate requirements for more_boring_task
        more_boring_task_requirements = MoreBoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "more_boring_task_failure_tolerance"
            ]
        )
        self.more_boring_task.set_subtask_requirements(more_boring_task_requirements)


@dataclass
class MoreBoringTaskRequirements(SubtaskRequirements):
    most_boring_requirement: float = None
    task_name: str = "more_boring_task"


fun_requirements = AmplifyAmplitudeRequirements(
    failure_tolerance=0.01,
    initial_state_overlap=0.5,
    prepare_state_requirements=SubtaskRequirements(),
    mark_subspace_requirements=SubtaskRequirements(),
)

amp_amp = FixedPointAmplitudeAmplification(requirements=fun_requirements)
amp_amp.prepare_state = BoringSubroutineModelA()
amp_amp.mark_subspace = BoringSubroutineModelB()


amp_amp.recursively_assign_requirements_to_all_subtasks()

# print("reqs are:", amp_amp.requirements.requirements_to_string())

graph = amp_amp.generate_requirements_dag()
graph.view(cleanup=True)  # This will open the generated diagram
