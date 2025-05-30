# pytorch reference: https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export

from dataclasses import dataclass
from typing import List

# auto import all ops, so that they can register themselves, keep this unused import
import tensorweaver.onnx.ops  # noqa: F401

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.helper import make_graph, make_tensor_value_info

from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.topological_sort import topological_sort
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.onnx.registry import registry
from tensorweaver.onnx.utils import ONNXTypeMapper, NameManager
from tensorweaver.parameter import Parameter


@dataclass
class GraphInfo:
    """Information about the computational graph structure.

    Attributes:
        inputs: List of input variables
        outputs: List of output variables
        parameters: List of parameters
        intermediate_tensors: List of intermediate tensors
        functions: List of function nodes
    """

    inputs: List[Tensor]
    outputs: List[Tensor]
    parameters: List[Parameter]
    intermediate_tensors: List[Tensor]
    functions: List[Operator]


def collect_graph_info(inputs: List[Tensor], output: Tensor) -> GraphInfo:
    """Collect information about all components in the computational graph.

    Traverses the computational graph based on topological sorting, collecting all
    inputs, outputs, parameters, intermediate tensors, and function nodes.
    The returned function list is guaranteed to be topologically sorted
    (if A's output is B's input, then A comes before B).

    Args:
        inputs: List of input variables
        output: Output variable

    Returns:
        GraphInfo: Information containing all components of the computational graph
    """
    # Get topologically sorted list of variables
    variables = topological_sort(output)

    # Collect all Function nodes, maintaining topological order
    # Since variables are topologically sorted, collecting their creators in order
    # ensures that the functions are also topologically sorted
    functions = []
    seen_functions = set()
    for var in variables:
        if var.creator and id(var.creator) not in seen_functions:
            functions.append(var.creator)
            seen_functions.add(id(var.creator))

    # Collect parameters and intermediate tensors
    parameters = []
    intermediate_tensors = []
    input_set = set(inputs)

    for var in variables:
        if var in input_set:
            continue
        elif isinstance(var, Parameter):
            parameters.append(var)
        elif (
            var != output
        ):  # Not an input, parameter, or output, so it's an intermediate tensor
            intermediate_tensors.append(var)

    return GraphInfo(
        inputs=inputs,
        outputs=[output],
        parameters=parameters,
        intermediate_tensors=intermediate_tensors,
        functions=functions,
    )


def convert_variable(var, name_manager):
    """Convert a Variable to ONNX ValueInfoProto.

    Args:
        var (Tensor): Variable to convert
        name_manager (NameManager): Variable name manager

    Returns:
        ValueInfoProto: ONNX variable information
    """
    return make_tensor_value_info(
        name_manager.get_name(var),
        ONNXTypeMapper.get_type(var.data.dtype),
        var.data.shape,
    )


def convert_parameter(var, name_manager):
    """Convert a Parameter to ONNX TensorProto.

    Args:
        var (Parameter): Parameter to convert
        name_manager (NameManager): Variable name manager

    Returns:
        TensorProto: ONNX tensor
    """
    return numpy_helper.from_array(var.data, name=name_manager.get_name(var))


class ONNXProgram:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    @classmethod
    def export(cls, model, args, f=None):
        """Export the model to ONNX format.

        Args:
            model: The model to export
            args: List of input parameters for the model

        Returns:
            ONNXProgram: ONNX program

        Raises:
            TypeError: If the model output is not a Variable type
            NotImplementedError: If an unsupported operator is encountered
        """
        # Create name manager
        name_manager = NameManager()

        # 1. Run the model to get output
        output = model(*args)
        if not isinstance(output, Tensor):
            raise TypeError(f"Model output must be a Variable, got {type(output)}")

        # 2. Collect computational graph information
        graph_info = collect_graph_info(list(args), output)

        # 3. Generate names for all variables
        all_vars = (
            graph_info.inputs
            + graph_info.outputs
            + graph_info.parameters
            + graph_info.intermediate_tensors
        )
        for var in all_vars:
            name_manager.get_name(var)

        # 4. Convert inputs and outputs
        inputs = [convert_variable(var, name_manager) for var in graph_info.inputs]
        outputs = [convert_variable(output, name_manager)]

        # 5. Convert parameters
        initializers = [
            convert_parameter(param, name_manager) for param in graph_info.parameters
        ]

        # 6. Convert intermediate variables
        value_infos = [
            convert_variable(var, name_manager)
            for var in graph_info.intermediate_tensors
        ]

        # 7. Convert function nodes
        nodes = []
        for func in graph_info.functions:
            op_onnx_fn = registry.get(func)
            if op_onnx_fn is None:
                supported_ops = registry.list_supported_ops()
                raise NotImplementedError(
                    f"ONNX export not implemented for {func.__class__.__name__}. "
                    f"Supported operators: {supported_ops}"
                )

            # Generate ONNX nodes
            onnx_nodes = op_onnx_fn(func, name_manager)
            nodes.extend(onnx_nodes)

        # 8. Create ONNX graph and model
        onnx_graph = make_graph(
            nodes,  # nodes
            "model",  # name
            inputs,  # inputs
            outputs,  # outputs
            initializers,  # initializer
            value_info=value_infos,  # information for all variables
        )

        # Determine a suitable IR version.
        # ONNXRuntime supports max IR version 10 according to the error.
        # Opset 21 is compatible with IR version 10 (introduced in ONNX 1.16.0).
        # For now, let's target IR version 10.
        target_ir_version = 10
        current_opset_version = 21  # As per your current code

        onnx_model = helper.make_model(
            onnx_graph,
            opset_imports=[helper.make_opsetid("", current_opset_version)],
            ir_version=target_ir_version,  # Explicitly set IR version
        )

        instance = cls(onnx_model)

        if f is not None:
            instance.save(f)

        return instance

    def save(self, destination):
        onnx.save(self.onnx_model, destination)


export = ONNXProgram.export
