# reference: https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul

from onnx.helper import make_node
from tensorweaver.onnx.registry import registry
from tensorweaver.operators.matmul import Matmul


@registry.register_converter(Matmul)
def onnx_matmul(node, name_manager):
    """Convert Matmul operation to ONNX node.

    Args:
        node: Matmul operation node
        name_manager (NameManager): Variable name manager

    Returns:
        list[NodeProto]: List of ONNX nodes, containing one Matmul node
    """
    # Ensure inputs and outputs have names
    input_names = [name_manager.get_name(input_var) for input_var in node.inputs]
    output_names = [name_manager.get_name(i) for i in node.outputs]

    return [make_node("MatMul", input_names, output_names)]
