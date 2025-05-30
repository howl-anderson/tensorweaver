from onnx.helper import make_node
from tensorweaver.onnx.registry import registry
from tensorweaver.operators.add import Add


@registry.register_converter(Add)
def onnx_add(node, name_manager):
    """Convert Add operation to ONNX node.

    Args:
        node: Add operation node
        name_manager (NameManager): Variable name manager

    Returns:
        list[NodeProto]: List of ONNX nodes, containing one Add node
    """
    # Ensure inputs and outputs have names
    input_names = [name_manager.get_name(input_var) for input_var in node.inputs]
    output_names = [name_manager.get_name(i) for i in node.outputs]

    return [make_node("Add", input_names, output_names)]
