class ONNXRegistry:
    """ONNX operator registry.

    This class is responsible for managing the registration and lookup of ONNX operators.
    It maintains a mapping from operator types to their ONNX conversion functions.
    All operator names are converted to lowercase for consistency.
    """

    def __init__(self):
        """Initialize the registry."""
        self._op_mapping = {}

    def register(self, op_type, symbolic_fn):
        """Register an ONNX conversion function for an operator.

        Args:
            op_type (str): Operator type name
            symbolic_fn (callable): Function to convert the operator to ONNX nodes.
                                  This function should accept two parameters: node and name_manager,
                                  and return a list of ONNX nodes.
        """
        self._op_mapping[op_type.lower()] = symbolic_fn

    def get(self, op_type):
        """Get the ONNX conversion function for an operator.

        Args:
            op_type (str): Operator type name

        Returns:
            callable: Conversion function, or None if not found
        """
        return self._op_mapping.get(op_type.lower())

    def list_supported_ops(self):
        """List all supported operators.

        Returns:
            list[str]: List of supported operator types
        """
        return list(self._op_mapping.keys())


# Global registry instance
registry = ONNXRegistry()
