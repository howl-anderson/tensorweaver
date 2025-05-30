import numpy as np
from onnx import TensorProto


class ONNXTypeMapper:
    """Maps NumPy data types to ONNX data types.

    This class provides a static mapping table and methods for converting NumPy data types to ONNX-supported data types.
    When encountering unsupported data types, it raises a ValueError.
    """

    _DTYPE_MAP = {
        np.float32: TensorProto.FLOAT,
        np.float64: TensorProto.DOUBLE,
        np.int32: TensorProto.INT32,
        np.int64: TensorProto.INT64,
    }

    @classmethod
    def get_type(cls, dtype):
        """Get the corresponding ONNX data type.

        Args:
            dtype: NumPy data type object

        Returns:
            int: ONNX data type enumeration value

        Raises:
            ValueError: When the input data type is not supported
        """
        if dtype.type not in cls._DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Supported types are: {list(cls._DTYPE_MAP.keys())}"
            )
        return cls._DTYPE_MAP[dtype.type]
