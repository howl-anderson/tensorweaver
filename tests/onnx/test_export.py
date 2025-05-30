import numpy as np
import onnx
import onnxruntime as ort
import pytest
import tempfile

import tensorweaver as tw
from tensorweaver.onnx.onnx_program import export
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.autodiff.operator import Operator
from tensorweaver.operators.add import Add
from tensorweaver import Parameter


class SimpleAddModel(tw.nn.Module):
    def forward(self, x, y):
        return x + y


@pytest.fixture
def model():
    return SimpleAddModel()


def test_simple_add(model):
    """Test simple addition model export"""
    # Create test inputs
    x = Tensor(np.array([1, 2, 3], dtype=np.float32), name="input_1")
    y = Tensor(np.array([4, 5, 6], dtype=np.float32), name="input_2")

    # Export model using temporary file
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        export_path = f.name
        program = export(model, (x, y))
        program.save(export_path)

        # Verify model structure
        onnx_model = onnx.load(export_path)
        assert len(onnx_model.graph.node) == 1  # Should have only one Add node
        node = onnx_model.graph.node[0]
        assert node.op_type == "Add"
        assert len(node.input) == 2
        assert len(node.output) == 1

        # Verify input and output names
        assert node.input[0] == "input_1"
        assert node.input[1] == "input_2"

        # Verify model validity
        onnx.checker.check_model(onnx_model)

        # Run model using ONNX Runtime
        session = ort.InferenceSession(export_path)
        ort_inputs = {"input_1": x.data, "input_2": y.data}
        ort_outputs = session.run(None, ort_inputs)

        # Verify results
        expected_output = x.data + y.data
        np.testing.assert_allclose(ort_outputs[0], expected_output)


def test_add_with_broadcast(model):
    """Test addition model export with broadcasting"""
    # Create test inputs (with broadcasting)
    x = Tensor(np.array([[1, 2, 3]], dtype=np.float32), name="input_1")  # shape: (1, 3)
    y = Tensor(np.array([4, 5, 6], dtype=np.float32), name="input_2")  # shape: (3,)

    # Export model using temporary file
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        export_path = f.name
        program = export(model, (x, y))
        program.save(export_path)

        # Verify model structure
        onnx_model = onnx.load(export_path)
        assert len(onnx_model.graph.node) == 1  # Should have only one Add node
        node = onnx_model.graph.node[0]
        assert node.op_type == "Add"

        # Verify inputs and outputs
        input1 = [t for t in onnx_model.graph.input if t.name == "input_1"][0]
        input2 = [t for t in onnx_model.graph.input if t.name == "input_2"][0]
        output = onnx_model.graph.output[0]

        # Verify tensor shapes
        assert input1.type.tensor_type.shape.dim[0].dim_value == 1  # batch size
        assert input1.type.tensor_type.shape.dim[1].dim_value == 3  # sequence length
        assert len(input2.type.tensor_type.shape.dim) == 1  # one-dimensional tensor
        assert input2.type.tensor_type.shape.dim[0].dim_value == 3

        # Verify model validity
        onnx.checker.check_model(onnx_model)

        # Run model using ONNX Runtime
        session = ort.InferenceSession(export_path)
        ort_inputs = {"input_1": x.data, "input_2": y.data}
        ort_outputs = session.run(None, ort_inputs)

        # Verify results
        expected_output = x.data + y.data
        np.testing.assert_allclose(ort_outputs[0], expected_output)


def verify_onnx_model(model, args, expected_output):
    """Helper function to verify ONNX model output matches expected output."""
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        # Export model
        program = export(model, args)
        program.save(f.name)

        # Run ONNX model
        sess = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
        input_feed = {f"tensor_{i}": arg.data for i, arg in enumerate(args)}
        onnx_output = sess.run(None, input_feed)[0]

        # Compare outputs
        np.testing.assert_allclose(onnx_output, expected_output.data, rtol=1e-5)


def test_simple_add():
    """Test basic binary addition operation.

    Computation formula:
        out = x + y

    Computational graph structure:
        x ---→
             Add --→ out
        y ---→

    Test points:
    1. Basic binary addition operation
    2. Input/output name mapping
    3. Verification of result correctness
    """

    class SimpleAdd(Operator):
        def __call__(self, x, y):
            return Add()(x, y)

    # Create inputs
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    args = [Tensor(x), Tensor(y)]

    # Run model and verify
    model = SimpleAdd()
    output = model(*args)
    verify_onnx_model(model, args, output)


def test_chain_add():
    """Test chained addition operations.

    Computation formula:
        out = (a + b) + c

    Computational graph structure:
        a ---→
             Add1 ---→
        b ---→         Add2 --→ out
                    ↗
        c ----------→

    Test points:
    1. Multi-level addition operations
    2. Processing of intermediate results
    3. Correctness of topological sorting
    """

    class ChainAdd(Operator):
        def __call__(self, a, b, c):
            temp = Add()(a, b)
            return Add()(temp, c)

    # Create inputs
    a = np.array([[1, 2]], dtype=np.float32)
    b = np.array([[3, 4]], dtype=np.float32)
    c = np.array([[5, 6]], dtype=np.float32)
    args = [Tensor(x) for x in [a, b, c]]

    # Run model and verify
    model = ChainAdd()
    output = model(*args)
    verify_onnx_model(model, args, output)


def test_tree_add():
    """Test tree-structured addition operations.

    Computation formula:
        out = (a + b) + (c + d)

    Computational graph structure:
        a ---→
             Add1 ---→
        b ---→         Add3 --→ out
                    ↗
        c ---→
             Add2 --→
        d ---→

    Test points:
    1. Parallel addition operations
    2. Multi-level computation
    3. Complex computational graph structure
    4. Processing of multiple intermediate results
    """

    class TreeAdd(Operator):
        def __call__(self, a, b, c, d):
            left = Add()(a, b)
            right = Add()(c, d)
            return Add()(left, right)

    # Create inputs
    inputs = [np.array([[i]], dtype=np.float32) for i in range(1, 5)]
    args = [Tensor(x) for x in inputs]

    # Run model and verify
    model = TreeAdd()
    output = model(*args)
    verify_onnx_model(model, args, output)


def test_diamond_add():
    """Test diamond-structured addition operations (variable reuse).

    Computation formula:
        t1 = x + y
        t2 = x + y  # reuse x and y
        out = t1 + t2

    Computational graph structure:
                 Add1 ---→
        x ----→↗         ↘
              ↘           Add3 --→ out
        y ----→ Add2 ---→↗

    Test points:
    1. Variable reuse
    2. Repeat same computation
    3. Variable sharing in computational graph
    4. Correctness of topological sorting for repeated nodes
    """

    class DiamondAdd(Operator):
        def __call__(self, x, y):
            left = Add()(x, y)
            right = Add()(x, y)  # reuse x and y
            return Add()(left, right)

    # Create inputs
    x = np.array([[1, 2]], dtype=np.float32)
    y = np.array([[3, 4]], dtype=np.float32)
    args = [Tensor(x), Tensor(y)]

    # Run model and verify
    model = DiamondAdd()
    output = model(*args)
    verify_onnx_model(model, args, output)


def test_parameter_add():
    """Test addition operation with parameters.

    Computation formula:
        out = x + param
        where param is a learnable parameter

    Computational graph structure:
        x -------→
                 Add --→ out
        param ---→
        (Parameter)

    Test points:
    1. Processing of learnable parameters
    2. Parameter initialization and export
    3. ONNX initializer handling
    4. Correct mapping of parameter names
    """

    class ParameterAdd(Operator):
        def __init__(self):
            super().__init__()
            self.param = Parameter(np.array([[1.0, 2.0]], dtype=np.float32))

        def __call__(self, x):
            return Add()(x, self.param)

    # Create input
    x = np.array([[3, 4]], dtype=np.float32)
    args = [Tensor(x)]

    # Run model and verify
    model = ParameterAdd()
    output = model(*args)
    verify_onnx_model(model, args, output)
