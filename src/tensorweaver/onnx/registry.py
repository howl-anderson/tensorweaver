class ONNXRegistry:
    def __init__(self):
        self._op_mapping = {}

    def register(self, func_cls, symbolic_fn):
        name = func_cls.__name__
        self._op_mapping[name] = symbolic_fn

    def register_converter(self, func_class):
        def decorator(func):
            self.register(func_class, func)
            return func

        return decorator

    def get(self, func_instance):
        name = func_instance.__class__.__name__
        return self._op_mapping.get(name)

    def list_supported_ops(self):
        return list(self._op_mapping.keys())


registry = ONNXRegistry()
