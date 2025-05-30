class NameManager:
    """Manages variable names during ONNX export.

    This class is responsible for generating and managing unique names for variables in the computational graph. It will:
    1. Preserve original variable names (if they exist)
    2. Generate names for variables without names
    3. Handle name conflicts, ensuring each variable has a unique name
    """

    def __init__(self):
        """Initialize the name manager."""
        self.var_to_name = {}  # Map from variables to names
        self.used_names = set()  # Set of already used names
        self.counter = 0  # Counter for generating unique names

    def get_name(self, var):
        """Get a unique name for a variable.

        Try to keep the original name if the variable already has one.
        If the variable has no name or the name is already used, generate a new unique name.

        Args:
            var: Variable object that needs a name

        Returns:
            str: Unique name for the variable
        """
        # If the variable already has a mapped name, return it directly
        if var in self.var_to_name:
            return self.var_to_name[var]

        # Determine the base name
        if var.name:
            base_name = var.name
        else:
            base_name = f"tensor_{self.counter}"
            self.counter += 1

        # Ensure the name is unique
        final_name = base_name
        while final_name in self.used_names:
            final_name = f"{base_name}_{self.counter}"
            self.counter += 1

        # Save and return the final name
        self.var_to_name[var] = final_name
        self.used_names.add(final_name)
        return final_name
