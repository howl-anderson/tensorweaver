from typing import List
import collections

from tensorweaver.autodiff.tensor import Tensor


def topological_sort(var: Tensor) -> List[Tensor]:
    """
    Using BFS to get topological sort of graph variables
    """

    queue = collections.deque([var])
    visited = {var}

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Skip if node is a leaf node (no creator)
        if node.creator is None:
            continue

        for next_node in node.creator.inputs:
            if next_node in visited:
                continue

            queue.append(next_node)
            visited.add(next_node)

    return result
