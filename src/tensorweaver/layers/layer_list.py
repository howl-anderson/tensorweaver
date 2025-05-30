from collections import UserList


class LayerList(UserList):
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        super().__init__(layers)
