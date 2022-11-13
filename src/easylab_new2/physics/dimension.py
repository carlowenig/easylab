from scipy import constants


class Dimension:
    def __init__(self, **components: int):
        self.components = components
