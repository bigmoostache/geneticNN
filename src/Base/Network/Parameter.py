from ..Base import Hashable
from typing import Any, List


class ParameterListener:
    def __init__(self):
        pass


class Parameter(Hashable):
    """
    This class represents a parameter used in a neural network model.

    :class:`Parameter` inherits from the `Hashable` class.

    Attributes:
        :name (str): Name of the parameter.
        :parameter_type: Type of the parameter.
        :parent: Object to which the parameter is attached.
        the object responsible for the parameter.
        :listeners: List of objects that listen to the parameter.

    Methods:
        attach_parent(model): Attaches the parameter to an Object.
        __format__(format_spec): Formats the parameter as a string.

    Example:
        parameter = Parameter('weight', 'float')
        parameter.attach_model(model)
        parameter.__format__('')
    """

    def __init__(
        self,
        name: str,
        parameter_type,
        parent: Any | None = None,
        listeners: list[ParameterListener] = None,
    ):
        super(Parameter, self).__init__()
        if listeners is None:
            listeners = []
        self.name = name
        self.parameter_type = parameter_type
        self.parent = parent
        self.listeners = listeners
        self.cluster: int = self.hash

    def attach_parent(self, parent) -> None:
        if self.parent is None or self.parent == parent:
            self.parent = parent
        else:
            raise Exception("Model already attached!")

    def __format__(self, format_spec) -> str:
        return f"{self.__class__.__name__}-{self.hash}('{self.name}' in {self.parent})"

    @staticmethod
    def solve_cluster(list_params: List["Parameter"]):
        if Parameter.check_compatibility(list_params):
            min_cluster = min([param.cluster for param in list_params])
            for param in list_params:
                param.cluster = min_cluster

    @staticmethod
    def check_compatibility(list_params: List["Parameter"]):
        return True

    def copy(self):
        return Parameter(self.name, self.parameter_type)

    def __delete__(self, instance):
        pass
