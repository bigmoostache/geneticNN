from typing import Dict, List, Set, Literal, Optional
from typing import TYPE_CHECKING


from ..Base import Hashable
from .Parameter import Parameter, ParameterListener


if TYPE_CHECKING:
    from ..modelskeleton import LayerModel


class VariableInstance(Hashable):
    """
    This class represents an instance of a variable. It has its own parameters and can be linked to other variable instances:
    If different layers in a network use the same model, then each layer would imply a different link between the input/output variables of the model.
    and the other variables in the network.

    Parameters
    ----------
    source : Variable
        The source variable. The one from which the instance is created.
    hook : Hashable
        The hook of the instance. The object to which the instance is attached (most likely a layer).
    """

    def __init__(self, source: "Variable", hooks: List[Hashable]):
        """
        Initializes a new instance of the VariableInstance class.

        Parameters
        ----------
        source : Variable
            The source variable. The one from which the instance is created.
        hook : Hashable
            The hook of the instance. The object to which the instance is attached (most likely a layer).
        """
        super().__init__()
        self.source: "Variable" = source
        self.hooks: List[Hashable] = hooks
        self.instance_parameters: Dict[Parameter, Parameter] = {}
        self.linked_instances: Set["VariableInstance"] = set()

    def instantiate_parameters(self, parameters: List[Parameter]) -> None:
        """
        Create a new instance of the parameters owned by the instance.

        Parameters
        ----------
        parameters : List[Parameter]
            The parameters to instantiate.

        Returns
        -------
        None

        """
        for param in parameters:
            instance_param = param.copy()
            instance_param.attach_parent(self)
            self.instance_parameters[param] = instance_param

    def add_instance_link(self, instance: "VariableInstance") -> None:
        """
        Adds a link between the instance and another instance.

        Parameters
        ----------
        instance : VariableInstance
            The instance to link to.

        Returns
        -------
        None

        """
        self.linked_instances.add(instance)

    def __delete__(self, instance):
        """
        Safe delete of the instance.

        Parameters
        ----------
        instance

        Returns
        -------

        """
        for other in self.linked_instances:
            if instance in other.linked_instances:
                other.linked_instances.remove(instance)
        for param in list(self.instance_parameters.keys()):
            del self.instance_parameters[param]


class Variable(Hashable, ParameterListener):
    """
    This class a variable in model used as input or output. It can be linked to other variables in the network.

    Attributes
    ----------
    name : str
        The name of the variable.
    dimension : int
        The number of dimensions of the variable.
    variable_io : Literal["in", "out"]
        The type of the variable. Either "in" for an input variable or "out" for an output variable.
    data_type : type
        The type of the data stored in the variable.
    linked_variables : set[Variable]
        The variables linked to this variable.

    Methods
    -------
    make_new_instance(hook)
        Creates a new instance of the variable.

    remove_instance(hook)
        Removes an instance of the variable linked a hook.


    add_linked_variables(linked_variables)
            Adds a variable or a list of variables to the linked variables.

    attach_model(model)
        Attaches the variable to a model.

    __format__(format_spec)
        Formats the variable as a string.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        variable_io: Literal["in", "out"],
        data_type: type,
        attached_model: Optional["LayerModel"],
        linked_variables: Optional[set["Variable"]],
        instantiable: bool = True,
    ):
        """
        Initializes a new instance of the Variable class.

        Parameters
        ----------
        name : str
            The name of the variable.
        dimension : int
            The number of dimensions of the variable.
        variable_io : Literal["in", "out"]
            The type of the variable. Either "in" for an input variable or "out" for an output variable.
        data_type : type
            The type of the data stored in the variable.
        attached_model : LayerModel
            The model to which the variable is attached.
        linked_variables : set[Variable]
            The variables linked to this variable.
        instantiable : bool
            Whether the variable can be instanciated or not.
        """
        super(Variable, self).__init__()
        self.name: str = name
        self.dimension: int = dimension
        self.variable_io: Literal["in", "out"] = variable_io
        self.data_type = data_type
        self.linked_variables: set["Variable"] = linked_variables
        self.attached_model: LayerModel = attached_model
        self.global_parameters: List[Parameter] = []
        self.instances: Dict[Hashable, VariableInstance] = {}
        self.instantiable: bool = instantiable

        if linked_variables is None:
            self.linked_variables = set()
        else:
            for var in linked_variables:
                var.add_linked_variables(self)

    def make_new_instance(self, hook: Hashable) -> "VariableInstance":
        """
        Creates a new instance of the variable and links it to a hook.

        Parameters
        ----------
        hook : Hashable
            The hook to which the instance is linked.

        Returns
        -------
        VariableInstance
            The new instance of the variable.
        """
        if not self.instantiable:
            if len(self.instances) == 0:
                self.instances[self] = VariableInstance(source=self, hooks=[self, hook])
            else:
                self.instances[self].hooks.append(hook)
            return self.instances[self]
        else:
            self.instances[hook] = VariableInstance(source=self, hooks=[hook])
            return self.instances[hook]

    def remove_instance(self, hook: Hashable) -> None:
        if not self.instantiable:
            if len(self.instances) == 0:
                raise Exception("No instance to remove!")
            else:
                instance = self.instances[self]
                instance.hooks.remove(hook)
                if len(instance.hooks) == 0:
                    del self.instances[self]
        else:
            if hook not in self.instances:
                raise Exception("No instance to remove!")
            instance = self.instances[hook]
            for other in instance.linked_instances:
                if instance in other.linked_instances:
                    other.linked_instances.remove(instance)
            del self.instances[hook]

    def add_linked_variables(
        self, linked_variables: "Variable" | List["Variable"]
    ) -> None:
        """
        Adds a variable or a list of variables to the linked variables.

        Parameters
        ----------
        linked_variables : Variable | List[Variable]
            The variables to add to the linked variables.

        Returns
        -------
        None
        """
        if isinstance(linked_variables, Variable):
            self.linked_variables.add(linked_variables)

        elif isinstance(linked_variables, list):
            self.linked_variables = self.linked_variables.union(set(linked_variables))
        else:
            raise TypeError(
                "Type not supported for linking variables: {type(linked_variables)}!"
            )

    def attach_model(self, model: "LayerModel") -> None:
        """
        Attaches the variable to a model.

        Parameters
        ----------
        model : LayerModel
            The model to which the variable is attached.

        Returns
        -------
        None
        """
        if self.attached_model is None or self.attached_model == model:
            self.attached_model = model
        else:
            raise Exception("Model already attached!")

    def make_instantiable(self, instantiable: bool):
        if self.instances:
            raise Exception(
                "Cannot change instantiability of a variable with instances!"
            )
        self.instantiable = instantiable

    def __format__(self, format_spec) -> str:
        return f"{self.__class__.__name__}-{self.hash}('{self.name}' in {self.attached_model})"
