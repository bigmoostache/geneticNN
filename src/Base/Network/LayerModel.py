from typing import List, Literal, Union, Optional
from typing import TYPE_CHECKING

import json

from ..Base import Hashable
from .Parameter import Parameter, ParameterListener
from .Variable import Variable, VariableInstance

if TYPE_CHECKING:
    from ..modelskeleton import ModelTemplate
    from .Layer import Layer


class LayerModel(Hashable):
    """
    LayerModel Class

    A class representing a model used in a layer of a neural network.

    Attributes:
        name (str): The name of the layer model.
        template (ModelTemplate): The model template associated with the layer model.
        input_variables (list): List of input variables attached to the layer model.
        output_variables (list): List of output variables attached to the layer model.
        attached_layers (list): List of neural layers attached to the layer model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the LayerModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the layer model.

        get_input_variables() -> list
            Returns the list of input variables attached to the layer model.

        get_output_variables() -> list
            Returns the list of output variables attached to the layer model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the layer model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the layer model.
    """

    def __init__(
        self,
        name: str,
        input_variables: Optional[List[Variable]] = None,
        output_variables: Optional[List[Variable]] = None,
    ):
        """
        Initializes a new instance of the LayerModel class.

        Parameters
        ----------
        name : str
            The name of the layer model.
        template : ModelTemplate
            The model template to use for the model.
        """
        super(LayerModel, self).__init__()
        self.name = name
        self.input_variables = input_variables or []
        self.output_variables = output_variables or []
        self.attached_layers = []

    def attach_variables(
        self,
        variables: Variable | List[Variable],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            The place to attach the variables. Either "in" for input variables or "out" for output variables.
            This parameter is defaulted to "in" and is not accounted for in input and output models.

        Returns
        -------

        """
        if isinstance(variables, Variable):
            if where == "in":
                self.input_variables.append(variables)
            elif where == "out":
                self.output_variables.append(variables)
            else:
                raise ValueError(
                    "incorrect place to put the variable : {where}".format(where=where)
                )
            variables.attach_model(self)

        elif isinstance(variables, list):
            if where == "in":
                self.input_variables.extend(variables)
            elif where == "out":
                self.output_variables.extend(variables)
            else:
                raise ValueError(
                    "incorrect place to put the variables : {where}".format(where=where)
                )
            for variable in variables:
                variable.attach_model(self)

    def attach_layer(self, layer: "Layer"):
        if isinstance(layer, Layer):
            if layer not in self.attached_layers:
                self.attached_layers.append(layer)
                for variable in self.input_variables:
                    variable.make_new_instance(layer)
                for variable in self.output_variables:
                    variable.make_new_instance(layer)
        else:
            raise TypeError(
                "incorrect type for layers to attach : {type}".format(type=type(layer))
            )

    def detach_layer(self, layer: Layer):
        if layer in self.attached_layers:
            self.attached_layers.remove(layer)
            for variable in self.input_variables:
                variable.remove_instance(layer)
            for variable in self.output_variables:
                variable.remove_instance(layer)

    def get_input_variables(self):
        return self.input_variables

    def get_output_variables(self):
        return self.output_variables

    def get_attached_layers(self):
        return self.attached_layers

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}-{self.hash}({self.name})"


class InputModel(LayerModel):
    """
    InputModel Class

    The model to use for the input layer of a neural network.

    Attributes:
        name (str): The name of the input model.
        template (ModelTemplate): The model template associated with the input model.
        input_variables (list): List of input variables attached to the input model.
        output_variables (list): List of output variables attached to the input model.
        attached_layers (list): List of neural layers attached to the input model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the InputModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the input model.

        get_input_variables() -> list
            Returns the list of input variables attached to the input model.

        get_output_variables() -> list
            Returns the list of output variables attached to the input model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the input model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the input model.
    """

    def __init__(
        self,
        name: str,
        output_variables: Optional[List[Variable]] = None,
    ):
        """
        Initializes a new instance of the InputModel class.

        Parameters
        ----------
        name : str
            The name of the input model.
        output_variables : Optional[List[Variable]]
            The output variables to attach to the model.

        """
        super(InputModel, self).__init__(
            name, input_variables=None, output_variables=output_variables
        )

    def attach_variables(
        self,
        variables: Variable | List[Variable],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model. This models neglects the "where" parameter and only attaches the variables
        as output variables.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            Unused parameter

        Returns
        -------

        """

        if isinstance(variables, Variable):
            self.output_variables.append(variables)
            variables.attach_model(self)
            variables.make_instantiable(False)

        elif isinstance(variables, list):
            self.output_variables.extend(variables)
            for variable in variables:
                variable.attach_model(self)
                variable.make_instantiable(False)


class OutputModel(LayerModel):
    """
    InputModel Class

    The model to use for the input layer of a neural network.

    Attributes:
        name (str): The name of the input model.
        input_variables (list): List of input variables attached to the input model.
        output_variables (list): List of output variables attached to the input model.
        attached_layers (list): List of neural layers attached to the input model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the InputModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the input model.

        get_input_variables() -> list
            Returns the list of input variables attached to the input model.

        get_output_variables() -> list
            Returns the list of output variables attached to the input model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the input model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the input model.
    """

    def __init__(
        self,
        name: str,
        input_variables: Optional[List[Variable]] = None,
    ):
        """
        Initializes a new instance of the InputModel class.

        Parameters
        ----------
        name : str
            The name of the input model.
        input_variables : Optional[List[Variable]]
            The input variables to attach to the model.
        """
        super(OutputModel, self).__init__(
            name, input_variables=input_variables, output_variables=[]
        )

    def attach_variables(
        self,
        variables: Variable | List[Variable],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model. This models neglects the "where" parameter and only attaches the variables
        as input variables.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            Unused parameter

        Returns
        -------

        """

        if isinstance(variables, Variable):
            self.input_variables.append(variables)
            variables.attach_model(self)
            variables.make_instantiable(False)

        elif isinstance(variables, list):
            self.input_variables.extend(variables)
            for variable in variables:
                variable.attach_model(self)
                variable.make_instantiable(False)


class TemplatedModel(LayerModel):
    def __init__(self, name: str, template_type, source, template_source=""):
        super(TemplatedModel, self).__init__(name)
        self.template_type = template_type
        self.source = source
        self.template_source = template_source
        self._props = self.load_template_props(template_source)

        # build variables and parameters

    def load_template_props(self, source_file: str):
        text = open(source_file).read()
        text2 = text.split("BEGIN_PROPS")[1]
        text2 = text2.split("END_PROPS")[0]
        return json.loads(text2)
