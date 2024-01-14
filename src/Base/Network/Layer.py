from typing import Dict, Tuple, List

from ..Base import Hashable
from .Variable import Variable
from .LayerModel import LayerModel


class LayerIOLink:
    def __init__(
        self,
        input_layer: "Layer",
        input_variable: Variable,
        output_layer: "Layer",
        output_variable: Variable,
    ):
        self.input_layer = input_layer
        self.input_variable = input_variable
        self.output_layer = output_layer
        self.output_variable = output_variable

    def __format__(self, format_spec):
        return (
            f"{self.__class__.__name__}({self.input_layer} - {self.input_variable}"
            + f"-> {self.output_layer} - {self.output_variable})"
        )


class Layer(Hashable):
    """
    Represents a layer in a neural network model.

    Args:
        model (LayerModel): The model used in the layer.

    Attributes:
        model (LayerModel): The model used in the layer.
        inputs (dict): A dictionary mapping input variables to tuple of input layer and input variable.
        outputs (dict): A dictionary mapping output variables to list of tuples of output layer and output variable.

    """

    def __init__(self, model: LayerModel):
        super(Layer, self).__init__()
        self.model = model
        model.attach_layer(self)
        self.inputs: Dict[Variable, Tuple[Layer, Variable]] = {}
        self.outputs: Dict[Variable, List[Tuple[Layer, Variable]]] = {}

    def set_io_links(self, links: LayerIOLink | list[LayerIOLink]):
        if isinstance(links, LayerIOLink):
            links = [links]
        elif isinstance(links, list):
            pass
        else:
            raise TypeError(
                "incorrect type for io links of Layers : {type}".format(
                    type=type(links)
                )
            )
        for link in links:
            if link.input_layer == self:
                variable_outputs = self.outputs.setdefault(link.input_variable, [])
                variable_outputs.append((link.output_layer, link.output_variable))

            elif link.output_layer == self:
                self.inputs[link.output_variable] = (
                    link.input_layer,
                    link.input_variable,
                )
            else:
                raise ValueError(
                    (
                        "Trying to set io links on a layer {variable} "
                        + "that does not appear on the link {link}"
                    ).format(link=link, variable=self)
                )

            link.input_variable.add_linked_variables(link.output_variable)
            link.output_variable.add_linked_variables(link.input_variable)

    def remove_io_link(self, link: LayerIOLink) -> None:
        if link.input_layer == self:
            self.outputs[link.input_variable].remove(
                (link.output_layer, link.output_variable)
            )
        if link.output_layer == self:
            self.inputs[link.output_variable] = None

    def get_inputs(self) -> Dict[Variable, Tuple["Layer", Variable]]:
        return self.inputs

    def get_input_layers(self) -> List["Layer"]:
        return [layer for layer, variable in self.inputs.values()]

    def get_outputs(self) -> Dict[Variable, List[Tuple["Layer", Variable]]]:
        return self.outputs

    def get_output_layers(self) -> List["Layer"]:
        return list(
            set(
                [
                    layer
                    for outputs in self.outputs.values()
                    for layer, variable in outputs
                ]
            )
        )

    def get_model(self) -> LayerModel:
        return self.model

    def is_following_from(self, layer: "Layer") -> bool:
        input_layers = self.get_input_layers()
        if not input_layers:
            return False
        if layer in input_layers:
            return True
        else:
            return any(
                [
                    input_layer.is_following_from(layer)
                    for input_layer in self.get_input_layers()
                ]
            )

    def is_followed_by(self, layer: "Layer") -> bool:
        return layer.is_following_from(self)

    def get_all_previous_layers(self) -> List["Layer"]:
        previous_layers = []

        # The input layers are previous layers
        input_layers = self.get_input_layers()
        previous_layers.extend(input_layers)

        # the previous layers from inputs layers are previous layers
        for layer in input_layers:
            previous_layers.extend(layer.get_all_previous_layers())
        return list(set(previous_layers))

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}-{self.hash} of model{self.model})"
