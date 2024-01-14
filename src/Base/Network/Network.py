from typing import List, Dict, Tuple, Set, Optional

from ..Base import Hashable
from .LayerModel import LayerModel
from .Layer import Layer, LayerIOLink


class Network:
    """
    Network Class

    A class representing a neural network.

    Attributes:
    ----------
    models : List[LayerModel]
        List of models used in the network.
    layers : List[Layer]
        List of layers used in the network.
    input_layer : Layer
        The input layer of the network.
    output_layer : Layer
        The output layer of the network.

    Methods:
    -------
    check_network()
        Checks the network for cycles.
    add_layers_link(link: LayerIOLink)
        Adds a link between two layers.
    layers_heights() -> Dict[Layer, int]
        Returns the height of each layer in the network.
    layers_orders() -> List[Layer]
        Returns the layers ordered by their height in the network.
    layers_usage() -> Dict[Layer, bool]
        Returns a dictionary mapping each layer to a boolean indicating whether it is used or not.
    used_layers() -> List[Layer]
        Returns the list of used layers.
    unused_layers() -> List[Layer]
        Returns the list of unused layers.
    add_model(model: LayerModel)
        Adds a model to the network.
    add_layer(layer: Layer)
        Adds a layer to the network.
    remove_layer(layer: Layer)
        Removes a layer from the network.
    remove_ghost_layers()
        Removes the unused layers from the network.
    remove_ghost_models()
        Removes the unused models from the network.
    get_unique_variables()
        Returns the list of unique variables in the network.

    """

    def __init__(
        self,
        models: List[LayerModel],
        layers: List[Layer],
        input_layer: Layer,
        output_layer: Layer,
    ):
        """
        Initializes a new instance of the Network class.

        Parameters
        ----------
        models : List[LayerModel]
            List of models used in the network.
        layers : List[Layer]
            List of layers used in the network.
        input_layer : Layer
            The input layer of the network.
        output_layer : Layer
            The output layer of the network.

        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = list(set(layers))
        self.models = models
        self.check_network()

    def check_network(self):
        """
        Check for network validity.

        Returns
        -------
        None
            Raises an exception if the network is invalid.
        """
        for layer in self.layers:
            if layer.is_following_from(layer):
                raise Exception(f"There is a cycle in the graph of layers!")

    def add_layers_link(self, link: LayerIOLink):
        """
        Adds a link between two layers.

        Parameters
        ----------
        link : LayerIOLink
            The link to add.

        Returns
        -------
        None
        """
        link.input_layer.set_io_links(link)
        link.output_layer.set_io_links(link)

    @property
    def layers_heights(self):
        """
        Returns the height of each layer in the network with respect to the output layer.

        Returns
        -------
        Dict[Layer, int]
            A dictionary mapping each layer to its height in the network.
        """
        n_layers = len(self.layers)
        heights = {layer: -1 for layer in self.layers}

        def set_height(layer: Layer, height):
            if height > n_layers:
                raise Exception("Cycle detected in the graph")
            if height > heights[layer]:
                heights[layer] = height
                for input_layer in layer.get_input_layers():
                    set_height(input_layer, height + 1)

        set_height(self.output_layer, 0)
        return heights

    @property
    def layers_orders(self):
        """
        Returns the layers ordered by their height in the network.

        Returns
        -------
        List[Layer]
            The layers ordered by their height in the network.
        """
        # compute height of layers
        heights = self.layers_heights
        max_height = max(heights.values())

        # order layers by their height in the graph
        ordered_layers = []
        for depth in range(max_height + 1):
            for layer in self.layers:
                if heights[layer] == max_height - depth:
                    ordered_layers.append(layer)

        if ordered_layers[0] != self.input_layer:
            raise Exception(
                f" Bad order of runs, the first layer is not the input layer but {ordered_layers[0]}"
            )

        if ordered_layers[-1] != self.output_layer:
            raise Exception(
                f" Bad order of runs, the last layer is not the output layer but {ordered_layers[-1]}"
            )

        return ordered_layers

    @property
    def layers_usage(self):
        """
        Returns a dictionary mapping each layer to a boolean indicating whether it is used or not.

        Returns
        -------
        Dict[Layer, bool]
            A dictionary mapping each layer to a boolean indicating whether it is used or not.
        """
        heights = self.layers_heights
        return {layer: (height == -1) for layer, height in heights.items()}

    @property
    def used_layers(self):
        """
        Returns the list of used layers.

        Returns
        -------
        List[Layer]
            The list of used layers.
        """
        return [layer for layer, used in self.layers_usage.items() if used]

    @property
    def unused_layers(self):
        """
        Returns the list of unused layers.

        Returns
        -------
        List[Layer]
            The list of unused layers.
        """
        return [layer for layer, used in self.layers_usage.items() if not used]

    def add_model(self, model: LayerModel):
        """
        Adds a model to the network.

        Parameters
        ----------
        model : LayerModel
            The model to add.

        Returns
        -------
        None
        """
        if model not in self.models:
            self.models.append(model)

    def add_layer(self, layer: Layer):
        """
        Adds a layer to the network.

        Parameters
        ----------
        layer: Layer
            The layer to add.

        Returns
        -------
        None
        """
        if layer not in self.layers:
            self.layers.append(layer)
        if layer.model not in self.models:
            self.add_model(layer.model)

    def remove_layer(self, layer: Layer):
        """
        Removes a layer from the network.

        Parameters
        ----------
        layer : Layer
            The layer to remove.

        Returns
        -------
        None
        """

        # remove all links with the layers to delete from its inputs
        for variable, (input_layer, input_var) in layer.get_inputs().items():
            link = LayerIOLink(input_layer, input_var, layer, variable)
            input_layer.remove_io_link(link)

        # remove all links with this layer from its outputs
        for variable, outputs in layer.get_outputs():
            for output_layer, output_var in outputs:
                link = LayerIOLink(layer, variable, output_layer, output_var)
                output_layer.remove_io_link(link)

        layer.get_model().detach_layer(layer)

        self.layers.remove(layer)

    def remove_ghost_layers(self):
        """
        Removes the unused layers from the network.

        Returns
        -------
        None
        """
        layers_usage = self.layers_usage
        unused_layers_indices = [
            layer for layer in self.layers if not layers_usage[layer]
        ]
        # used_layers = [layer for layer in self.layers if layers_usage[layer]]

        for layer in self.unused_layers:
            self.remove_layer(layer)

    def remove_ghost_models(self):
        """
        Removes the unused models from the network.

        Returns
        -------
        None
        """

        used_models = [
            i for i, model in enumerate(self.models) if model.get_attached_layers()
        ]
        self.models = used_models

    def get_unique_variables(self):
        pass
