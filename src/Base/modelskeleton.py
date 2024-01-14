import copy
import warnings

from .Network.Layer import Layer
from .modeltemplate import *
from typing import List, Dict, Literal, Set

"""
A class that contains the bare-bone structure describing a model: the input and output of the model, 
the different layers used, the order in which variables are passed in layers and
the parameters of the model and their transcription to parameters of its layers

#Parameters of the class

"""


class Hashable:
    """
    A class that represents an object that can be hashed and compared for equality.

    Attributes:
        counter (int): A counter used to assign unique hash values to each instance.

    Methods:
        __init__: Initializes a new instance of the Hashable class.
        __hash__: Returns the hash value of the instance.
        __eq__: Compares the instance with another object for equality.

    """

    counter = 0

    def __init__(self):
        self.hash = Hashable.counter
        Hashable.counter += 1

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash


c


class Variable(Hashable, Parameter_listener):
    """

    :class: Variable

    This class represents a variable used in a model as input or output.

    **Parameters:**

    - `name` (str): The name of the variable.
    - `dimension` (int): The dimensionality of the variable.
    - `variable_io` (Literal['in', 'out']): The input/output type of the variable.
    - `data_type`: The data type of the variable.
    - `attached_model` (LayerModel | None, optional): The model to which the variable is attached. Defaults to None.
    - `linked_variables` (set[Variable] | None, optional): Set of linked variables from other models. Defaults to None.

    **Methods:**

    - `add_linked_variables(linked_variables) -> None`: Adds linked variables to the set of linked variables in the object.
        - `linked_variables` (Variable | list[Variable]): The variable or list of variables to add.

    - `attach_model(model) -> None`: Attaches the variable to a model.
        - `model`: The model to attach.

    - `__format__(format_spec) -> str`: Formats the variable as a string.
        - `format_spec`: The format specification.

    This class extends the `Hashable` class.
    """

    class Instance(Hashable):
        def __init__(self, source: Variable, hook: Hashable):
            super().__init__()
            self.source: Variable = source
            self.hook: Hashable = hook
            self.instance_parameters: Dict[Parameter, Parameter] = {}
            self.linked_instances: Set[Variable.Instance] = set()

        def instantiate_parameters(self, parameters: List[Parameter]) -> None:
            for param in parameters:
                instance_param = param.copy()
                instance_param.attach_parent(self)
                self.instance_parameters[param] = instance_param

        def add_instance_link(self, instance: Variable.Instance) -> None:
            self.linked_instances.add(instance)

        def __delete__(self, instance):
            for other in self.linked_instances:
                if instance in other.linked_instances:
                    other.linked_instances.remove(instance)
            for param in list(self.instance_parameters.keys()):
                del self.instance_parameters[param]

    def __init__(
        self,
        name: str,
        dimension: int,
        variable_io: Literal["in", "out"],
        data_type: type,
        attached_model: LayerModel | None = None,
        linked_variables: set[Variable] | None = None,
    ):
        super(Variable, self).__init__()
        self.name: str = name
        self.dimension: int = dimension
        self.variable_io: Literal["in", "out"] = variable_io
        self.data_type = data_type
        self.linked_variables: set[Variable] = linked_variables
        self.attached_model: LayerModel = attached_model
        self.global_parameters: List[Parameter] = []
        self.instances: Dict[Hashable, Variable.Instance] = {}

        if linked_variables is None:
            self.linked_variables = set()
        else:
            for var in linked_variables:
                var.add_linked_variables(self)

    def make_new_instance(self, hook: Hashable) -> Variable.Instance:
        self.instances[hook] = Variable.Instance(source=self, hook=hook)
        return self.instances[hook]

    def remove_instance(self, hook):
        instance = self.instances[hook]
        for other in instance.linked_instances:
            if instance in other.linked_instances:
                other.linked_instances.remove(instance)
        del self.instances[hook]

    def add_linked_variables(self, linked_variables) -> None:
        if isinstance(linked_variables, Variable):
            self.linked_variables.add(linked_variables)

        elif isinstance(linked_variables, list):
            self.linked_variables = self.linked_variables.union(set(linked_variables))
        else:
            raise TypeError(
                "Type not supported for linking variables: {type(linked_variables)}!"
            )

    def attach_model(self, model) -> None:
        if self.attached_model is None or self.attached_model == model:
            self.attached_model = model
        else:
            raise Exception("Model already attached!")

    def __format__(self, format_spec) -> str:
        return f"{self.__class__.__name__}-{self.hash}('{self.name}' in {self.attached_model})"


class NeuralNetwork:
    def __init__(
        self,
        models: List[LayerModel],
        layers: List[Layer],
        output_layer: Layer,
        input_layer: Layer,
    ):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = list(set(layers))
        self.models = models
        self.check_network()

    def check_network(self):
        for layer in self.layers:
            if layer.is_following_from(layer):
                raise Exception(f"There is a cycle in the graph of layers!")

    def add_layers_link(self, link: Layer.IOLink):
        link.input_layer.set_io_links(link)
        link.output_layer.set_io_links(link)

    @property
    def layers_heights(self):
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
        heights = self.layers_heights
        return {layer: (height == -1) for layer, height in heights.items()}

    @property
    def layers_used(self):
        return [layer for layer, used in self.layers_usage.items() if used]

    @property
    def unused_layers(self):
        return [layer for layer, used in self.layers_usage.items() if not used]

    def add_model(self, model: LayerModel):
        if model not in self.models:
            self.models.append(model)

    def add_layer(self, layer: Layer):
        if layer not in self.layers:
            self.layers.append(layer)
        if layer.model not in self.models:
            self.add_model(layer.model)

    def remove_layer(self, layer: Layer):
        # remove all links with this layers from its inputs
        for variable, (input_layer, input_var) in layer.get_inputs().items():
            link = Layer.IOLink(input_layer, input_var, layer, variable)
            input_layer.remove_io_link(link)

        # remove all links with this layer from its outputs
        for variable, outputs in layer.get_outputs():
            for output_layer, output_var in outputs:
                link = Layer.IOLink(layer, variable, output_layer, output_var)
                output_layer.remove_io_link(link)

        layer.get_model().detach_layer(layer)

        self.layers.remove(layer)

    def remove_ghost_layers(self):
        layers_usage = self.layers_usage
        unused_layers_indices = [
            layer for layer in self.layers if not layers_usage[layer]
        ]
        # used_layers = [layer for layer in self.layers if layers_usage[layer]]

        for layer in self.unused_layers:
            self.remove_layer(layer)

    def remove_ghost_models(self):
        used_models = [
            i for i, model in enumerate(self.models) if model.get_attached_layers()
        ]
        self.models = used_models

    def get_unique_variables(self):
        pass


class ModelSkeleton:
    def __init__(self, submodels: dict[str, ModelTemplate], runs, outputs, inputs=None):
        """
        Initializes a ModelSkeleton object.

        :param inputs: A dictionary where keys are input variables and values are their properties
        (properties are not directly used in the class).

        :param outputs: A dictionary where keys are output variables and values are their properties. the properties are
        used are the following:
            - 'id' : the index of the run whose output will we used for the output variable
            -'variable': the output variable of the run to use as an output

        :param submodels: A dictionary of submodels where keys are submodel names and values are their properties
        (properties are not directly used in the class).

        :param runs: A list of dictionaries indicating the runs to perform. Each dictionary should be in format
        {'id' : model_id, 'inputs': dict_of_inputs} where dict_of_inputs's keys correspond to input variables of the model
        used in the run and the values correspond to a pair [id, name] (in list format). 'id' is the run whose output to use and
        'name' is the relevant output variable (use -1 for input variable ids).
        """
        self.submodels = submodels
        self.runs = runs
        self.outputs = outputs
        self.inputs = inputs
        if self.inputs is None:
            self.inputs = self.find_inputs()

    def find_inputs(self):
        """
        returns the input variables of the model
        :return: a list of [model_id,variable_name]
        """
        inputs = []
        for run_id, run in enumerate(self.runs):
            for input in run["inputs"]:
                if run["inputs"][input][0] == -1:
                    inputs.append(run["inputs"][input][1])
        return inputs

    def find_models_variables_connected_to_input(self, input_id):
        variables = []
        for run_id, run in enumerate(self.runs):
            for input in run["inputs"]:
                if run["inputs"][input][0] == -1:
                    variables.append([run_id, input])
        return variables

    def check_graph(self):
        for run in self.runs:
            # check submodel exist
            if not run["id"] in self.submodels:
                return False
            for i in list(run["inputs"]):
                if (not run["inputs"][i][0] <= len(self.runs)) or not run["inputs"][i][
                    0
                ] >= -1:
                    s = (
                        "when checking graph, the following run was referred in the edges inputs but not existing : ",
                        +run["inputs"][i][0],
                    )
                    warnings.warn(s, category=Warning)
                    return False
        return True

    def is_parent(self, parent_id, child_id):
        if child_id == -1:
            return parent_id == -1

        direct_parents = self.get_direct_parents(child_id)
        # check direct connection:
        if parent_id in direct_parents:
            return True

        # no direct connection, search in parents of child
        for i in direct_parents:
            if self.is_parent(parent_id, i):
                return True

        return False

    # returns the runs whose outputs are input of the given run
    def get_direct_parents(self, run_id):
        if run_id == -1:
            return []
        parents = []
        for input_var in self.runs[run_id]["inputs"]:
            parents.append(self.runs[run_id]["inputs"][input_var][0])
        return list(set(parents))

    def get_parents(self, run_id):
        if run_id == -1:
            return []
        parents = []
        direct_parents = self.get_direct_parents(run_id)
        parents.extend(direct_parents)
        for parent in direct_parents:
            parents.extend(self.get_parents(parent))
        return parents

    def get_direct_children(self, run_id):
        child_runs = []
        for index, run in enumerate(self.runs):
            for input_var in run["inputs"]:
                if run["inputs"][input_var][0] == run_id:
                    child_runs.append(index)
        return child_runs

    def get_children(self, run):
        children = []
        direct_children = self.get_direct_children(run)
        children.extend(direct_children)
        for child in direct_children:
            children.extend(self.get_children(child))
        return list(set(children))

    def is_connected_to_output(self, run):
        outputting_runs = []
        for key in self.outputs:
            outputting_runs.append(self.outputs[key][0])
        outputting_runs = set(outputting_runs)
        for run_ in outputting_runs:
            if run_ == run or self.is_parent(run, run_):
                return True
        return False

    def is_connected_to_input(self, run):
        return self.is_parent(-1, run)

    def get_runs_heights(self):
        runs_size = len(self.runs)
        heights = [-1 for i in range(runs_size)]

        def set_height(run_id, height):
            if run_id == -1:
                return
            if height > runs_size:
                raise Exception("Cycle detected in the graph")
            if height > heights[run_id]:
                heights[run_id] = height

                for input_var in self.runs[run_id]["inputs"]:
                    set_height(self.runs[run_id]["inputs"][input_var][0], height + 1)

        for output in self.outputs:
            set_height(self.outputs[output][0], 0)
        return heights

    def get_unused_runs(self):
        unused_runs = []
        heights = self.get_runs_heights()
        for i, h in enumerate(heights):
            if h == -1:
                unused_runs.append(i)

        return unused_runs

    def find_runs_order(self):
        heights = self.get_runs_heights()

        runs_size = len(self.runs)
        max_height = max(heights)
        ordered_items = []
        for i in range(max_height + 1):
            for j in range(runs_size):
                if heights[j] == max_height - i:
                    ordered_items.append(j)

        return ordered_items

    def reorder_runs(self, new_order=None):
        if new_order is None:
            new_order = self.find_runs_order()
        inverter = [-1 for i in range(len(self.runs))]
        for index, value in enumerate(new_order):
            inverter[value] = index
        # print('obtained new runs order: ', new_order)
        # print('inverter: ', inverter)
        self.runs = [self.runs[i] for i in new_order]
        for run in self.runs:
            for input_var in run["inputs"]:
                if run["inputs"][input_var][0] != -1:
                    run["inputs"][input_var][0] = inverter[run["inputs"][input_var][0]]
        for output in self.outputs:
            self.outputs[output][0] = inverter[self.outputs[output][0]]

    def get_runs_of_model(self, model_id):
        node_runs = []
        for i, run in enumerate(self.runs):
            if run["id"] == model_id:
                node_runs.append(i)
        return node_runs

    def get_first_run_of_model(self, model_id):
        for i, run in enumerate(self.runs):
            if run["id"] is model_id:
                return i
        return None

    def del_run(self, index):
        for run in self.runs:
            for input_var in run["inputs"]:
                if run["inputs"][input_var][0] > index:
                    run["inputs"][input_var][0] -= 1
        for output in self.outputs:
            if self.outputs[output][0] > index:
                self.outputs[output][0] -= 1
        run_model = self.runs[index]["id"]
        del self.runs[index]
        self.remove_unused_submodels()

    def add_model(self, model_parameters):
        model_name = model_parameters["name"]
        model_parameters = copy.copy(model_parameters)
        del model_parameters["name"]
        self.submodels[model_name] = model_parameters
        return True

    def add_run(self, run_to_insert):
        self.runs.append(run_to_insert)
        return len(self.runs) - 1

    def remove_unused_submodels(self):
        models_to_remove = []
        for model in self.submodels:
            if len(self.get_runs_of_model(model)) == 0:
                models_to_remove.append(model)
        for model in models_to_remove:
            del self.submodels[model]

    def build_templates(self):
        pass
