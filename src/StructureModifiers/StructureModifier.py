import itertools
import copy
from typing import Tuple

from ..Base.modelskeleton import ModelSkeleton
from ..Base.modelproperties import ModelProperties, ModelPropertiesMapper
from abc import ABC, abstractmethod


class StructureModifier(ABC):
    def __init__(self, allowed_models: list,  max_model_reuse: int = 1):
        """
        an object allowing easy modification of a model
        :param allowed_models: a list of allowed submodel in the models structure

        """
        self.allowed_models = allowed_models
        self.max_model_reuse = max_model_reuse

        request_list = [(model["type"], model) for model in allowed_models]
        self.allowed_models_props = ModelProperties.generate_model_properties_from_model_list(request_list)

    def add_submodel(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper],
                     submodel_to_add,
                     input_vars,
                     output_vars,
                     inplace=False) -> ModelSkeleton:
        """
        Add a submodel to the model connected to the inputs and outputs given.

        :param model: The model to start from : a tupe [ModelSkeleton, ModelProperties] where ModelProperties can be
        None
        :param submodel_to_add: The data of the submodel to add. Requires the fields "name", "source","type"
        :param input_vars: The inputs the run is going to receive
        :param output_vars: the runs and variables that are going to receive the output of the model
        :param inplace: Add the submodel inplace or create a copy of the model and perform the operation on the copy
        :return: The model skeleton on which the operation has been performed
        """
        model_skeleton, model_properties = model

        model_name = submodel_to_add["name"]
        model_type = submodel_to_add["type"]
        if model_type not in self.allowed_models_props:
            raise ValueError(f"Not allowed to add a model of type {submodel_to_add['type']}")

        # verifies if the inputs and outputs are valid
        input_runs = set([var[0] for var in input_vars.values()])
        output_runs = set([var[0] for var in output_vars if var[0] != 'out'])
        for i, o in itertools.product(input_runs, output_runs):
            if i == o or model_skeleton.is_parent(o, i):
                raise ValueError(f"inputs of run must be calculated before the outputs: got input {i} and output {o}")
        models_uses = self.check_models_reuse(model)
        if model_name in models_uses.keys():
            models_uses[submodel_to_add["name"]] += 1
        else:
            models_uses[submodel_to_add["name"]] = 1
        max_models_use = max(models_uses.values())
        if max_models_use > self.max_model_reuse:
            keys = [k for k, v in models_uses.items() if v > max_models_use]
            raise RuntimeError(f"Too many uses for the model(s) : {keys}")

        new_model_skeleton = model_skeleton
        if not inplace:
            new_model_skeleton = copy.deepcopy(new_model_skeleton)

        # add model to model list
        if model_name not in new_model_skeleton.submodels:
            new_model_skeleton.add_model(submodel_to_add)
        # check

        new_run = {'id': submodel_to_add["name"], "inputs": input_vars}
        run_id = new_model_skeleton.add_run(new_run)
        for output_run, output_run_var, output_var, connection_info in output_vars:
            run_to_use = output_run
            var_to_use = output_run_var
            if connection_info == "add":
                # add an Add unit
                add_model = {"source": "basis_templates", "type": "Add"}
                add_model["name"] = self.new_model_name(model, add_model)
                if output_run == 'out':
                    output_run_object = new_model_skeleton.outputs
                else:
                    output_run_object = new_model_skeleton.runs[output_run]['inputs']

                input_add = output_run_object[output_run_var]
                add_run = {'id': add_model["name"], "inputs": {"X1": input_add}}
                new_model_skeleton.add_model(add_model)
                run_to_use = new_model_skeleton.add_run(add_run)
                var_to_use = "X2"
                if output_run == 'out':
                    new_model_skeleton.outputs[output_run_var] = [run_to_use, 'Y']
                else:
                    new_model_skeleton.runs[output_run]['inputs'][output_run_var] = [run_to_use, 'Y']

            var_data = [run_id, output_var]
            if run_to_use == 'out':
                new_model_skeleton.outputs[var_to_use] = var_data
            else:
                new_model_skeleton.runs[run_to_use]['inputs'][var_to_use] = var_data
        new_model_skeleton.reorder_runs()
        new_model_skeleton.remove_unused_submodels()
        return new_model_skeleton

    @abstractmethod
    def propose_random_add(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None]):
        pass

    @abstractmethod
    def propose_random_remove(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None]):
        pass

    def remove_run(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None],
                   run_to_remove,
                   replacement_pairings,
                   inplace=False) -> ModelSkeleton:
        model_skeleton, model_properties = model
        if not inplace:
            model_skeleton = copy.deepcopy(model_skeleton)
        for input, output in replacement_pairings:
            output_run, output_var = output
            if output_run == 'out':
                model_skeleton.outputs[output_var] = input
            else:
                model_skeleton.runs[output_run]['inputs'][output_var] = input
        if len(model_skeleton.get_direct_children(run_to_remove)) > 0:
            raise AssertionError(f"not all output links of run {run_to_remove} have been removed!")
        model_skeleton.del_run(run_to_remove)
        model_skeleton.reorder_runs()
        model_skeleton.remove_unused_submodels()
        return model_skeleton

    def check_models_reuse(self, model):
        model_skeleton, model_properties = model
        return {model: len(model_skeleton.get_runs_of_model(model)) for model in model_skeleton.submodels}

    def new_model_name(self, model, model_parameters):
        model_skeleton, _ = model
        keys = list(model_skeleton.submodels)
        node_type = model_parameters['type']
        nums = [s.split(node_type + '_')[1] for s in keys if node_type + '_' in s]
        nums = sorted([int(s) for s in nums if s.isdigit()])
        n_used = -1
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] != 1:
                n_used = nums[i] + 1
        if n_used == -1:
            if len(nums) == 0:
                n_used = 0
            else:
                n_used = nums[-1] + 1
        model_name = node_type + '_' + str(n_used)
        return model_name



