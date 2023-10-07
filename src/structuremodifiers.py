import itertools
import copy
from typing import Tuple

from modelskeleton import ModelSkeleton
from modelproperties import ModelProperties, ModelPropertiesMapper
import numpy as np
from abc import ABC, abstractmethod


class StructureModifier(ABC):
    def __init__(self, allowed_models: list, model_skeleton: ModelSkeleton,
                 model_properties: ModelPropertiesMapper = None, max_model_reuse: int = 1):
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
            if model_skeleton.is_parent(o, i):
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

        return new_model_skeleton

    @abstractmethod
    def propose_random_add(self):
        pass

    @abstractmethod
    def propose_random_remove(self):
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


class SimpleProbabilisticModifier(StructureModifier):
    def __init__(self, allowed_models: list,
                 model_skeleton: ModelSkeleton,
                 model_properties: ModelPropertiesMapper = None,
                 fixed_runs: list = None,
                 fixed_models: list = None):
        super(SimpleProbabilisticModifier, self).__init__(allowed_models=allowed_models,
                                                          model_skeleton=model_skeleton,
                                                          model_properties=model_properties,
                                                          max_model_reuse=1
                                                          )
        if fixed_models is None:
            fixed_models = []
        if fixed_runs is None:
            fixed_runs = []
        self.fixed_runs = fixed_runs
        self.fixed_models = fixed_models
        self.model_skeleton = model_skeleton
        self.model_props = model_properties
        if self.model_props is None:
            self.model_props = ModelPropertiesMapper(self.model_skeleton)

    def get_model_probabilities_add(self):
        outcomes = [model for model in self.allowed_models]
        p = 1. / len(self.allowed_models)
        return outcomes, [p for outcome in outcomes]

    def get_input_probabilities_add(self, model, proposed_output, variable_to_choose):
        outcomes = [[-1, var] for var in self.model_skeleton.inputs.keys()]
        for i, run in enumerate(self.model_skeleton.runs):
            model = run['id']
            model_props = self.model_props.sub_props[model]
            output_vars = [[i, k] for k, v in model_props['variables'].items() if v['io'] == 'out']
            outcomes.extend(output_vars)
        p = 1. / len(outcomes)
        return [outcomes, [p for o in outcomes]]

    def get_output_probabilities_add(self, model):
        model_props = self.allowed_models_props[model["type"]]
        output_vars = [k for k, v in model_props['variables'].items() if v['io'] == 'out']
        outcomes = [['out', var, out] for var in self.model_skeleton.outputs.keys() for out in output_vars]
        for i, run in enumerate(self.model_skeleton.runs):
            input_vars = [[i, k, out] for k in run['inputs'].keys() for out in output_vars]
            outcomes.extend(input_vars)
        p = 1. / len(outcomes)
        return outcomes, [p for o in outcomes]

    def get_model_probabilities_remove(self):
        outcomes = [model for model in self.model_skeleton.submodels.keys() if model not in self.fixed_models]
        p = 1. / len(outcomes)
        return outcomes, [p for o in outcomes]

    def get_pairing_probabilities_remove(self, run, output_to_pair):
        outcomes = [input_ for input_ in self.model_skeleton.runs[run]['inputs'].values()]
        p = 1. / len(outcomes)
        return outcomes, [p for o in outcomes]

    def check_input_output_compatibility(self, input_, output):
        if output[0] == "out":
            return True

        if input_[0] == output[0]:
            return False

        ok = not self.model_skeleton.is_parent(output[0], input_[0])

        return ok

    def check_fixed_add(self, proposition):
        model_proposition, proposed_inputs, proposed_output = proposition
        for run in self.fixed_runs:
            for output in proposed_output:
                output_run, output_run_var, output_var = output
                if output_run == "out":
                    current_input_run, current_input_var = self.model_skeleton.outputs[output_run_var]
                else:
                    current_input_run, current_input_var = self.model_skeleton.runs[output_run]['inputs'][
                        output_run_var]
                if self.model_skeleton.is_parent(run, current_input_run):
                    # change output link to add : perform an add before
                    return "add"
                else:
                    return None
        return None

    def propose_random_add(self):
        model_list, model_probabilities = self.get_model_probabilities_add()

        model_proposition = np.random.choice(model_list,
                                             p=model_probabilities)
        variables = self.allowed_models_props[model_proposition["type"]]['variables']
        input_vars = {k: v for k, v in variables.items() if v['io'] == 'in'}

        output_vars, output_probs = self.get_output_probabilities_add(model_proposition)
        i_choice = np.random.choice(a=len(output_vars), p=output_probs)
        proposed_output = [output_vars[i_choice]]

        proposed_inputs = {}
        for var_name, var_data in input_vars.items():
            found_input = False
            p_tot = 1.
            inputs, input_probs = self.get_input_probabilities_add(model_proposition, proposed_output, var_name)
            while (not found_input) and len(inputs) > 0:
                i_choice = np.random.choice(a=len(inputs), p=np.array(input_probs) / p_tot)
                proposed_input = inputs[i_choice]
                found_input = True
                for output in proposed_output:
                    found_input = found_input and self.check_input_output_compatibility(proposed_input, output)
                if found_input:
                    proposed_inputs[var_name] = proposed_input
                else:
                    inputs.pop(i_choice)
                    p_tot -= input_probs[i_choice]
                    input_probs.pop(i_choice)

        # found model and position in the graph, now give a complete proposition
        model_proposition["name"] = self.new_model_name([self.model_skeleton, self.model_props], model_proposition)

        # check we don't lose fixed runs
        proposition = [model_proposition, proposed_inputs, proposed_output]
        output_info = self.check_fixed_add(proposition)
        proposed_output[0].append(output_info)

        return model_proposition, proposed_inputs, proposed_output

    def propose_random_remove(self):
        models_list, models_probabilities = self.get_model_probabilities_remove()
        found_model = False
        model_proposition = None
        allowed_runs = []
        p_tot = 1.
        while (not found_model) and len(models_list) > 0:
            model_proposition = np.random.choice(models_list,
                                                 p=np.array(models_probabilities) / p_tot)
            model_runs = self.model_skeleton.get_runs_of_model(model_proposition)
            allowed_runs = [run for run in model_runs if not (run in self.fixed_runs)]
            found_model = len(allowed_runs) > 0
            if not found_model:
                model_index = min([i for i, model in enumerate(models_list) if model == model_proposition])
                models_list.pop(model_index)
                p_tot -= models_probabilities[model_index]
                models_probabilities.pop(model_index)

        p = 1. / len(allowed_runs)
        runs_probabilities = [p for run in allowed_runs]
        proposed_run = np.random.choice(allowed_runs, p=runs_probabilities)
        # change outputs to inputs:
        outputs_of_run = self.model_skeleton.get_direct_children(proposed_run)
        outputs_of_run = [[run, var] for run in outputs_of_run for var in
                          self.model_skeleton.runs[run]['inputs'].keys()]
        output_vars_connected_to_run = [['out', k] for k, v in self.model_skeleton.outputs.items() if
                                        v[0] == proposed_run]
        outputs_of_run.extend(output_vars_connected_to_run)
        proposed_pairings = []
        for output in outputs_of_run:
            found_input = False
            inputs_list, inputs_probabilities = self.get_pairing_probabilities_remove(proposed_run, output)
            p_tot = 1.
            while (not found_input) and len(inputs_list) > 0:

                i_chosen = np.random.choice(len(inputs_list), p=np.array(inputs_probabilities) / p_tot)
                proposed_input = inputs_list[i_chosen]
                found_input = self.check_input_output_compatibility(proposed_input, output)
                if found_input:
                    proposed_pairings.append((proposed_input, output))
                if not found_input:
                    inputs_list.pop(i_chosen)
                    p_tot -= inputs_probabilities[i_chosen]
                    inputs_probabilities.pop(i_chosen)

        return proposed_run, proposed_pairings

    # override the remove_run from super to give compatibility with fixed_runs:
    def remove_run(self, run_to_remove, replacement_pairings, inplace=False, **kwargs) -> ModelSkeleton:
        model_skeleton = super().remove_run((self.model_skeleton, self.model_props),
                                            run_to_remove,
                                            replacement_pairings,
                                            inplace)
        for run in self.fixed_runs:
            if run > run_to_remove:
                run -= 1
        return model_skeleton

    def add_submodel(self, submodel_to_add, input_vars, output_vars, inplace=False, **kwargs) -> ModelSkeleton:
        model_skeleton = super().add_submodel((self.model_skeleton, self.model_props),
                                              submodel_to_add,
                                              input_vars,
                                              output_vars,
                                              inplace)
        return model_skeleton
