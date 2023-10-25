from .StructureModifier import StructureModifier
from ..Base.modelskeleton import ModelSkeleton
from ..Base.modelproperties import ModelPropertiesMapper
import numpy as np

class SimpleProbabilisticModifier(StructureModifier):
    def __init__(self, allowed_models: list,
                 model_skeleton: ModelSkeleton,
                 model_properties: ModelPropertiesMapper = None,
                 fixed_runs: list = None,
                 fixed_models: list = None):
        super(SimpleProbabilisticModifier, self).__init__(allowed_models=allowed_models,
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
        for run in self.fixed_runs:
            if run > run_to_remove:
                run -= 1
        model_skeleton = super().remove_run((self.model_skeleton, self.model_props),
                                            run_to_remove,
                                            replacement_pairings,
                                            inplace)

        return model_skeleton

    def add_submodel(self, submodel_to_add, input_vars, output_vars, inplace=False, **kwargs) -> ModelSkeleton:
        model_skeleton = super().add_submodel((self.model_skeleton, self.model_props),
                                              submodel_to_add,
                                              input_vars,
                                              output_vars,
                                              inplace)
        return model_skeleton
