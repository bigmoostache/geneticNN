from .StructureModifier import StructureModifier
from ..Base.modelskeleton import ModelSkeleton
from ..Base.modelproperties import ModelPropertiesMapper
import numpy as np

from typing import Tuple


class SmallUpdateModifier(StructureModifier):

    def __init__(self, allowed_models: list, max_model_reuse: int = 1):
        super(SmallUpdateModifier, self).__init__(allowed_models, max_model_reuse)

    def _get_links_probabilities(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None]):
        model_skeleton, model_properties = model
        links = [(inputs, [index, variable])
                 for index, run in enumerate(model_skeleton.runs) for variable, inputs in run['inputs'].items()]
        outputs_links = [(inputs, ['out', var]) for var, inputs in model_skeleton.outputs.items()]
        links.extend(outputs_links)
        p = 1. / len(links)
        probabilities = [p for l in links]
        return links, probabilities

    def _get_models_probabilities(self, model, link):
        probabilities = [1 / len(self.allowed_models) for _ in self.allowed_models]
        return self.allowed_models.copy(), probabilities

    def _get_available_inputs(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper], link, test_submodel):
        model_skeleton, model_properties = model
        unavailable_runs = [] if link[1][0] is 'out' else model_skeleton.get_children(link[1][0])
        available_runs = [i for i, _ in enumerate(model_skeleton.runs) if i not in unavailable_runs and i != link[1][0]]
        available_inputs = []

        for run_id in available_runs:
            run = model_skeleton.runs[run_id]
            model_props = model_properties.sub_props[run['id']]
            outputs = [[run_id, k] for k, v in model_props['variables'].items() if v['io'] == 'out']
            available_inputs.extend(outputs)

        global_inputs = [[-1, k] for k in model_skeleton.inputs]
        available_inputs.extend(global_inputs)
        return available_inputs

    def _random_variable_input(self, inputs_candidates):
        var_input_id = np.random.choice(len(inputs_candidates))
        return inputs_candidates[var_input_id]

    def _generate_candidate_links(self, inputs_candidates, in_vars, link, target_variable):
        """
        generates the candidates links for the variables in in_vars and where the target variable is necessarily
        connected to the input of the link
        """
        links_candidates = {}
        for variable in in_vars:
            links_candidates[variable] = link[0] if variable == target_variable else self._random_variable_input(
                inputs_candidates)

        return links_candidates

    def _find_variables_links(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper],
                              link, test_submodel, max_attempts = 50):
        """
        find all the links necessary for the variables of the new submodel to add
        """
        variables = self.allowed_models_props[test_submodel['type']]['variables']

        in_vars = [k for k, v in variables.items() if v['io'] == 'in']
        out_vars = [k for k, v in variables.items() if v['io'] == 'out']

        inputs_candidates = self._get_available_inputs(model, link, test_submodel)

        for _ in range(max_attempts):
            input_variable_id = np.random.choice(len(in_vars))
            input_variable = in_vars[input_variable_id]

            output_variable_id = np.random.choice(len(out_vars))
            output_variable = out_vars[output_variable_id]

            output_candidate = [(*link[1], output_variable, None)]

            link_candidates = self._generate_candidate_links(inputs_candidates, in_vars, link, input_variable)

            if link_candidates:
                return link_candidates, output_candidate

        return None

    def propose_random_add(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None]):
        # find a random link to change:
        links, links_probabilities = self._get_links_probabilities(model)

        for _ in range(len(links)):
            test_link_id = np.random.choice(len(links))
            test_link = links[test_link_id]

            # find model to add in the middle
            models_options, models_probabilities = self._get_models_probabilities(model, test_link)
            for _ in range(len(models_options)):
                test_submodel_id = np.random.choice(len(models_options))
                test_submodel = models_options[test_submodel_id]
                outputs = self._find_variables_links(model, test_link, test_submodel)
                if outputs:
                    # good variables link found
                    test_submodel['name'] = self.new_model_name(model, test_submodel)
                    return test_submodel, *outputs
                else:
                    # No viable link found, remove the proposed model and try again with one the models left
                    models_options.pop(test_submodel_id)

            else:
                # No good  model found for this link, try to change another link
                links.pop(test_link_id)

        return None

    def check_input_output_compatibility(self, model_skeleton, input_, output):
        if output[0] == "out":
            return True

        if input_[0] == output[0]:
            return False

        ok = not model_skeleton.is_parent(output[0], input_[0])

        return ok

    def propose_random_remove(self, model: Tuple[ModelSkeleton, ModelPropertiesMapper | None],
                              fixed_runs=None):
        if fixed_runs is None:
            fixed_runs = []
        model_skeleton, model_properties = model

        model_proposition = None
        allowed_runs = [i for i, _ in enumerate(model_skeleton.runs) if i not in fixed_runs]
        runs_probabilities = [1/len(allowed_runs) for run in allowed_runs]
        proposed_run = np.random.choice(allowed_runs, p=runs_probabilities)
        # change outputs to inputs:
        outputs_of_run = model_skeleton.get_direct_children(proposed_run)
        outputs_of_run = [[run, var] for run in outputs_of_run for var in
                          model_skeleton.runs[run]['inputs'].keys()]
        output_vars_connected_to_run = [['out', k] for k, v in model_skeleton.outputs.items() if
                                        v[0] == proposed_run]
        outputs_of_run.extend(output_vars_connected_to_run)
        proposed_pairings = []
        for output in outputs_of_run:
            found_input = False
            inputs_list = [input_ for input_ in model_skeleton.runs[proposed_run]['inputs'].values()]
            p_tot = 1.
            for _ in range(len(inputs_list)):

                i_chosen = np.random.choice(len(inputs_list))
                proposed_input = inputs_list[i_chosen]

                found_input = self.check_input_output_compatibility(model_skeleton, proposed_input, output)
                if found_input:
                    proposed_pairings.append((proposed_input.copy(), output.copy()))
                    break
                else:
                    inputs_list.pop(i_chosen)

        return proposed_run, proposed_pairings
