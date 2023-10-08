import copy
import warnings
import numpy as np

'''
A class that contains the bare-bone structure describing a model: the input and output of the model, 
the different layers used, the order in which variables are passed in layers and
the parameters of the model and their transcription to parameters of its layers

#Parameters of the class

'''


class ModelSkeleton:
    def __init__(self, submodels, runs, outputs, inputs = None ):
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
            for input in run['inputs']:
                if run['inputs'][input][0] == -1:
                    inputs.append(run['inputs'][input][1])
        return inputs

    def find_models_variables_connected_to_input(self,input_id):
        variables = []
        for run_id, run in enumerate(self.runs):
            for input in run['inputs']:
                if run['inputs'][input][0] == -1:
                    variables.append([run_id, input])
        return variables

    def check_graph(self):
        for run in self.runs:
            # check submodel exist
            if not run['id'] in self.submodels:
                return False
            for i in list(run['inputs']):
                if (not run['inputs'][i][0] <= len(self.runs)) or not run['inputs'][i][0] >= -1:
                    s = 'when checking graph, the following run was referred in the edges inputs but not existing : ', + \
                        run['inputs'][i][0]
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
        for input_var in self.runs[run_id]['inputs']:
            parents.append(self.runs[run_id]['inputs'][input_var][0])
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
            for input_var in run['inputs']:
                if run['inputs'][input_var][0] == run_id:
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
                raise Exception('Cycle detected in the graph')
            if height > heights[run_id]:
                heights[run_id] = height

                for input_var in self.runs[run_id]['inputs']:
                    set_height(self.runs[run_id]['inputs'][input_var][0], height + 1)

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
                if heights[j] == max_height-i:
                    ordered_items.append(j)

        return ordered_items

    def reorder_runs(self, new_order=None):
        if new_order is None:
            new_order = self.find_runs_order()
        inverter = [-1 for i in range(len(self.runs))]
        for i in new_order:
            inverter[i] = new_order.index(i)
        # print('obtained new runs order: ', new_order)
        # print('inverter: ', inverter)
        self.runs = [self.runs[i] for i in new_order]
        for run in self.runs:
            for input_var in run['inputs']:
                if run['inputs'][input_var][0] != -1:
                    run['inputs'][input_var][0] = inverter[run['inputs'][input_var][0]]
        for output in self.outputs:
            self.outputs[output][0] = inverter[self.outputs[output][0]]

    def get_runs_of_model(self, model_id):
        node_runs = []
        for i, run in enumerate(self.runs):
            if run['id'] == model_id:
                node_runs.append(i)
        return node_runs

    def get_first_run_of_model(self, model_id):
        for i, run in enumerate(self.runs):
            if run['id'] is model_id:
                return i
        return None

    def del_run(self, index):
        for run in self.runs:
            for input_var in run['inputs']:
                if run['inputs'][input_var][0] > index:
                    run['inputs'][input_var][0] -= 1
        for output in self.outputs:
            if self.outputs[output][0] > index:
                self.outputs[output][0] -= 1
        run_model = self.runs[index]['id']
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