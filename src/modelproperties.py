import json
import re, os
from .modelskeleton import ModelSkeleton
from typing import Iterable, Dict, Any
from sympy import symbols, Eq, solve, sympify
import copy

from torch.nn import Linear


# utility class to manage families of parameters generated by constraints
class Subsets:
    def __init__(self, elements: Iterable):
        self.data = {el: {el} for el in elements}
        self.reduced_list = {el for el in elements}
        self.inverse_reduced_list = {id(self.data[el]): el for el in self.reduced_list}

    def __getitem__(self, item):
        if item in self.data:
            return self.inverse_reduced_list[id(self.data[item])]
        else:
            raise KeyError(f'No key {item} found')

    def merge(self, el1, el2):
        if el1 in self.data and el2 in self.data:
            ref_1, ref_2 = self[el1], self[el2]
            del self.inverse_reduced_list[id(self.data[el1])]
            if ref_1 != ref_2:
                del self.inverse_reduced_list[id(self.data[el2])]
            union = self.data[el1].union(self.data[el2])
            for el in union:
                self.data[el] = union
            self.reduced_list.discard(ref_2)
            self.inverse_reduced_list[id(union)] = ref_1
        else:
            raise ValueError(f'One or both elements are not found: {el1} , {el2}')

    def check_add(self, el: object):
        if el not in self.data:
            self.data[el] = {el}
            self.reduced_list.add(el)

    def check_extend(self, elements: Iterable):
        for el in elements:
            self.check_add(el)


class ModelProperties:
    def __init__(self):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.props = {}

    @staticmethod
    def generate_properties_from_file(model_file):
        props = ModelProperties()
        props.props = ModelProperties.get_props(model_file)

    # returns the props of a model read in the source file
    @staticmethod
    def get_props(model_file):
        text = open(model_file).read()
        text2 = text.split("BEGIN_PROPS")[1]
        text2 = text2.split("END_PROPS")[0]
        props = json.loads(text2)
        return props


def get_variable_parameter_name(model_id, variable_id, dim):
    """free dimension not implemented yet
    if dim is -1:
        dim = "default"
    """
    return f"{model_id}__{variable_id}_{dim}"


def get_global_parameter(model_id: str, parameter_id: str):
    return f"{model_id}_{parameter_id}"


class ModelPropertiesMapper:
    def __init__(self, model_skeleton, detailed_props=True, n_constraints_trial=1):

        self.param_subset: Subsets = None
        self.model_skeleton = model_skeleton
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.props: dict = None
        self.constraints: dict = {"equality": [], "symbolic": [], "parameters":{}}
        self.symbolic_constraints_sets: list = []
        self.sub_props: dict = {}
        self.sympy_data: dict = {}

        self.generate_properties()
        self.generate_constraints()
        self.generate_global_parameters()
        self.solve_symbolic_constraints()

    # Solves the constraints on the model and generate the local and global dictionaries of parameters
    def generate_properties(self):
        # loads the sub-models properties
        for submodel in self.model_skeleton.submodels:
            data = self.model_skeleton.submodels[submodel]
            model_file = ""
            if os.path.isabs(data["source"]):
                model_file = data["source"]
            else:
                model_file = self.script_directory + '/' + data["source"]
            model_file = model_file + "/" + data["type"] + ".py"
            self.sub_props[submodel] = ModelProperties.get_props(model_file)
            subprops = self.sub_props[submodel]
            for var in subprops["variables"].values():
                var["dim"] = int(var["dim"])

        # add the parameters corresponding to the variables
        self.check_and_generate_variables_parameters()

    def generate_constraints(self):
        runs = self.model_skeleton.runs
        for submodel_id, submodel in self.sub_props.items():
            for index, run in enumerate(runs):
                for var_name, data in run['inputs'].items():
                    input_run, input_var = data
                    if input_run > -1:
                        self.convert_run_linking_to_equality_constraint(((input_run,input_var),(index,var_name)))
            for constraint in submodel['constraints']:
                if constraint[0] == 'equality':
                    global_constraint = [get_global_parameter(submodel_id, param) for param in constraint[1]]
                    self.add_an_equality_constraint(global_constraint)
                if constraint[0] == 'symbolic':
                    global_constraint = copy.copy(constraint[1])
                    global_constraint[0] = [get_global_parameter(submodel_id, param) for param in global_constraint[0]]
                    self.add_a_symbolic_constraint(global_constraint)
                if constraint[0] == 'parameter':
                    global_constraint = (get_global_parameter(submodel_id, constraint[1][0]), constraint[1][1])
                    self.add_a_parameter_constraint(global_constraint)

    def generate_global_parameters(self):

        # generates list of global names for all parameters
        all_params = [get_global_parameter(model_id, parameter_id) for model_id, model in self.sub_props.items() for
                      parameter_id in model['parameters'].keys()]
        self.param_subset = Subsets(all_params)
        for constraint in self.constraints['equality']:
            param_0 = constraint[0]
            for param in constraint[1:]:
                self.param_subset.merge(param_0, param)

        # rewrite symbolic constraints in terms of global variables only
        for constraint in self.constraints['symbolic']:
            for parameter in constraint[0]:
                parameter = self.param_subset[parameter]
        for param in self.constraints['parameters']:
            reference = self.param_subset[param]
            if reference is not param:
                self.constraints['parameters'][reference].extend(self.constraints['parameters'][param])
                del self.constraints['parameters'][param]

    def solve_symbolic_constraints(self):
        if len(self.constraints['symbolic']) > 0:
            global_parameters = self.param_subset.reduced_list
            sympy_variables = {param: symbols(param) for param in global_parameters}
            ineq = []
            eq = []

            # generate the formulae
            for constraint in self.constraints['symbolic']:
                params, formula = constraint
                for param_id, param in enumerate(params):
                    to_replace = f"${param_id} "
                    replacing = f"{param} "
                    new_formula = formula.replace(to_replace, replacing)
                    if ' = ' in new_formula:
                        e1, e2 = new_formula.split("=")
                        eq.append(Eq(sympify(e1), sympify(e2)))
                    if '<' in new_formula or '>' in new_formula:
                        ineq.append(sympify(new_formula))

            # find the constrained variables and their formulation
            constrained_params = set()
            for model_id, model in self.sub_props.items():
                for param, param_data in model['parameters'].items():
                    if 'constrained' in param_data and param_data['constrained'] is True:
                        constrained_params.add(self.param_subset[get_global_parameter(model_id, param)])
            constrained_params_temp = {sympy_variables[param] for param in constrained_params}

            # replace the constrained variables by their formulation
            constrained_params_formulas: dict = {}
            while len(constrained_params_temp) > 0:
                variable_to_remove = None
                removing_expression = None
                for equation in eq:
                    if equation.lhs in constrained_params_temp:
                        variable_to_remove = equation.lhs
                        removing_expression = equation.rhs
                        break
                    if equation.rhs in constrained_params_temp:
                        variable_to_remove = equation.rhs
                        removing_expression = equation.lhs
                        break
                constrained_params_formulas[variable_to_remove] = removing_expression
                for i, equation in enumerate(eq):
                    eq[i] = equation.subs(variable_to_remove, removing_expression)

                for i, inequality in enumerate(ineq):
                    eq[i] = inequality.subs(variable_to_remove, removing_expression)

                constrained_params_temp.discard(variable_to_remove)

            # solve for the other unknowns
            solved_eq_dict = solve(eq)

            # find free variables
            free_vars = [var_name for var_name, var in sympy_variables.items() if var not in solved_eq_dict]
            self.sympy_data['free'] = free_vars
            # store constrained variables formula in sympy_data
            self.sympy_data['constrained'] = solved_eq_dict
            for param in constrained_params:
                self.sympy_data['constrained'][param] = constrained_params_formulas[sympy_variables[param]].subs(
                    solved_eq_dict)

    def add_a_symbolic_constraint(self, constraint):
        """
        :param constraint: A tuple that represents a linking constraint. It contains three elements:
                            - in_key: A tuple representing the input model property key.
                            - out_key: A tuple representing the output model property key.
                            - constraint_properties: A dictionary representing the properties of the constraint.

        :return: None

        This method adds a constraint linking two parameters in the graph of constraint
        """
        constraints = self.constraints.setdefault('symbolic', [])
        constraint.add(constraint)

    def add_an_equality_constraint(self, parameters_list: list):
        """
        :param parameters_list: A list of parameters that are deemed equals
        :return: None

        This method adds an equality constraint between al parameters in a list in the graph of constraint
        """
        constraints = self.constraints.setdefault('equality', [])
        constraints.append(parameters_list)

    def add_a_parameter_constraint(self, constraint):
        """
        :param constraint: A tuple representing the constraint to be added. The tuple should contain three elements: the model property to which the constraint applies, the parameter to which the constraint applies, and the properties of the constraint.
        :return: None

        The `add_a_parameter_constraint` method is used to add a constraint to a specific parameter of a model property in the ModelPropertiesMapper class. The constraint is represented as a tuple containing three elements:
        1. The model property to which the constraint applies.
        2. The parameter to which the constraint applies.
        3. The properties of the constraint.

        The method first retrieves the sub-properties of the given constraint by accessing the corresponding index in the `sub_props` list attribute of the ModelPropertiesMapper object. Then, it retrieves the `constraints` and `parameters` dictionaries from the `model_properties` dictionary of the sub-properties. After that, it adds the constraint to the `constraints` dictionary by setting the parameter as the key and a set of constraint properties as the value.

        Note that this method does not return anything. It modifies the `model_properties` dictionary directly.
        """
        (parameter, constraint_properties) = self.sub_props[constraint[0]], constraint[1], constraint[2]

        constraints = self.constraints
        constraints = constraints.setdefault('parameters', {})
        current_constraint = constraints.setdefault(parameter, [])
        current_constraint.append(constraint_properties)

    def get_local_parameter(self, parameter_id: str):
        model_id, local_parameter = parameter_id.split('_', 1)
        return model_id, local_parameter

    def check_and_generate_variables_parameters(self):
        """
        Iterates through submodel variables and generates corresponding parameters if they are missing.

        :return: None
        """
        for submodel_id, submodel in self.sub_props.items():
            for variable, var_data in submodel['variables'].items():
                var_dim = var_data['dim']
                '''free dimension not implemented yet
                if var_dim is -1:
                    #add a default parameter
                    param_name = self.get_variable_parameter_name(variable,-1)
                    if param_name not in submodel['params']:
                        # Generate the parameter corresponding to the dimension index of the variable
                        submodel['params'][param_name] = {'type': var_data['type']}
                else:
                '''
                for index in range(var_data['dim']):
                    param_name = f"_{variable}_{index}"

                    if param_name not in submodel['parameters']:
                        # Generate the parameter corresponding to the dimension index of the variable
                        submodel['parameters'][param_name] = {'type': 'int'}

    def convert_run_linking_to_equality_constraint(self, constraint):
        """
        :param constraint: A tuple representing the constraint to be converted. The tuple should have two elements: the input constraint location and the output constraint location.
        :return: None

        This method is used to convert a constraint linking two variables to a constraint on the parameters of the variables. It takes in a constraint tuple, extract the necessary information from it, and performs the conversion. The converted constraint is then added as a linking constraint.

        The constraint tuple should be of the following format:
            - The input constraint location should be a tuple consisting of two elements: the submodel key and the variable key of the input constraint.
            - The output constraint location should be a tuple consisting of two elements: the submodel key and the variable key of the output constraint.

        The method first accesses the properties of the submodels stored in the `sub_props` attribute of the `ModelPropertiesMapper` instance. It retrieves the input and output variables based on the provided constraint locations.

        Next, the method checks the compatibility of the dimensions of the input and output variables. If they are not compatible, an `AssertionError` is raised with a descriptive error message.

        Finally, the method adds a parameter constraint for each dimension of the input variable. The parameter constraint is created using the submodel keys, the modified constraint locations with dimension indexes appended, and a boolean indicating that it is a parameter constraint. The created parameter constraint is then added as a linking constraint using the `add_a_linking_constraint` method.

        Example usage:
            mapper = ModelPropertiesMapper()
            constraint = (("submodel_key_1", "variable_key_1"), ("submodel_key_2", "variable_key_2"))
            mapper.convert_variable_to_parameter_constraint(constraint)
        """
        i_constraint_loc, o_constraint_loc = constraint
        input_run_id, variable_key_inp = i_constraint_loc
        output_run_id, variable_key_out = o_constraint_loc

        input_model_id = self.model_skeleton.runs[input_run_id]['id']
        output_model_id = self.model_skeleton.runs[output_run_id]['id']

        # Accessing properties of submodels
        subs = self.sub_props

        input_var = subs[input_model_id]['variables'][variable_key_inp]
        output_var = subs[output_model_id]['variables'][variable_key_out]

        # Checking compatibility of variables
        if input_var['dim'] is not output_var['dim']:
            raise AssertionError(
                f" Dimension of variables not matching for variable {i_constraint_loc}"
                f" (dim : {input_var['dim']}) and "
                f"variable {o_constraint_loc} (dim : {output_var['dim']})")

        # Adding parameter constraints
        for dim_index in range(input_var['dim']):
            input_param = get_variable_parameter_name(input_model_id, variable_key_inp, dim_index)
            output_param = get_variable_parameter_name(output_model_id, variable_key_out, dim_index)
            param_constraint = [input_param, output_param]
            self.add_an_equality_constraint(param_constraint)

    def resolve_a_parameter_linking_constraint_set(self, base_parameter):
        params_set = self.param_subset[base_parameter]
        constraints = [(param, constraint) for param in params_set for constraint in
                       self.constraints['linking']['up'][param]]
        down_constraints = [(param, constraint) for param in params_set for constraint in
                            self.constraints['linking']['up'][param]]
        constraints.extend(down_constraints)
        constraints = [constraint for constraint in constraints if constraint[1][1] is True]
        while (len(constraints) > 0):
            for param, constraint in constraints:
                param_1 = param
                param_2 = constraint[0]
                self.param_subset.merge(param_1, param_2)
