import re, os, json, logging
logging.basicConfig(level=logging.INFO)

class Subsets:
    def __init__(self, elements):
        self.data = {el: {el} for el in elements}
        
    def __getitem__(self, item):
        if item in self.data:
            return min(self.data[item])
        else:
            raise KeyError(f'No key {item} found')

    def merge(self, el1, el2):
        if el1 in self.data and el2 in self.data:
            union = self.data[el1].union(self.data[el2])
            for el in union:
                self.data[el] = union
        else:
            raise ValueError('One or both elements are not found')
        
class Author():
    
    def __init__(self, model_name, models, runs, output):
        """
        Constructor for the model generator class.

        Args:
        model_name (str): The name of the model to be generated.
        path_templates (str): The directory path where template files are located.
        models (dict): A dictionary containing the model specifications.
        runs (list): A list containing information on how to combine different models.
        output (dict): A dictionary defining the outputs of the final model.
        path_basic_templates (str): The directory path where basic template files are located.

        This function initializes the class, sorts the runs, and creates initial models based on provided specifications.
        It then generates the full model, resolving any warnings related to missing parameters or incompatible dimensions.
        If warnings persist after a maximum number of iterations (N_MAX = 20), it raises an exception. 
        Finally, it merges the parameters and writes the full model into a file in the path_templates directory.

        Note:
        It's assumed that the function is used within a larger framework where the input arguments are prepared and provided.
        """
        N_MAX = 2

        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.path_templates = os.path.join(script_directory, 'templates')
        self.path_basic_templates = os.path.join(script_directory, 'basic_templates')
        self.model_generation_file = 'model_generation.py'
        self.model_generation_file = os.path.join(script_directory,self.model_generation_file)
        self.model_name = model_name
        self.runs =  runs
        self.sort_runs()
        self.output = output
        self.models = models
        self.defaults = {}
        self.create_initial_models()
        lines, WARNINGS = self.full_model()
        i=0
        while WARNINGS and i<N_MAX:
            for WARNING in WARNINGS:
                self.solve_a_warning(WARNING)
            lines, WARNINGS = self.full_model()
            i += 1
        if i>N_MAX:
            print(WARNINGS)
            #raise Exception('max number of iterations reached')
        self.defaults = self.get_defaults()
        self.merge_parameters()
        lines, WARNINGS = self.full_model()
        model_file = model_name+'.py'
        model_path = os.path.join(self.path_templates, model_file)
        if not os.path.exists(model_path) or not os.path.exists(self.model_generation_file):
            self.add_model_generation_file(model_name=model_name)
            open(os.path.join(self.path_templates, '__init__.py'),'a').write(f"\nfrom . import  {model_name}")
        open(model_path, 'w').write(re.sub(r" +", " ", "\n".join(lines)))
        
    def sort_runs(self):
        """
        Sorts the 'runs' list based on their dependencies.

        The function goes through each pair of 'runs' and checks if an 'id' of a run 
        is in the 'inputs' of the other run. If it is, then they are swapped so that 
        the dependent run comes after the run it depends on.

        The function does not return anything. It modifies the 'runs' attribute of the 
        class in-place.
        """
        runs = self.runs
        n = len(runs)
        for i in range(n):
            for j in range(i+1, n):
                run_a = runs[i]
                run_b = runs[j]
                run_a_inputs = [v[0] for k,v in run_a['inputs'].items()]
                run_b_id = run_b['id']
                first = run_a
                second = run_b
                if run_b_id in run_a_inputs:
                    first = run_b
                    second = run_a
                runs[i] = first
                runs[j] = second
        self.runs = runs 

    def get_defaults(self):
        """
        Fetches the default values of the parameters used in the model.

        This method iterates through all the parameters used in the model.
        For each parameter, it finds a model that uses the parameter and
        fetches the default value of the parameter from the model. 

        The default values are then stored in the 'defaults' attribute of the class.
        
        Raises:
            Exception: If more than one default value is found for a parameter.
            
        Returns:
            None
        """

        self.defaults = {}
        self.get_all_parameters()
        for parameter in self.parameters:
            # First let's find a model that uses that parameter
            found = False
            for model_id, model in self.models.items():
                if found:
                    break
                for l_p, g_p in model["parameters"].items():
                    if g_p == self.subsets[parameter]:
                        found = (model_id, l_p)
                        break
            assert found
            default_value = self.models[found[0]]['defaults'].split('\n')
            default_value = [x for x in default_value if l_p in x]
            if len(default_value)!=1:
                raise Exception(f"Default values found fo {parameter} :{default_value}")
            default_value = default_value[0].split('=')[1]
            self.defaults[parameter] = default_value

    def create_initial_models(self):
        """
        Initializes the model structures and their parameters, and manages parameter equivalence classes.

        This method iterates through all the models, and for each model, 
        it sets up its associated parameters based on the model's template.
        It also parses and sets the properties and default values of the model. 

        During the initialization of model parameters, it checks whether the 
        template of the model is available in the 'path_basic_templates' directory. 
        If not, it looks for the template in the 'path_templates' directory.
        
        As part of the process, this function also constructs a 'Subsets' object
        containing all parameters from all models. The 'Subsets' object is used to 
        manage equivalence classes for parameters, enabling the merging of sets of 
        parameters that are deemed to be equal. This facilitates the handling of parameter 
        dependencies across different models.
        
        Raises:
            FileNotFoundError: If the model's template is not found in either 
            'path_basic_templates' or 'path_templates' directories.
            
        Returns:
            None
        """

        for model_id, model in self.models.items():
            model_file= model['template'] + '.py'
            if model_file in os.listdir(self.path_basic_templates):
                path = os.path.join(self.path_basic_templates, model_file)
            else:
                path = os.path.join(self.path_templates, model_file)
            # Load the text
            text = open(path).read()
            # Find the list of local parameters
            text2 = text.split("def __init__(self,")[1].split("):")[0]
            pattern = r"___(.*?)____"
            matches = re.findall(pattern, text2)
            matches = list(set(matches))
            # Initilize associated global parameters
            self.models[model_id]["parameters"] = {
                f"___{p}____" : f"___{p}_{model_id}____" for p in matches
            }
            # to delete? _ = self.models[model_id]["parameters"]
            # Load json and parse it a little
            text2 = text.split("BEGIN_PROPS")[1]
            text2 = text2.split("END_PROPS")[0]
            props = json.loads(text2)
            props['IN'] = [x.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') for x in props['IN']]
            props['OUT'] = [x.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') for x in props['OUT']]

            self.models[model_id]["props"] = props
            # Load the defaults
            defaults = text.split("def __init__(self,")[1].split("):")[0].replace(", ", "").replace(",", "")
            self.models[model_id]['defaults'] = defaults
        all_parameters = [
            param
            for model in self.models.values()
            for param in model["parameters"].values()
        ]
        self.subsets = Subsets(all_parameters)
    
    def solve_a_warning(self, warning):
        """
        Resolves a warning raised due to potential parameter inconsistencies in a model.
        
        This method identifies the model and the variable causing the warning and attempts to resolve the issue.
        It first locates the model and the corresponding variable that has raised the warning. 
        If the model_id is '0', it indicates that the input is a global input to the main model, not an output 
        of a sub-model. In this case, the function simply returns as there's no warning to be solved within 
        a sub-model.
        
        The method then identifies the parameters involved in the warning and attempts to find a match
        between the warning pattern and the output patterns of the model. 
        If a match is found, it suggests that the model's output can potentially resolve the warning.
        
        As part of the process, the equivalence classes of the matching parameters (local parameters in the model's
        output and the global parameters in the warning) are merged in the 'subsets' object.
        This effectively makes these parameters equivalent, helping to resolve the warning.
        
        If no matching output pattern is found for the warning pattern, an exception is raised indicating an
        'Unsolvable requirement'.
        
        Args:
            warning (str): The warning message to be resolved, which usually indicates a parameter inconsistency.
            
        Raises:
            Exception: If no output pattern in the model matches the warning pattern, indicating an unsolvable requirement.
        
        Returns:
            None
        """
        logging.info(f"trying to solve {warning}")
        # First, let's retrieve the interesting model
        _ = [x.split(':') for x in warning.split(' ') if "VAR:" in x][0]
        model_id, variable_name = _[2], _[1]
        if model_id in {0, '0'}:
            return
        warning = warning.replace(f"VAR:{variable_name}:{model_id}", f"VAR:{variable_name}")
        # Second let's find the possible outputs that could give you that
        params_involved = re.findall(r"___(.*?)____", warning)
        params_involved = [f"___{x}____" for x in list(set(params_involved))]
        warning_template = warning

        for param in params_involved:
            warning_template = warning_template.replace(param, "___###____")
        to_match = None
        for out in self.models[model_id]["props"]["OUT"]:
            params_involved_out = re.findall(r"___(.*?)____", out)
            params_involved_out = [f"___{x}____" for x in params_involved_out]
            out_template = out
            for param in params_involved_out:
                out_template = out_template.replace(param, "___###____")
            if out_template==warning_template:
                to_match = out
                break
        if not to_match:
            raise Exception(f"Unsolvable requirement {warning} in model {model_id}")
        for local_param, global_param in zip(params_involved_out, params_involved):
            logging.info(f"setting {local_param} to {global_param} in model {model_id}")
            self.subsets.merge(self.models[model_id]['parameters'][local_param], global_param)

    def get_all_translated_props(self):
        """
        Processes the input and output properties of all the models and returns a list of input and output 
        properties that are well-formed and a list of warnings for ill-formed inputs.

        This method iterates through each run and model, translating local parameters to global parameters 
        for both inputs (INs) and outputs (OUTs). It then filters out any outputs from the set of inputs.
        An input is considered 'well-formed' if all its variables are global inputs to the main model (i.e., 
        model_id is '0'). Any input that doesn't meet this criterion is considered 'ill-formed' and is added 
        to the warnings list.
        
        The method also processes the output list, replacing any matched variables with their corresponding 
        output variables and keeping only those outputs that contain at least one matched variable.
        
        Args:
            None
            
        Returns:
            tuple: A tuple containing three lists:
                INs (list): A list of well-formed input properties.
                OUTs (list): A list of well-formed output properties.
                WARNINGS (list): A list of ill-formed input properties.
        """
        INs, OUTs = [],  []
        for run in self.runs:
            model = self.models[run["id"]]
            INs += [self.local_to_global_run(x, run) for x in model["props"]['IN']]
        for model_id, model in self.models.items():
            OUTs += [self.local_to_global_output(OUT, model_id) for OUT in model["props"]["OUT"]]
        INs = set(INs)
        OUTs = list(set(OUTs))

        def ok(IN):
            tokens = set([x.split(":")[2] for x in IN.split(" ") if 'VAR:' in x])
            return set(tokens) == {'0'}
        
        def keep_out(OUT):
            variables = {f'VAR:{v["variable"]}:{v["model_id"]}':f'VAR:{k}' for k,v in self.output.items()}
            keep = False
            for variable in variables:
                if variable in OUT:
                    keep = True
                    OUT = OUT.replace(variable, variables[variable])
            return keep, OUT

        WARNINGS = {IN for IN in INs if not ok(IN)}
        for OUT in OUTs:
            WARNINGS.discard(OUT)
        WARNINGS = list(WARNINGS)

        INs = [IN.replace(":0", "") for IN in INs if ok(IN)]
        OUTs = [keep_out(x) for x in OUTs]
        OUTs = [x[1] for x in OUTs if x[0]]
        return INs, OUTs, WARNINGS
    
    def local_to_global_run(self, statement, run):
        """
        Transforms a statement from local parameters to global parameters for a given run.

        This method takes a statement and a run, and converts local parameters in the statement to
        global parameters according to the mappings stored in self.models for the given run. 
        After this conversion, it further transforms each token in the statement that contains a variable, 
        by replacing it with a formatted string containing the original variable's name and the ID of the model 
        from where this variable originates.

        Args:
            statement (str): The statement in which local parameters need to be transformed to global parameters.
            run (dict): The run for which the transformation needs to be performed.

        Returns:
            str: The transformed statement with global parameters and variables formatted with their origin model's ID.
        """
        model_id = run['id']
        for k,v in self.models[model_id]['parameters'].items():
            statement = statement.replace(k,self.subsets[v])
        statement = statement.split(' ')
        def transform(token):
            if 'VAR:' not in token:
                return token
            variable_name = token.split(':')[1]
            variable_origin = run["inputs"][variable_name]
            model_origin_id = variable_origin[0]
            variable_origin_name = variable_origin[1]
            token = f"VAR:{variable_origin_name}:{model_origin_id}"
            return token
        statement = ' '.join([transform(token) for token in statement])
        return statement
    
    def local_to_global_output(self, statement, model_id):
        """
        Converts a statement from local parameters to global parameters and transforms variable tokens in the context of model output.

        This method first replaces local parameters in the statement with global parameters according to the mappings stored in self.models for a given model_id. 
        Then, it further transforms each token in the statement that contains a variable, by appending the model_id to it.

        Args:
            statement (str): The statement in which local parameters need to be transformed to global parameters.
            model_id (str): The ID of the model for which the transformation needs to be performed.

        Returns:
            str: The transformed statement with global parameters and variables appended with the model_id.
        """
        for k,v in self.models[model_id]['parameters'].items():
            statement = statement.replace(k,self.subsets[v])
        def transform(token):
            if 'VAR:' not in token:
                return token
            return f"{token}:{model_id}"
        statement =  ' '.join([transform(token) for token in statement.split(' ')])
        return statement
    
    def initilization_lines(self):
        """
        Generates the initialization lines of code for the model, taking into account all the submodels contained within it.

        This method retrieves all parameters across models and generates the initialization lines of code for each model.
        For each model, a line of code is created to initialize it using its template name and parameters.
        It then prefaces these lines with the `super` call to initialize the parent class and indents all lines for proper Python syntax.
        It also generates the method signature of the '__init__' method with default values for all parameters.
        The final output is a list of all lines of code required for initialization of the model.

        Returns:
            list: A list of lines of Python code representing the initialization of the model.
        """
        parameters = self.get_all_parameters()
        # Initialization part
        lines = []
        for model_id, model in self.models.items():
            template_name = model["template"]
            line = f"self.model_{model_id} = {template_name}.{template_name}("
            for k,v in model["parameters"].items():
                line += f"{k} = {self.subsets[v]}, "
            line += ")"
            lines += [
                f"# Initializing model {model_id}",
                line
            ]
        lines = [f"super({self.model_name}, self).__init__()"] + lines
        lines = ["\t" + x for x in lines]
        first_lines = [f"def __init__(self, "] 

        self.get_defaults()

        for parameter in self.parameters:
            default_value = self.defaults[parameter]
            first_lines.append(f"\t{parameter} = {default_value},")

        first_lines += ["\t):"]
        lines = first_lines + lines
        return lines

    def forward_lines(self):
        """
        Generates the forward pass lines of code for the model.

        This method loops through all the runs of submodels contained within the main model.
        For each run, it creates a dictionary `Z` that maps each input variable to its corresponding value.
        These values are either fetched from the output of another submodel or from the main model's input.
        It then makes a forward pass through the submodel and stores the output in a model-specific output dictionary.
        Finally, it creates lines of code to aggregate the results from all submodels and return the final output.

        Returns:
            list: A list of lines of Python code representing the forward pass of the model.
        """
        # let's write the forward part
        lines = []
        import json
        for run_id, run in enumerate(self.runs):
            model_id = run['id']
            inputs =  run['inputs']
            lines += [
                f"# Sub-model run {run_id}",
                "Z = {"
            ]
            for k,v in inputs.items():
                if v[0] != 0:
                    lines.append(f"  \"{k}\" : model_output_{v[0]}[\"{v[1]}\"],")
                else:
                    lines.append(f"  \"{k}\" : X[\"{v[1]}\"],")
            lines += [
                "}",
                f"model_output_{model_id} = self.model_{model_id}(Z)",
            ]
        lines.append("# Aggregating results")
        lines.append("RESULT = {}")
        for output_name, output_source in self.output.items():
            model_id = output_source["model_id"]
            variable_to_fetch = output_source["variable"]
            lines.append(f"RESULT[\"{output_name}\"] = model_output_{model_id}[\"{variable_to_fetch}\"]")
        lines.append("return RESULT")

        lines = ["def forward(self, X):"]+['\t'+x for x in lines]
        return lines

    def imports(self):
        """
        Generates a list of Python import statements for the model.

        This method goes through all the submodels in the main model and creates an import line for each unique submodel template.
        The function also adds an import statement for the `torch` module. 
        The output is a list of unique import statements.

        Returns:
            list: A list of unique import statements needed for the model.
        """
        lines = ["import torch"]
        for model in self.models.values():
            import_name = model["template"]
            import_file = import_name+'.py'
            if import_file in os.listdir(self.path_basic_templates):
                lines.append(f"from src.basic_templates import {import_name}")
            else:
                lines.append(f"from src.templates.{import_name} import {import_name}")
        lines = list(set(lines))
        return lines

    def full_model(self):
        """
        Generates a list of all the lines that make up the model.

        This method combines the import statements, class initializations, and forward function, along with the constraints 
        for the inputs (IN), outputs (OUT), and warnings. 

        The constraints are stored in JSON format, between "BEGIN_PROPS" and "END_PROPS" tags.

        The method returns all the lines of the model and any warnings that may occur during the generation of 
        the translated properties.

        Returns:
            tuple: A tuple containing a list of lines of code that make up the model and a list of warnings.
        """
        self.get_all_parameters()

        import_lines = self.imports()

        class_lines = self.initilization_lines()
        class_lines.append('')
        class_lines += self.forward_lines()
        class_lines = ['\t' + x for x in class_lines]
        class_lines = [f"class {self.model_name}(torch.nn.Module):"]+class_lines

        IN, OUT, WARNINGS = self.get_all_translated_props()
        constraints = {
            "IN":IN,
            "OUT":OUT
        }
        if WARNINGS:
            constraints["WARNINGS"] = WARNINGS
        constraint_lines = ["","\"\"\"" , "BEGIN_PROPS"]
        constraint_lines += [json.dumps(constraints
        ,indent = 15)]
        constraint_lines += ["END_PROPS", "\"\"\""]


        lines = import_lines + constraint_lines + [''] + class_lines
        return lines, WARNINGS

    def merge_parameters(self):
        """
        Merges the 'device' and 'dtype' parameters of all models.

        This method iterates over each model's parameters. It assigns the first 'device' and 'dtype' parameter it encounters 
        as the representative for each category. It then merges subsequent 'device' and 'dtype' parameters with their 
        respective representatives using the Subsets class's merge method.

        This results in all 'device' parameters referring to the same device, and all 'dtype' parameters referring 
        to the same data type.
        """
        rep_device, rep_dtype = None, None
        for model_id, model in self.models.items():
            for k, v in model["parameters"].items():
                if 'device' in v:
                    if rep_device == None:
                        rep_device = self.models[model_id]["parameters"][k]
                    else:
                        self.subsets.merge(rep_device, self.models[model_id]["parameters"][k])
                if 'dtype' in v:
                    if rep_dtype == None:
                        rep_dtype = self.models[model_id]["parameters"][k]
                    else:
                        self.subsets.merge(rep_dtype, self.models[model_id]["parameters"][k])

    def get_all_parameters(self):
        """
        Retrieves and sorts all unique parameters across all models.

        This method iterates over each parameter in the Subsets object data. It then obtains its equivalent global 
        parameter using the Subsets class's __getitem__ method, resulting in a list of global parameters. 
        
        This list is converted into a set to remove duplicates, and then it is converted back into a list. The list of 
        unique global parameters is then sorted and assigned to the instance variable 'parameters'.
        """
        self.parameters = list(set([self.subsets[d] for d in self.subsets.data]))
        self.parameters.sort()

    def create_model_generation_file(self):
        file = """
import src.templates as templates
import src.basic_templates as basic_templates
def get_model(model_name):
    """
        imports = ""
        for model_file in os.listdir(self.path_basic_templates):
            model_name = model_file.split(".")[0]
            file += f"""
    if model_name == "{model_name}":
        return basic_templates.{model_name}.{model_name}()"""
            imports += f'''from . import {model_name}\n'''
        open(os.path.join(self.path_basic_templates, '__init__.py'),'w').write(imports)
        imports = ""
        for model_file in os.listdir(self.path_templates):
            model_name = model_file.split(".")[0]
            file += f"""
    if model_name == "{model_name}":
        return templates.{model_name}.{model_name}()"""
            imports += f'''from . import {model_name}\n'''
        open(os.path.join(self.path_templates, '__init__.py'),'w').write(imports)
        open(self.model_generation_file, 'w').write(file)
    def add_model_generation_file(self,model_name):
        if not os.path.exists(self.model_generation_file):
            self.create_model_generation_file()
        else:
            string_to_add = f'''
    if model_name == "{model_name}":
        return templates.{model_name}.{model_name}()'''
            open(self.model_generation_file,'a').write(string_to_add)
