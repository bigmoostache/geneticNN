import re, os, json, logging
from .modelskeleton import ModelSkeleton
from .modelproperties import ModelPropertiesMapper

logging.basicConfig(level=logging.INFO)


class Author:

    def __init__(self, model_name: str, model_skeleton: ModelSkeleton, model_properties: ModelPropertiesMapper = None,
                 save_dir: str = "", source_dir="", logger = logging):
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
        logging.info(f"Building a model {model_name}!")
        self.source_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if source_dir=="" \
            else source_dir

        if os.path.isabs(save_dir):
            self.save_dir = save_dir
        else:
            self.save_dir = self.source_directory + '/' + save_dir

        self.model_factory_file = 'model_factory'
        self.model_factory_file = os.path.join(self.save_dir, self.model_factory_file)

        self.model_name = model_name
        self.graph = model_skeleton
        self.graph.reorder_runs()  # should not impact the constraints for now as they are model dependent and the reordering only reorders the list with impacting the relationship between runs/models

        self.model_properties = model_properties
        if self.model_properties is None:
            logging.info("no model properties given, building default!")
            self.model_properties = ModelPropertiesMapper(model_skeleton)

        self.defaults = self.model_properties.get_global_defaults()
        self.parameters = self.model_properties.get_all_globals()
        self.props = self.model_properties.get_high_order_props()
        lines = self.full_model()

        model_file = model_name + '.py'
        model_path = os.path.join(self.save_dir, model_file)
        '''
        if not os.path.exists(model_path) or not os.path.exists(self.model_factory_file):
            self.add_model_generation_file(model_name=model_name)
            open(os.path.join(self.save_dir, '__init__.py'), 'a').write(f"\nfrom . import  {model_name}")
        '''
        open(model_path, 'w').write(re.sub(r" +", " ", "\n".join(lines)))

    def initialization_lines(self):
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
        parameters = self.model_properties.get_all_globals()
        # Initialization part
        lines = [f"self.device = device", f"self.dtype = dtype"]
        for model_id, model in self.model_properties.model_skeleton.submodels.items():
            model_template = model["type"]
            model_source = model["source"]
            line = f"self.model_{model_id} = {model_source}_{model_template}("
            model_props = self.model_properties.sub_props[model_id]
            for k,prop in model_props["parameters"].items():
                if prop.get("virtual", False):
                    continue
                line += f"{k}={self.model_properties.get_global_parameter(model_id,k)}, "
            line += f"device=device, "
            line += f"dtype=dtype"
            line += ")"
            lines += [
                f"# Initializing model {model_id}",
                line
            ]
        lines = [f"super({self.model_name}, self).__init__()"] + lines
        lines = ["\t" + x for x in lines]
        first_lines = [f"def __init__(self, "]

        for parameter in self.parameters:
            if not self.props["parameters"][parameter].get("virtual", False):
                default_value = self.defaults[parameter]
                first_lines.append(f"\t\t\t {parameter}={default_value},")
        first_lines += ["\t\t\t device='cpu',", "\t\t\t dtype=torch.float32"]
        first_lines += ["\t\t\t ):"]
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
        for run_id, run in enumerate(self.graph.runs):
            model_id = run['id']
            inputs = run['inputs']
            lines += [
                f"# Sub-model run {run_id}",
                "Z = {"
            ]
            for k, v in inputs.items():
                if v[0] != -1:
                    lines.append(f"  \"{k}\" : run_output_{v[0]}[\"{v[1]}\"],")
                else:
                    lines.append(f"  \"{k}\" : X[\"{v[1]}\"],")
            lines += [
                "}",
                f"run_output_{run_id} = self.model_{model_id}(Z)",
            ]
        lines.append("# Aggregating results")
        lines.append("RESULT = {}")
        for output_name, output_source in self.graph.outputs.items():
            run_id = output_source[0]
            variable_to_fetch = output_source[1]
            lines.append(f"RESULT[\"{output_name}\"] = run_output_{run_id}[\"{variable_to_fetch}\"]")
        lines.append("return RESULT")

        lines = ["def forward(self, X):"] + ['\t' + x for x in lines]
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
        lines = []
        for model in self.graph.submodels.values():
            import_source = model["source"]
            import_name = model["type"]
            import_file = import_name + '.py'
            lines.append(f"from {import_source}.{import_name} import {import_name} as {import_source}_{import_name}")

        lines = list(set(lines))
        lines = ["import torch"] + lines
        return lines

    def props_lines(self):
        lines = ["", "\"\"\"", "BEGIN_PROPS"]

        lines += [json.dumps(self.props, indent="\t")]

        lines += ["END_PROPS", "\"\"\""]

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

        import_lines = self.imports()

        class_lines = self.initialization_lines()
        class_lines.append('')
        class_lines += self.forward_lines()
        class_lines = ['\t' + x for x in class_lines]
        class_lines = [f"class {self.model_name}(torch.nn.Module):"] + class_lines

        props_lines = self.props_lines()

        lines = import_lines + props_lines + [''] + class_lines
        return lines

    def create_model_generation_file(self):
        file = """
import src.templates as templates
import src.basic_templates as basic_templates
def get_model(model_name):
    """
        imports = ""
        for model_file in os.listdir(self.save_dir):
            model_name = model_file.split(".")[0]
            if model_name == '__init__':
                continue
            file += f"""
    if model_name == "{model_name}":
        return {model_name}.{model_name}()"""
            imports += f'''from . import {model_name}\n'''
        open(os.path.join(self.save_dir, '../__init__.py'), 'w').write(imports)
        open(self.model_factory_file, 'w').write(file)

    def add_model_generation_file(self, model_name):
        if not os.path.exists(self.model_factory_file):
            self.create_model_generation_file()
        else:
            string_to_add = f'''
    if model_name == "{model_name}":
        return templates.{model_name}.{model_name}()'''
            open(self.model_factory_file, 'a').write(string_to_add)
