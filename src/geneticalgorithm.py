import heapq
from abc import ABC, abstractmethod

from .Base.author import Author
from .Base.modelloader import ModelLoader
from .StructureModifiers.SmallUpdateModifier import SmallUpdateModifier
from .StructureModifiers.StructureModifier import  StructureModifier
import random
import math
from .Base.modelskeleton import ModelSkeleton
from .Base.modelproperties import ModelPropertiesMapper
from typing import Type, Callable, Tuple
import logging
import torch


def sample_integer_gaussian(mu=0, sigma=1):
    return math.trunc(random.gauss(mu=mu, sigma=sigma))


class Model:
    def __init__(self, model_name: str, model_skeleton: ModelSkeleton, model_properties: ModelPropertiesMapper = None,
                 model_parameters: dict = None) -> None:
        self.model_name = model_name
        self.model_skeleton = model_skeleton
        self.model_parameters = model_parameters
        self.model_properties = model_properties


class GeneticHistory(ABC):

    @abstractmethod
    def register_new_generation(self, generation):
        pass


class SimpleGeneticHistory(GeneticHistory):

    def __init__(self, history_size=100, numbers_per_generation=5):
        self.history_size = history_size
        self.number_per_generation = numbers_per_generation
        self.history = []

    def register_new_generation(self, generation):
        n_left = self.history_size - len(self.history) - self.number_per_generation
        if n_left < 0:
            self.history = self.history[n_left:]
        best_models = heapq.nsmallest(self.number_per_generation, generation, key=lambda x: x["loss"])
        self.history.extend(best_models)


class GeneticAlgorithm:
    def __init__(self,
                 starting_models: list[Model],
                 allowed_list: list[dict],
                 trials_per_generation: int = 10,
                 number_of_models_to_keep: int = 2,
                 structure_modifier: Type[StructureModifier] = SmallUpdateModifier,
                 model_sampler: Callable[[list[Tuple[int, Model]]], Tuple[int, Model]] = random.choice,
                 integer_sampler: Callable[[int], int] = sample_integer_gaussian,
                 float_sampler: Callable[[float], float] = random.gauss,
                 history: GeneticHistory = SimpleGeneticHistory(),
                 base_path: str = None,
                 save_dir: str = "genetic",
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float,
                 ignore_exceptions:bool = True):

        self.trials_per_generation = trials_per_generation
        self.number_of_models_to_keep = number_of_models_to_keep
        self.structure_modifier = structure_modifier(allowed_models=allowed_list)
        self.model_sampler = model_sampler
        self.integer_sampler = integer_sampler
        self.float_sampler = float_sampler
        self.bool_sampler = lambda: random.randint(a=0, b=1)
        self.history = history
        self.bests = [{"loss": 10000, "model": model} for model in starting_models]
        self.allowed_list = allowed_list
        self.generation = 0
        self.model_loader = ModelLoader(base_path=base_path)
        self.save_dir = save_dir
        self.device = device
        self.dtype = dtype
        self.current_gen = None
        for model_dict in self.bests:
            model = model_dict["model"]
            if model.model_properties is None:
                model.model_properties = ModelPropertiesMapper(model.model_skeleton)
            if model.model_parameters is None:
                model.model_parameters = model.model_properties.get_global_defaults()
            #self.keep_constraints(model)
        self.ignore_exceptions = ignore_exceptions
        self.errors = {}

    def resolve_parameters(self, old_model: Model, new_model: Model):
        old_parameters = old_model.model_parameters
        old_properties = old_model.model_properties
        new_properties = new_model.model_properties
        new_parameter = new_properties.get_global_defaults()
        for parameter, value in old_parameters.items():
            for equivalent_name in old_properties.param_subset.get_equivalence_class(parameter):
                if equivalent_name in new_properties.param_subset.data:
                    new_parameter[new_properties.param_subset[equivalent_name]] = value
                    break

        return new_parameter

    def keep_constraints(self, model: Model, io_size_constraints):

        input_models = model.model_skeleton.get_direct_children(-1)
        input_params = {}
        for run_id in input_models:
            run = model.model_skeleton.runs[run_id]
            for input_var, data in run["inputs"].items():
                if data[0] == -1:

                    model_props = model.model_properties
                    variable_parameter = model_props.get_variable_parameter_name(run['id'], input_var, 0)
                    global_param = model_props.param_subset[variable_parameter]
                    input_params[data[1]] = global_param
                    model.model_parameters[
                        model_props.param_subset[global_param]] = io_size_constraints['inputs'][data[1]]

        output_params = {}
        for var, data in model.model_skeleton.outputs.items():
            model_id = model.model_skeleton.runs[data[0]]['id']
            variable_parameter = model.model_properties.get_variable_parameter_name(model_id, data[1], 0)
            global_param = model.model_properties.param_subset[variable_parameter]
            output_params[data[1]] = global_param
            model.model_parameters[global_param] = io_size_constraints['outputs'][data[1]]

        #check input-output compatibility
        for i,p_i in input_params.items():
            for o,p_o in output_params.items():
                if p_i == p_o and io_size_constraints['inputs'][i] != io_size_constraints['outputs'][o]:
                    return False
        return True

    def random_update_parameters(self, model: Model):
        model_props = model.model_properties.get_high_order_props()
        for parameter in model.model_parameters.keys():
            param_props = model_props["parameters"][parameter]
            if param_props['type'] == 'bool':
                model.model_parameters[parameter] = self.bool_sampler()
            elif param_props['type'] == 'int':
                value = model.model_parameters[parameter]
                model.model_parameters[parameter] = max(1, self.integer_sampler(mu=value,sigma=30))
            elif param_props['type'] == 'float':
                value = model.model_parameters[parameter]
                model.model_parameters[parameter] = self.float_sampler(mu=value,sigma=30)
            else:
                raise TypeError(f"param {parameter} of unsupported type : {param_props['type']}")
        return model.model_parameters

    def generate_candidates(self, generation, number_of_candidates, io_sizes_constraints):
        new_models_list = []
        for model_dict in self.bests:
            model = model_dict["model"]
        k = 0
        while len(new_models_list) < number_of_candidates:
            starting_id, starting_model = self.model_sampler([(i, m['model']) for i, m in enumerate(self.bests)])
            is_add_transform = self.bool_sampler()
            # don't remove the only run if there is only one left
            if len(starting_model.model_skeleton.runs) == 1:
                is_add_transform = True
            if is_add_transform:
                (proposed_submodel,
                 proposed_input,
                 proposed_output) = self.structure_modifier.propose_random_add((starting_model.model_skeleton,
                                                                                starting_model.model_properties))
                new_model_skeleton = self.structure_modifier.add_submodel(model=(starting_model.model_skeleton,
                                                                                starting_model.model_properties),
                                                                          submodel_to_add=proposed_submodel,
                                                                          input_vars=proposed_input,
                                                                          output_vars=proposed_output,
                                                                          inplace=False)
            else:
                (proposed_run,
                 proposed_pairings) = self.structure_modifier.propose_random_remove((starting_model.model_skeleton,
                                                                                     starting_model.model_properties))
                new_model_skeleton = self.structure_modifier.remove_run(model=(starting_model.model_skeleton,
                                                                               starting_model.model_properties),
                                                                        run_to_remove=proposed_run,
                                                                        replacement_pairings=proposed_pairings,
                                                                        inplace=False)

            new_model = Model(model_name=f"model_{generation}_{k}", model_skeleton=new_model_skeleton)
            new_model.model_properties = ModelPropertiesMapper(new_model_skeleton)
            new_model.model_parameters = self.resolve_parameters(starting_model, new_model)
            new_model.model_parameters = self.random_update_parameters(new_model)
            if not self.keep_constraints(new_model, io_sizes_constraints):
                continue
            new_models_list.append(new_model)
            k+=1
        return new_models_list

    def test_model(self,closure, model_name, parameters):
        model_instance = self.model_loader.new(self.save_dir,
                                               model_name,
                                               parameters,reload = True)
        return closure(model_instance)

    def next_generation(self, closure, io_sizes_constraints, logger = logging):
        logging.info(f"--genetic algorithm generation {self.generation}:--")
        models_to_test = self.generate_candidates(generation=self.generation,
                                                  number_of_candidates=self.trials_per_generation,
                                                  io_sizes_constraints=io_sizes_constraints)
        self.current_gen = models_to_test
        logger.info(f"Testing models:")
        generation_result = []
        total = len(models_to_test)
        for i, model in enumerate(models_to_test):
            logger.info(f"\t model {i + 1}/{total} ")
            Author(model_name=model.model_name,
                   model_skeleton=model.model_skeleton,
                   model_properties=model.model_properties,
                   save_dir=self.save_dir, logger = logger)
            parameters = model.model_parameters.copy()
            parameters['device'] = self.device
            parameters['dtype'] = self.dtype
            if self.ignore_exceptions:
                loss = 11000
                try:
                    loss = self.test_model(closure, model.model_name, parameters)
                except Exception as e:
                    print(f"model {i} of generation {self.generation} had an error")
                    self.errors[f"{self.generation}_{i}"] = e
            else:
                loss = self.test_model(closure, model.model_name, parameters)

            generation_result.append({"loss": loss, "model": model})
        best_result = heapq.nsmallest(1, iterable=generation_result, key=lambda x: x["loss"])
        logger.info(f"best model loss : {best_result[0]['loss']}")

        self.bests = heapq.nsmallest(self.number_of_models_to_keep,
                                     iterable=generation_result,
                                     key=lambda x: x["loss"])

        self.history.register_new_generation(generation=generation_result)

        self.generation += 1
