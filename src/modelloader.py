import os
import sys
import importlib


class ModelLoader:
    def __init__(self, base_path=None):
        self.models_modules = {}
        self.models_classes = {}
        self.base_path = base_path or os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        sys.path.append(self.base_path)

    def load_model_class(self, source, model_type, reload=False):
        base_name = os.path.basename(self.base_path)
        model_module_name = f"{base_name}.{source}.{model_type}"

        if reload or model_module_name not in sys.modules:
            model_module = importlib.import_module(model_module_name)
        else:
            model_module = importlib.reload(sys.modules[model_module_name])

        self.models_modules.setdefault(source, {})[model_type] = model_module
        self.models_classes.setdefault(source, {})[model_type] = getattr(model_module, model_type)

    def new(self, source, model_type, model_parameters, reload=False):
        if source not in self.models_classes or model_type not in self.models_classes[source]:
            self.load_model_class(source, model_type)
        elif reload:
            self.load_model_class(source, model_type, reload=True)

        return self.models_classes[source][model_type](**model_parameters)
