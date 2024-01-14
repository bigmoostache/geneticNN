import json
from typing import Any


class ModelTemplate:
    def __init__(self, template_type, source, template_source=""):
        self.template_type = template_type
        self.source = source
        self.template_source = template_source
        self._props = {}

    @property
    def properties(self):
        if self._props == {}:
            self.load_template(self.template_source)
        return self._props

    @properties.setter
    def properties(self, props: dict):
        self._props = props

    def load_template(self, source_file: str):
        text = open(source_file).read()
        text2 = text.split("BEGIN_PROPS")[1]
        text2 = text2.split("END_PROPS")[0]
        self._props = json.loads(text2)

    def build_complete_template(self):
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        if self._props is not None:
            return self._props["parameters"]


class InputTemplate(ModelTemplate):
    def __init__(self):
        super().__init__("input", "Base", "Base")

    @property
    def properties(self):
        self._props = []
        return self._props

    def load_template(self, source_file: str):
        pass


class OutputTemplate(ModelTemplate):
    def __init__(self):
        super().__init__("output", "Base", "Base")

    @property
    def properties(self):
        self._props = []
        return self._props

    def load_template(self, source_file: str):
        pass
