import json
import re,os
def get_props(model_file):
    text = open(model_file).read()
    text2 = text.split("BEGIN_PROPS")[1]
    text2 = text2.split("END_PROPS")[0]
    props = json.loads(text2)
    return props


class Model_Properties():
    def __init__(self):

        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.path_templates = os.path.join(script_directory, 'templates')
        self.path_basic_templates = os.path.join(script_directory, 'basic_templates')
        self.props = None
        self.loc_glob_dic = {}
        self.glob_loc_dic = {}
        
    def generate_properties_from_file(self,model_file):
        self.props = get_props(model_file)
        for param in self.props['parameters']:
            self.loc_glob_dic['param']  = param
    
    def get_props(model_file):
        text = open(model_file).read()
        text2 = text.split("BEGIN_PROPS")[1]
        text2 = text2.split("END_PROPS")[0]
        props = json.loads(text2)
        return props

    def generate_properties_from_skeleton(model_skeleton):
        for model_type in model
