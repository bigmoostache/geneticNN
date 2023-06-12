import sys,os

#add the folder to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..')
sys.path.append(project_path)

from src.author import Author

#starts with an MLP
models_init = { # For now, only the template is required
    'Linear_1' : {"template" : "Linear"}, 
    'Linear_2' : {"template" : "Linear"}, 
    'Linear_3' : {"template" : "Linear"}, 
    'ReLU1' : {"template" : "ReLU"}, 
    'ReLU2' : {"template" : "ReLU"}, 
    'ReLU3' : {"template" : "ReLU"}, 
}

runs_init = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs':{'X':(0, 'X')}},
    {'id' : 'Linear_2', 'inputs':{'X':('ReLU1', 'Y')}},
    {'id' : 'Linear_3', 'inputs':{'X':('ReLU2', 'Y')}},
    {'id' : 'ReLU1', 'inputs':{'X':('Linear_1', 'Y')}},
    {'id' : 'ReLU2', 'inputs':{'X':('Linear_2', 'Y')}},
    {'id' : 'ReLU3', 'inputs':{'X':('Linear_3', 'Y')}}
]

output = { # Model outputs
    "Y" : {"model_id" :'ReLU3', "variable": "Y"},
}

r_init= Author("model_01", models_init, runs_init, output)
from src import model_generation
model = model_generation.get_model("model_00")
print(model)


