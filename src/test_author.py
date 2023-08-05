
from graph import Graph
from author import Author
# MLP
models = { # For now, only the template is required
    'Linear_1' : {"template" : "Linear"}, 
    'Linear_2' : {"template" : "Linear"}, 
    'Linear_3' : {"template" : "Linear"}, 
    'ReLU1' : {"template" : "ReLU"}, 
    'ReLU2' : {"template" : "ReLU"}, 
    'ReLU3' : {"template" : "ReLU"}, 
}

runs = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs':{'X':[-1, 'X']}},
    {'id' : 'Linear_2', 'inputs':{'X':[3, 'Y']}},
    {'id' : 'Linear_3', 'inputs':{'X':[4, 'Y']}},
    {'id' : 'ReLU1', 'inputs':{'X':[0, 'Y']}},
    {'id' : 'ReLU2', 'inputs':{'X':[1, 'Y']}},
    {'id' : 'ReLU3', 'inputs':{'X':[2, 'Y']}}
]

output = { # Model outputs
    "Y" : {"run_id" :5, "variable": "Y"},
}

r= Author("MLP", Graph(models, runs, output))