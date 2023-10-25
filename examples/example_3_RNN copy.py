import sys
sys.path.append("C:/Users/Guillaume Draznieks/Documents/PROJETS/Biological GNN")
from src.Base.author import Author
# MLP
models = { # For now, only the template is required
    'Linear_1' : {"template" : "Linear"}, 
}

runs = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs':{'X':(0, 'X')}},
    {'id' : 'Linear_1', 'inputs':{'X':('Linear_1', 'Y')}},
]

output = { # Model outputs
    "Y" : {"model_id" :'Linear_1', "variable": "Y"},
}

r= Author("RNN_v2", models, runs, output)