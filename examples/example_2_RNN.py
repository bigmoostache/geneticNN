import sys
sys.path.append("../src")
from src.author import Author
from src.modelskeleton import ModelSkeleton
# MLP
models = { # For now, only the template is required
    'Linear_1' : {"source": "basic_templates", "type": "Linear"},
    'Linear_2' : {"source": "basic_templates", "type": "Linear"},
}

runs = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs': {'X': [-1, 'X']}},
    {'id' : 'Linear_2', 'inputs': {'X':[0, 'Y']}},
    {'id' : 'Linear_1', 'inputs': {'X':[1, 'Y']}},
]

output = { # Model outputs
    "Y" : [2, "Y"],
}

model_skeleton = ModelSkeleton(models, runs, output)
r= Author("RNN", model_skeleton, save_dir="test_output")