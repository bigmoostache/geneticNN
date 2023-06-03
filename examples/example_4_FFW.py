import sys
sys.path.append("C:/Users/Guillaume Draznieks/Documents/PROJETS/Biological GNN")
from src.author import Author

# MLP
models = {
    'Transpose': {"template": "Transpose"},
    'MATMUL': {"template": "MatMul"},
    'Scale': {"template": "Scale"},
    'Softmax': {"template": "Softmax"},
    'MATMUL2': {"template": "MatMul"},
}

runs = [
    {'id': 'Transpose', 'inputs': {'X': (0, 'K')}},
    {'id': 'MATMUL', 'inputs': {'M1': (0, 'Q'), 'M2':('Transpose', 'Y')}},
    {'id': 'Scale', 'inputs': {'X': ('MATMUL', 'Y')}},
    {'id': 'Softmax', 'inputs': {'X': ('Scale', 'Y')}},
    {'id': 'MATMUL2', 'inputs': {'M1': ('Softmax', 'Y'), 'M2':(0, 'V')}},
]

output = {
    "Y": {"model_id": 'MATMUL2', "variable": "Y"},
}

r = Author("ATTENTION", models, runs, output)
