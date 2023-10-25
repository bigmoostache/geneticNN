import sys
sys.path.append("../src")
from src.Base.author import Author
from src.Base.modelskeleton import ModelSkeleton

# MLP
models = {
    'Transpose': {"source": "basic_templates",  "type": "Transpose"},
    'MATMUL': {"source": "basic_templates",  "type": "MatMul"},
    'Scale': {"source": "basic_templates",  "type": "Scale"},
    'Softmax': {"source": "basic_templates",  "type": "Softmax"},
    'MATMUL2': {"source": "basic_templates",  "type": "MatMul"},
}

runs = [
    {'id': 'Transpose', 'inputs': {'X': [-1, 'K']}},
    {'id': 'MATMUL', 'inputs': {'M1': [-1, 'Q'], 'M2':[0, 'Y']}},
    {'id': 'Scale', 'inputs': {'X': [1, 'Y']}},
    {'id': 'Softmax', 'inputs': {'X': [2, 'Y']}},
    {'id': 'MATMUL2', 'inputs': {'M1': [3, 'Y'], 'M2': [-1, 'V']}}
]

output = {
    "Y": [4, "Y"],
}

model_skeleton = ModelSkeleton(models, runs, output)

r = Author("FFW", model_skeleton,save_dir="test_output" )
