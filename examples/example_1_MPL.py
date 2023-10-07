import sys, os

#script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the project folder
#project_path = os.path.join(script_dir, '..')

# Add the project folder to the Python path
#sys.path.append(project_path)

sys.path.append("../src")
from src.author import Author
from src.modelskeleton import ModelSkeleton

if __name__ == "__main__":
    # MLP
    models = {  # For now, only the template is required
        'Linear_1': {"source": "basic_templates", "type": "Linear"},
        'Linear_2': {"source": "basic_templates", "type": "Linear"},
        'Linear_3': {"source": "basic_templates", "type": "Linear"},
        'ReLU1': {"source": "basic_templates", "type": "ReLU"},
        'ReLU2': {"source": "basic_templates", "type": "ReLU"},
        'ReLU3': {"source": "basic_templates", "type": "ReLU"},
    }

    runs = [  # Does not have to be sorted, it will be sorted by myRealWriter
        {'id': 'Linear_1', 'inputs': {'X': [-1, 'X']}},
        {'id': 'ReLU1', 'inputs': {'X': [0, 'Y']}},
        {'id': 'Linear_2', 'inputs': {'X': [1, 'Y']}},
        {'id': 'ReLU2', 'inputs': {'X': [2, 'Y']}},
        {'id': 'Linear_3', 'inputs': {'X': [3, 'Y']}},
        {'id': 'ReLU3', 'inputs': {'X': [4, 'Y']}}
    ]

    output = {  # Model outputs
        "Y": [5, "Y"],
    }

    model_skeleton = ModelSkeleton(models, runs, output)
    r = Author("MLP", model_skeleton, save_dir="test_output")

#%%
