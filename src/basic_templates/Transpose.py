import torch

"""
BEGIN_PROPS
{
"variables":{
    "X":{
        "io": "in",
        "dim": 2,
        "type": "float"
    },
    "Y":{
        "io": "out",
        "dim": 2,
        "type": "float"
    }
},
"parameters": {
},
"constraints": [
    ["equality", ["_X_0", "_Y_1"]],
    ["equality", ["_X_1","_Y_0"]]
]
}
END_PROPS
"""


class Transpose(torch.nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, X):
        res = {"Y": X['X'].T}
        return res
