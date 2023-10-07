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
    "scale": {"type": "float", "default": 1.0}
},
"constraints": [
    ["equality", ["_X_0", "_Y_0"]],
    ["equality", ["_X_1","_Y_1"]]
]
}
END_PROPS
"""


class Scale(torch.nn.Module):
    def __init__(self,
        device = 'cpu',
        dtype = torch.float32,
        scale = 1.0
        ):
        super(Scale, self).__init__()
        self.device = device
        self.dtype = dtype
        self.scale = scale

    def forward(self, X):
        res = {"Y": X['X']*self.scale}
        return res
