import torch

"""
BEGIN_PROPS
{
"variables":{
    "X1":{
        "io": "in",
        "dim": 1,
        "type": "float"
    },
    "X2":{
        "io": "in",
        "dim": 1,
        "type": "float"
    },
    "Y":{
        "io": "out",
        "dim": 1,
        "type": "float"
    }
},
"parameters": {
},
"constraints": [
    ["equality", ["_X1_0", "_X2_0", "_Y_0"]]
]
}
END_PROPS
"""


class Add(torch.nn.Module):
    def __init__(self,
                 device='cpu',
                 dtype=torch.float32,
                 ):
        super(Add, self).__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, X):
        x1 = X['X1']
        x2 = X['X2']
        assert x1.shape == x2.shape, "The tensors to be added must have the same shape"
        res = {"Y": x1 + x2}
        return res
