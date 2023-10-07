import torch

"""
BEGIN_PROPS
{
"variables":{
    "M1":{
        "io": "in",
        "dim": 2,
        "type": "float"
    },
    "M2":{
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
    ["equality", ["_M1_0", "_Y_0"]],
    ["equality", ["_M1_1", "_M2_0"]],
    ["equality", ["_M2_1", "_Y_1"]]
]
}
END_PROPS
"""


class MatMul(torch.nn.Module):
    def __init__(self,
        device = 'cpu',
        dtype = torch.float32,
        ):
        super(MatMul, self).__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, X):
        x1 = X['M1']
        x2 = X['M2']
        assert x1.dim() == 2 and x2.dim() == 2, "The tensors to be multiplied must be 2D"
        assert x1.shape[-1] == x2.shape[0], "The inner dimensions of the tensors must be the same for matrix multiplication"
        res = {"Y": torch.matmul(x1, x2)}
        return res
