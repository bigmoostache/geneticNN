import torch

"""
BEGIN_PROPS
{
"variables": {
    "X": {
        "io" : "in",
        "dim": 1,
        "type" : "float"
        },
    "Y": {
        "io" : "out",
        "dim": 1,
        "type": "float"
        }
    },
"parameters" : {
    "bias": {"type" : "bool", "default": 1},
    "input_dim": {"type" : "int", "default": 10}, 
    "output_dim": {"type" : "int", "default": 10}
},
"constraints" : [
    ["equality",["input_dim", "_X_0"]], 
    ["equality",["output_dim", "_Y_0"]]
]
}
END_PROPS
"""


class Linear(torch.nn.Module):
    def __init__(self, 
        input_dim = 10,
        output_dim  = 10,
        bias = True,
        device = 'cpu',
        dtype = torch.float32, **kwargs
        ):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(input_dim,
                                output_dim,
                                bias=bias,
                                device=device,
                                dtype=dtype
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.linear(x)}
        return res