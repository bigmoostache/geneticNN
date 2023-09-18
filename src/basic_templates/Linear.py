import torch

"""
BEGIN_PROPS
{
"variables": {
    "X": {
        "io" : "in",
        "dim": "1",
        "type" : "float"
        },
    "Y": {
        "io" : "out",
        "dim": "1",
        "type": "float"
        }
    },
"parameters" : {
    "bias": {"type" : "bool"},
    "input_dim": {"type" : "int"}, 
    "output_dim": {"type" : "int"}
},
"constraints" : [
    ["equality",["input_dim", "_X_0"]], 
    ["equality",["output_dim", "_Y_0"]]
]
 "OUT": [
 "VAR:Y ___output_dim____"
 ],
 "IN": [
 "VAR:X ___input_dim____"
 ]
}
END_PROPS
"""


class Linear(torch.nn.Module):
    def __init__(self, 
        ___input_dim____ = 10, 
        ___output_dim____  = 10, 
        ___bias____ = True, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(___input_dim____, 
                                ___output_dim____, 
                                bias=___bias____, 
                                device=___device____, 
                                dtype=___dtype____
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.linear(x)}
        return res