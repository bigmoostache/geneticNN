import torch

"""
BEGIN_PROPS
{
"variables": {
    "X": {
        "io" : "in",
        "dim": "1",
        "type": "float"
        },
    "Y": {
        "io" : "out",
        "dim": "1",
        "type":"float"
        }
    },
"parameters" : {
    "bias": {"type" : "bool"},
    "relu_dim": {"type" : "int"}
},
"constraints" : [
    ["equality",["relu_dim", "_X_0","_Y_0"]]
]
}
END_PROPS
"""

class ReLU(torch.nn.Module):
    """
    Applies the rectified linear unit function element-wise:
    ReLU(x) = max(0, x)

    Parameters:
    inplace (bool): can optionally do the operation in-place. Default: False

    Shape:
    Input: (∗), where ∗ means any number of dimensions.
    Output: (∗), same shape as the input.
    """
    def __init__(self, 
        ___input_dim____ = 10, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(ReLU, self).__init__()
        self.relu = torch.nn.ReLU(inplace = False, 
                                    )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.relu(x)}
        return res

