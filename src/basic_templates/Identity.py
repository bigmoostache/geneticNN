import torch

"""
BEGIN_PROPS
{
"variables": {
    "X": {
        "io" : "in",
        "dim": 1,
        "type": "float"
        },
    "Y": {
        "io" : "out",
        "dim": 1,
        "type":"float"
        }
    },
"parameters" : {},
"constraints" : [
    ["equality",[ "_X_0","_Y_0"]]
]
}
END_PROPS
"""


class Identity(torch.nn.Module):
    """
    Applies the Identity Operator


    Shape:
    Input: (∗), where ∗ means any number of dimensions.
    Output: (∗), same shape as the input.
    """

    def __init__(self,
                 device='cpu',
                 dtype=torch.float32,
                 ):
        super(Identity, self).__init__()

    def forward(self, X):
        x = X['X']
        res = {"Y": x}
        return res
