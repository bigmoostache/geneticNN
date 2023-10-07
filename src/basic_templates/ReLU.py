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
                 device='cpu',
                 dtype=torch.float32,
                 ):
        super(ReLU, self).__init__()
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, X):
        x = X['X']
        res = {"Y": self.relu(x)}
        return res
