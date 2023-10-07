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
"parameters" : {},
"constraints" : [
    ["equality",[ "_X_0","_Y_0"]]
]
}
END_PROPS
"""

class Sigmoid(torch.nn.Module):
    """
    Applies the element-wise function:
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    Shape:
    Input: (∗), where ∗ means, it can take any number of dimensions.
    Output: (∗), same shape as the input.
    """

    def __init__(self,
                 device='cpu',
                 dtype=torch.float32
                 ):
        super(Sigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        x = X['X']
        res = {"Y": self.sigmoid(x)}
        return res
