import torch

"""
BEGIN_PROPS
{
"variables":{
    "X":{
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
"parameters": {},
"constraints": [
    ["equality", ["_X_0", "_Y_0"]]
]
}
END_PROPS
"""

class Softmax(torch.nn.Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor rescaling them so that 
    the elements of the n-dimensional output Tensor lie in the range [0, 1] and sum to 1.
    
    NOTE: This module doesn’t work directly with NLLLoss, which expects the Log to be computed 
    between the Softmax and itself. Use LogSoftmax instead (it’s faster and has better numerical properties).
    """
    def __init__(self, 
        dim = None,
        device = 'cpu',
        dtype= torch.float32,
        ):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=dim,
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.softmax(x)}
        return res
