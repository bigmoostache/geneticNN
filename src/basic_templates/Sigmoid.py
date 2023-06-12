import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___input_dim____"
 ],
 "IN": [
 "VAR:X ___input_dim____"
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
        ___input_dim____ = 10, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(Sigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        x = X['X']
        res = {"Y":self.sigmoid(x)}
        return res
