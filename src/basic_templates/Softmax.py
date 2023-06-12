import torch

"""
BEGIN_PROPS
{
 "IN": [
 "VAR:X ___input_dim____"
 ],
 "OUT": [
 "VAR:Y ___input_dim____"
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
        ___dim____ = None, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ___input_dim____ = 10,
        ):
        super(Softmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=___dim____, 
                                device=___device____, 
                                dtype=___dtype____
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.softmax(x)}
        return res
