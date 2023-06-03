import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___input_dim____"
 ],
 "IN": [
 "VAR:X1 ___input_dim____",
 "VAR:X2 ___input_dim____"
 ]
}
END_PROPS
"""


class Multiply(torch.nn.Module):
    def __init__(self, 
        ___input_dim____ = 10, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(Add, self).__init__()
        self.device = ___device____
        self.dtype = ___dtype____

    def forward(self, X):
        x1 = X['X1']
        x2 = X['X2']
        assert x1.shape == x2.shape, "The tensors to be added must have the same shape"
        res = {"Y": x1 * x2}
        return res
