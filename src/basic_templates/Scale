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


class Add(torch.nn.Module):
    def __init__(self, 
        ___input_dim____ = 10, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ___scale____ = 1.0
        ):
        super(Add, self).__init__()
        self.device = ___device____
        self.dtype = ___dtype____
        self.scale = ___scale____

    def forward(self, X):
        res = {"Y": X['X']*self.scale}
        return res
