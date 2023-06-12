import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y first dimension is ___input_dim____",
 "VAR:Y ___output_dim____"
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
        ___output_dim____ = 10
        ):
        super(Add, self).__init__()

    def forward(self, X):
        res = {"Y": X['X'].T}
        return res
