import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___output_dim____"
 ],
 "IN": [
 "VAR:M1 ___input_dim____",
 "VAR:M2 first dimension is ___input_dim____",
 "VAR:M2 ___output_dim____"
 ]
}
END_PROPS
"""


class MatMul(torch.nn.Module):
    def __init__(self, 
        ___input_dim____ = 10, 
        ___output_dim____ = 10, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(MatMul, self).__init__()
        self.device = ___device____
        self.dtype = ___dtype____

    def forward(self, X):
        x1 = X['M1']
        x2 = X['M2']
        assert x1.dim() == 2 and x2.dim() == 2, "The tensors to be multiplied must be 2D"
        assert x1.shape[-1] == x2.shape[0], "The inner dimensions of the tensors must be the same for matrix multiplication"
        res = {"Y": torch.matmul(x1, x2)}
        return res
