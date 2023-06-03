import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:H' ___hidden_size____"
 ],
 "IN": [
 "VAR:X ___input_size____",
 "VAR:H ___hidden_size____"
 ]
}
END_PROPS
"""


class GRUCell(torch.nn.Module):
    """
    A gated recurrent unit (GRU) cell

    The GRU cell computes the following function:

    r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
    z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
    n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))
    h' = (1 - z) * n + z * h

    Where:
    - σ is the sigmoid function
    - ∗ is the Hadamard (element-wise) product.

    Parameters:
    ___input_size____ (int) – The number of expected features in the input x
    ___hidden_size____ (int) – The number of features in the hidden state h
    ___bias____ (bool) – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    ___device____ (str) – The device to run on. Default: 'cpu'
    ___dtype____ (torch.dtype) – The data type to use. Default: torch.float32

    Inputs: VAR_INPUT, VAR_HIDDEN
    - VAR_INPUT : tensor containing input features
    - VAR_HIDDEN : tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

    Outputs: VAR_H'
    - VAR_H' : tensor containing the next hidden state for each element in the batch
    """
    def __init__(self, 
        ___input_size____ = 10, 
        ___hidden_size____  = 10, 
        ___bias____ = True, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(GRUCell, self).__init__()
        self.gru_cell = torch.nn.GRUCell(___input_size____, 
                                ___hidden_size____, 
                                bias=___bias____, 
                                device=___device____, 
                                dtype=___dtype____
                                )
        self.___hidden_size____ = ___hidden_size____
        self.___device____ = ___device____

    def forward(self, X):
        x = X['X']
        if len(x.shape)==2:
            shape  = (x.shape[0], self.___hidden_size____)
        else:
            shape = self.___hidden_size____
        hidden = X.get('H', torch.zeros(shape).to(self.___device____))
        h_next = {"H'":self.gru_cell(x, hidden)}
        return h_next
