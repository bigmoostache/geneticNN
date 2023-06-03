import torch

"""
BEGIN_PROPS
{
 "OUT": [
  "VAR:H ___hidden_size____",
  "VAR:C ___hidden_size____"
 ],
 "IN": [
  "VAR:X ___input_size____",
  "VAR:H ___hidden_size____",
  "VAR:C ___hidden_size____"
 ]
}
END_PROPS
"""

class LSTMCell(torch.nn.Module):
    def __init__(self, 
        ___input_size____ = 10, 
        ___hidden_size____  = 10, 
        ___bias____ = True, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(LSTMCell, self).__init__()
        self.lstmcell = torch.nn.LSTMCell(___input_size____, 
                                ___hidden_size____, 
                                bias=___bias____, 
                                device=___device____, 
                                dtype=___dtype____
                                )
        self.dev = ___device____
        self.hid = ___hidden_size____

    def forward(self, X):
        input_x = X['x']
        h = X.get('H', torch.zeros(self.hid, device = self.dev))
        c = X.get('C', torch.zeros(self.hid, device = self.dev))
        h_1, c_1 = self.lstmcell(input_x, (h, c))
        res = {"H": h_1, "C": c_1}
        return res
