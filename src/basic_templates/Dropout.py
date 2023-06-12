import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___dim____"
 ],
 "IN": [
 "VAR:X ___dim____"
 ]
}
END_PROPS
"""


class Dropout(torch.nn.Module):
    def __init__(self, 
        ___dim____ = 10, 
        ___p____ = 0.5
        ):
        super(Dropout, self).__init__()
        self.dropout = torch.nn.Dropout(___p____, 
                                inplace=False
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.dropout(x)}
        return res
