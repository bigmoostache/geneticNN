import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___num_features____"
 ],
 "IN": [
 "VAR:X ___num_features____"
 ]
}
END_PROPS
"""

class BatchNorm1d(torch.nn.Module):
    """
    Applies Batch Normalization over a 2D or 3D input as described in the paper 
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
    
    Parameters:
    num_features (int) – number of features or channels of the input
    eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
    momentum (float) – the value used for the running_mean and running_var computation. Default: 0.1
    affine (bool) – a boolean value that when set to True, this module has learnable affine parameters. Default: True
    track_running_stats (bool) – a boolean value that when set to True, this module tracks the running mean and variance. Default: True
    """
    def __init__(self, 
        ___num_features____ = 10, 
        ___eps____ = 1e-05, 
        ___momentum____ = 0.1, 
        ___affine____ = True, 
        ___track_running_stats____ = True, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32,
        ):
        super(BatchNorm1d, self).__init__()
        self.batch_norm = torch.nn.BatchNorm1d(___num_features____, 
                                eps=___eps____, 
                                momentum=___momentum____, 
                                affine=___affine____, 
                                track_running_stats=___track_running_stats____, 
                                device=___device____, 
                                dtype=___dtype____
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.batch_norm(x)}
        return res
