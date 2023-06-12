import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___normalized_shape____"
 ],
 "IN": [
 "VAR:X ___normalized_shape____"
 ]
}
END_PROPS
"""

class LayerNorm(torch.nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization.
    This layer uses statistics computed from input data in both training and evaluation modes.

    Parameters:
    ___normalized_shape____ (int or list or torch.Size) – input shape from an expected input of size.

    ___eps____ (float) – a value added to the denominator for numerical stability. Default: 1e-5

    ___elementwise_affine____ (bool) – a boolean value that when set to True, 
    this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.
    """
    def __init__(self, 
        ___normalized_shape____ = 10, 
        ___eps____ = 1e-5, 
        ___elementwise_affine____ = True, 
        ___device____ = 'cpu', 
        ___dtype____ = torch.float32
        ):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(___normalized_shape____, 
                                eps = ___eps____, 
                                elementwise_affine = ___elementwise_affine____, 
                                device = ___device____, 
                                dtype = ___dtype____
                                )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.layer_norm(x)}
        return res
