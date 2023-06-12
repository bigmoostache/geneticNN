import torch

"""
BEGIN_PROPS
{
 "OUT": [
 "VAR:Y ___embedding_dim____"
 ],
 "IN": [
 "VAR:X ___some_dimension____"
 ]
}
END_PROPS
"""


class Embedding(torch.nn.Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices. 
    The input to the module is a list of indices, and the output is the corresponding word embeddings.
    """

    def __init__(self, 
                 ___num_embeddings____ = 100, 
                 ___embedding_dim____ = 50, 
                 ___padding_idx____ = None, 
                 ___max_norm____ = None, 
                 ___norm_type____ = 2.0, 
                 ___scale_grad_by_freq____ = False, 
                 ___sparse____ = False, 
                 ___device____ = 'cpu', 
                 ___dtype____ = torch.float32,
                 ___some_dimension____ = 10,
                ):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(___num_embeddings____, 
                                             ___embedding_dim____, 
                                             padding_idx=___padding_idx____, 
                                             max_norm=___max_norm____, 
                                             norm_type=___norm_type____, 
                                             scale_grad_by_freq=___scale_grad_by_freq____, 
                                             sparse=___sparse____, 
                                             device=___device____, 
                                             dtype=___dtype____
                                            )

    def forward(self, X):
        x = X['X']
        res = {"Y":self.embedding(x)}
        return res
