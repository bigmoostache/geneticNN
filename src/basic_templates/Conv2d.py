"""
BEGIN_PROPS
{
"variables": {
    "X": {
        "io" : "in",
        "dim": "3",
        "type" : "float"
        },
    "Y": {
        "io" : "out",
        "dim": "3",
        "type": "float"
        }
    },
"parameters" : {
    "bias": {"type" : "bool"},
    "kernel_0": {"type": "int"},
    "kernel_1": {"type": "int"},
    "stride_0": {"type": "int"},
    "stride_1": {"type": "int"},
    "padding_0": {"type": "int"},
    "padding_1": {"type": "int"},
    "dilation_0": {"type": "int"},
    "dilation_1": {"type": "int"},
    "in_channels": {"type": "int"},
    "out_channels": {"type": "int"},
},
"constraints" : [
    ["equality",["_X_0", "in_channels"]],
    ["equality",["_Y_0", "out_channels"]],
    ["symbolic",["_X_1","_Y_1","kernel_0","stride_0","padding_0","dilation_0"],"( $1 + 1 ) * $3  = $0 + 2 * $4 - $5 * ( $2 - 1 )"],
    ["symbolic",["_X_2","_Y_2","kernel_1","stride_1","padding_1","dilation_1"],"( $1 + 1 ) * $3  = $0 + 2 * $4 - $5 * ( $2 - 1 )"]

]
}
END_PROPS
"""

import torch


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_0,
                 kernel_1,
                 stride_0=1,
                 stride_1=1,
                 padding_0=0,
                 padding_1=0,
                 dilation_0=0,
                 bias=True,
                 ___device____='cpu',
                 ___dtype____=torch.float32
                 ):
        super(Conv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=(kernel_0, kernel_1),
                                    padding=(padding_0, padding_1),
                                    stride=(stride_0, stride_1),
                                    bias=bias,
                                    device=___device____,
                                    dtype=___dtype____
                                    )

    def forward(self, X):
        x = X['X']
        res = {"Y": self.conv(x)}
        return res
