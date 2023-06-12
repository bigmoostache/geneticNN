from src.basic_templates import Linear
import torch

"""
BEGIN_PROPS
{
 "IN": [
 "VAR:X ___input_dim_Linear_1____"
 ],
 "OUT": [
 "VAR:Y ___input_dim_Linear_1____"
 ]
}
END_PROPS
"""

class RNN_v2(torch.nn.Module):
	def __init__(self, 
		___bias_Linear_1____ = True,
		___device_Linear_1____ = 'cpu',
		___dtype_Linear_1____ = torch.float32,
		___input_dim_Linear_1____ = 10,
		):
		super(RNN_v2, self).__init__()
		# Initializing model Linear_1
		self.model_Linear_1 = Linear(___output_dim____ = ___input_dim_Linear_1____, ___dtype____ = ___dtype_Linear_1____, ___input_dim____ = ___input_dim_Linear_1____, ___device____ = ___device_Linear_1____, ___bias____ = ___bias_Linear_1____, )
	
	def forward(self, X):
		# Sub-model run 0
		Z = {
		 "X" : X["X"],
		}
		model_output_Linear_1 = self.model_Linear_1(Z)
		# Sub-model run 1
		Z = {
		 "X" : model_output_Linear_1["Y"],
		}
		model_output_Linear_1 = self.model_Linear_1(Z)
		# Aggregating results
		RESULT = {}
		RESULT["Y"] = model_output_Linear_1["Y"]
		return RESULT