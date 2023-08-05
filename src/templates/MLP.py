from src.basic_templates import Linear
import torch
from src.basic_templates import ReLU

"""
BEGIN_PROPS
{
 "IN": [
 "VAR:X ___input_dim_Linear_1____"
 ],
 "OUT": [
 "VAR:Y ___input_dim_ReLU3____"
 ]
}
END_PROPS
"""

class MLP(torch.nn.Module):
	def __init__(self, 
		___bias_Linear_1____ = True,
		___bias_Linear_2____ = True,
		___bias_Linear_3____ = True,
		___device_Linear_1____ = 'cpu',
		___dtype_Linear_1____ = torch.float32,
		___input_dim_Linear_1____ = 10,
		___input_dim_Linear_2____ = 10,
		___input_dim_Linear_3____ = 10,
		___input_dim_ReLU3____ = 10,
		):
		super(MLP, self).__init__()
		# Initializing model Linear_1
		self.model_Linear_1 = Linear.Linear(___input_dim____ = ___input_dim_Linear_1____, ___bias____ = ___bias_Linear_1____, ___dtype____ = ___dtype_Linear_1____, ___output_dim____ = ___input_dim_Linear_2____, ___device____ = ___device_Linear_1____, )
		# Initializing model Linear_2
		self.model_Linear_2 = Linear.Linear(___input_dim____ = ___input_dim_Linear_2____, ___bias____ = ___bias_Linear_2____, ___dtype____ = ___dtype_Linear_1____, ___output_dim____ = ___input_dim_Linear_3____, ___device____ = ___device_Linear_1____, )
		# Initializing model Linear_3
		self.model_Linear_3 = Linear.Linear(___input_dim____ = ___input_dim_Linear_3____, ___bias____ = ___bias_Linear_3____, ___dtype____ = ___dtype_Linear_1____, ___output_dim____ = ___input_dim_ReLU3____, ___device____ = ___device_Linear_1____, )
		# Initializing model ReLU1
		self.model_ReLU1 = ReLU.ReLU(___input_dim____ = ___input_dim_Linear_2____, ___dtype____ = ___dtype_Linear_1____, ___device____ = ___device_Linear_1____, )
		# Initializing model ReLU2
		self.model_ReLU2 = ReLU.ReLU(___input_dim____ = ___input_dim_Linear_3____, ___dtype____ = ___dtype_Linear_1____, ___device____ = ___device_Linear_1____, )
		# Initializing model ReLU3
		self.model_ReLU3 = ReLU.ReLU(___input_dim____ = ___input_dim_ReLU3____, ___dtype____ = ___dtype_Linear_1____, ___device____ = ___device_Linear_1____, )
	
	def forward(self, X):
		# Sub-model run 0
		Z = {
		 "X" : X["X"],
		}
		model_output_0 = self.model_Linear_1(Z)
		# Sub-model run 1
		Z = {
		 "X" : model_output_0["Y"],
		}
		model_output_1 = self.model_ReLU1(Z)
		# Sub-model run 2
		Z = {
		 "X" : model_output_1["Y"],
		}
		model_output_2 = self.model_Linear_2(Z)
		# Sub-model run 3
		Z = {
		 "X" : model_output_2["Y"],
		}
		model_output_3 = self.model_ReLU2(Z)
		# Sub-model run 4
		Z = {
		 "X" : model_output_3["Y"],
		}
		model_output_4 = self.model_Linear_3(Z)
		# Sub-model run 5
		Z = {
		 "X" : model_output_4["Y"],
		}
		model_output_5 = self.model_ReLU3(Z)
		# Aggregating results
		RESULT = {}
		RESULT["Y"] = model_output_5["Y"]
		return RESULT