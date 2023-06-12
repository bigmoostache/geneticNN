import torch
from src.basic_templates import Transpose
from src.basic_templates import MatMul
from src.basic_templates import Softmax
from src.basic_templates import Scale

"""
BEGIN_PROPS
{
 "IN": [
 "VAR:V first dimension is ___input_dim_MATMUL2____",
 "VAR:Q ___input_dim_MATMUL____",
 "VAR:K ___input_dim_MATMUL____",
 "VAR:V ___output_dim_MATMUL2____"
 ],
 "OUT": [
 "VAR:Y ___output_dim_MATMUL2____"
 ]
}
END_PROPS
"""

class ATTENTION(torch.nn.Module):
	def __init__(self, 
		___device_MATMUL2____ = 'cpu',
		___dim_Softmax____ = None,
		___dtype_MATMUL2____ = torch.float32,
		___input_dim_MATMUL2____ = 10,
		___input_dim_MATMUL____ = 10,
		___output_dim_MATMUL2____ = 10,
		___scale_Scale____ = 1.0,
		):
		super(ATTENTION, self).__init__()
		# Initializing model Transpose
		self.model_Transpose = Transpose(___input_dim____ = ___input_dim_MATMUL____, ___output_dim____ = ___input_dim_MATMUL2____, )
		# Initializing model MATMUL
		self.model_MATMUL = MatMul(___input_dim____ = ___input_dim_MATMUL____, ___dtype____ = ___dtype_MATMUL2____, ___device____ = ___device_MATMUL2____, ___output_dim____ = ___input_dim_MATMUL2____, )
		# Initializing model Scale
		self.model_Scale = Scale(___input_dim____ = ___input_dim_MATMUL2____, ___dtype____ = ___dtype_MATMUL2____, ___scale____ = ___scale_Scale____, ___device____ = ___device_MATMUL2____, )
		# Initializing model Softmax
		self.model_Softmax = Softmax(___device____ = ___device_MATMUL2____, ___dtype____ = ___dtype_MATMUL2____, ___dim____ = ___dim_Softmax____, ___input_dim____ = ___input_dim_MATMUL2____, )
		# Initializing model MATMUL2
		self.model_MATMUL2 = MatMul(___input_dim____ = ___input_dim_MATMUL2____, ___dtype____ = ___dtype_MATMUL2____, ___device____ = ___device_MATMUL2____, ___output_dim____ = ___output_dim_MATMUL2____, )
	
	def forward(self, X):
		# Sub-model run 0
		Z = {
		 "X" : X["K"],
		}
		model_output_Transpose = self.model_Transpose(Z)
		# Sub-model run 1
		Z = {
		 "M1" : X["Q"],
		 "M2" : model_output_Transpose["Y"],
		}
		model_output_MATMUL = self.model_MATMUL(Z)
		# Sub-model run 2
		Z = {
		 "X" : model_output_MATMUL["Y"],
		}
		model_output_Scale = self.model_Scale(Z)
		# Sub-model run 3
		Z = {
		 "X" : model_output_Scale["Y"],
		}
		model_output_Softmax = self.model_Softmax(Z)
		# Sub-model run 4
		Z = {
		 "M1" : model_output_Softmax["Y"],
		 "M2" : X["V"],
		}
		model_output_MATMUL2 = self.model_MATMUL2(Z)
		# Aggregating results
		RESULT = {}
		RESULT["Y"] = model_output_MATMUL2["Y"]
		return RESULT