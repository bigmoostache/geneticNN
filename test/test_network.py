import Base.Network.Layer
import src.Base.modelskeleton as base

# load test libraries
import unittest


class TestNeuralNetwork(unittest.TestCase):
    def test_init(self):
        input_variables = [base.Variable("input", 2, "in", float)]
        input_model = base.LayerModel("input", base.InputTemplate())
        input_layer = Base.Network.Layer.Layer(input_model)

        output_variables = [base.Variable("output", 2, "out", float)]
        output_model = base.LayerModel("output", base.OutputTemplate())
        output_layer = Base.Network.Layer.Layer(output_model)

        input_model.attach_variables(input_variables, "in")
        output_model.attach_variables(output_variables, "out")
