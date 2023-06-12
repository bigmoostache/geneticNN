
import src.templates as templates
import src.basic_templates as basic_templates
def get_model(model_name):
    
    if model_name == "Linear":
        return basic_templates.Linear.Linear()
    if model_name == "ReLU":
        return basic_templates.ReLU.ReLU()
    if model_name == "Scale":
        return basic_templates.Scale.Scale()
    if model_name == "Sigmoid":
        return basic_templates.Sigmoid.Sigmoid()
    if model_name == "Softmax":
        return basic_templates.Softmax.Softmax()
    if model_name == "Transpose":
        return basic_templates.Transpose.Transpose()
    if model_name == "Multiply":
        return basic_templates.Multiply.Multiply()
    if model_name == "MatMul":
        return basic_templates.MatMul.MatMul()
    if model_name == "LSTMCell":
        return basic_templates.LSTMCell.LSTMCell()
    if model_name == "LayerNorm":
        return basic_templates.LayerNorm.LayerNorm()
    if model_name == "GRUCell":
        return basic_templates.GRUCell.GRUCell()
    if model_name == "Embedding":
        return basic_templates.Embedding.Embedding()
    if model_name == "Dropout":
        return basic_templates.Dropout.Dropout()
    if model_name == "BatchNorm1D":
        return basic_templates.BatchNorm1D.BatchNorm1D()
    if model_name == "Add":
        return basic_templates.Add.Add()
    if model_name == "__pycache__":
        return basic_templates.__pycache__.__pycache__()
    if model_name == "__init__":
        return basic_templates.__init__.__init__()
    if model_name == "__pycache__":
        return templates.__pycache__.__pycache__()
    if model_name == "ATTENTION":
        return templates.ATTENTION.ATTENTION()
    if model_name == "MLP":
        return templates.MLP.MLP()
    if model_name == "RNN":
        return templates.RNN.RNN()
    if model_name == "RNN_v2":
        return templates.RNN_v2.RNN_v2()
    if model_name == "model_00":
        return templates.model_00.model_00()
    if model_name == "model_01":
        return templates.model_01.model_01()
    if model_name == "__init__":
        return templates.__init__.__init__()