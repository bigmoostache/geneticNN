import genetic

models_init = { # For now, only the template is required
    'Linear_1' : {"template" : "Linear"}, 
    'Linear_2' : {"template" : "Linear"}, 
    'Linear_3' : {"template" : "Linear"},
    'Add_1' : {"template" : "Add"}, 
    'ReLU1' : {"template" : "ReLU"}, 
    'ReLU2' : {"template" : "ReLU"}, 
    'ReLU3' : {"template" : "ReLU"}, 
}

runs_init = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs':{'X':[0, 'X']}},
    {'id' : 'Linear_2', 'inputs':{'X':['ReLU1', 'Y']}},
    {'id' : 'ReLU1', 'inputs':{'X':['Linear_1', 'Y']}},
    {'id' : 'ReLU2', 'inputs':{'X':['Linear_2', 'Y']}},
    {'id' : 'Add_1', 'inputs' :{'X1': ['ReLU1','Y'], 'X2' : ['ReLU2','Y']}},
    {'id' : 'Linear_3', 'inputs':{'X':['Add_1', 'Y']}},
    {'id' : 'ReLU3', 'inputs':{'X':['Linear_3', 'Y']}}
]

output = { # Model outputs
    "Y" : {"model_id" :'ReLU3', "variable": "Y"},
}

graph = genetic.Graph(models_init,runs_init,output)
print((graph.nodes,graph.runs,graph.outputs))
gen = genetic.geneticNetwork((graph,0),10,1,0.1,0.1,1)
graph = gen.remove_node(graph,'ReLU2')
print((graph.nodes,graph.runs,graph.outputs))