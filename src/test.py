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
    {'id' : 'Linear_1', 'inputs':{'X':[-1, 'X']}},
    {'id' : 'Linear_2', 'inputs':{'X':[2, 'Y']}},
    {'id' : 'ReLU1', 'inputs':{'X':[0, 'Y']}},
    {'id' : 'ReLU2', 'inputs':{'X':[0, 'Y']}},
    {'id' : 'ReLU2', 'inputs':{'X':[1, 'Y']}},
    {'id' : 'Add_1', 'inputs' :{'X1': [3,'Y'],'X2': [4,'Y']}},
    {'id' : 'Linear_3', 'inputs':{'X':[5, 'Y']}},
    {'id' : 'ReLU3', 'inputs':{'X':[6, 'Y']},}
]

output = { # Model outputs
    "Y" : {"run_id" :7, "variable": "Y"},
}

graph = genetic.Graph(models_init,runs_init,output)
#graph.solve_for_run_order()
graph.reorder_runs()
print((graph.nodes,graph.runs,graph.outputs))

gen = genetic.geneticNetwork((graph,0),10,1,0.1,0.1,1)
graph = gen.remove_run(graph, 3)
print((graph.nodes,graph.runs,graph.outputs))