from genetic import geneticNetwork

models_init = { # For now, only the template is required
    'Linear_1' : {"template" : "Linear"}, 
    'Linear_2' : {"template" : "Linear"}, 
    'Linear_3' : {"template" : "Linear"}, 
    'ReLU1' : {"template" : "ReLU"}, 
    'ReLU2' : {"template" : "ReLU"}, 
    'ReLU3' : {"template" : "ReLU"}, 
}

runs_init = [ # Does not have to be sorted, it willbe sorted by myRealWriter
    {'id' : 'Linear_1', 'inputs':{'X':[0, 'X']}},
    {'id' : 'Linear_2', 'inputs':{'X':['ReLU1', 'Y']}},
    {'id' : 'Linear_3', 'inputs':{'X':['ReLU2', 'Y']}},
    {'id' : 'ReLU1', 'inputs':{'X':['Linear_1', 'Y']}},
    {'id' : 'ReLU2', 'inputs':{'X':['Linear_2', 'Y']}},
    {'id' : 'ReLU3', 'inputs':{'X':['Linear_3', 'Y']}}
]

output = { # Model outputs
    "Y" : {"model_id" :'ReLU3', "variable": "Y"},
}

gen = geneticNetwork((models_init,runs_init,output,0),10,1,0.1,0.1,1)
print (gen.random_decrease((models_init,runs_init,output,0)))