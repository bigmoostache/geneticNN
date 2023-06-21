import random,os
from graph import Graph

script_directory = os.path.dirname(os.path.abspath(__file__))
path_templates = os.path.join(script_directory, 'templates')
path_basic_templates = os.path.join(script_directory, 'basic_templates')

def get_possibles_models():
    model_list = []
    for model_file in os.listdir(path_basic_templates):
        model_name = model_file.split(".")[0]
        model_list.append(model_name)
    return model_list

def is_double_input(model_name):
    return model_name == 'Add' or model_name =='Multiply' or model_name == 'MatMul' or model_name == 'LSTMCell'

def get_previous_nodes(edges,node):
    previous_nodes = []
    for e in edges:
        if e["id"] == node:
            for input in list(e['inputs']):
                previous_nodes.append(e['inputs'][input][0])
    return previous_nodes

def get_next_nodes(edges,node):
    next_nodes = []
    for e in edges:
        for input in list(e['inputs']):
            if input[0] == node:
                next_nodes.append(e['id'])
    return next_nodes

def can_decrease(nodes,edges,node_to_verify):
    node_type = nodes[node_to_verify]['template']
    if node_type=="Add" or node_type=="Embedding":
        return False
    else:
        return True


class geneticNetwork():
    def __init__(self,initial_network,generation_size, n_best_to_save, network_enlargment_p,network_decrease_p, parameter_std):
        self.init = initial_network
        self.n_best_to_save = n_best_to_save
        self.generation_size = generation_size
        self.network_enlargment_p = network_enlargment_p
        self.network_decrease_p = network_decrease_p
        self.parameter_std = parameter_std
        self.model_list = get_possibles_models()

        initial_nets = [self.init for _ in range(self.n_best_to_save)]
        #self.population = self.diversify(initial_nets)


    def move_one_generation(self, input_func, reward_func):
        best_nets = self.get_best_nets()
        self.population = self.diversify(best_nets)

    def diversify(self, best_nets):
        population = []
        for _ in range(self.generation_size):
            net = best_nets[random.randint(0,self.n_best_to_save - 1)]
            if random.random() < self.network_decrease_p:
                net = self.random_decrease(net)
            if random.random() < self.network_enlargment_p:
                net = self.random_enlargment(net)
            net = self.random_update_parameters(net)
            population.append(net)
        return population

    def get_best_nets(self):
        reward_index = 3
        individuals_with_rewards = sorted(self.population, key=lambda x: x[reward_index], reverse=True)[0:self.n_best_to_save]
        return individuals_with_rewards
    
    def random_decrease(self,net):
        repeat = True
        graph = net[0]
        n_nodes = len(graph.nodes)
        while(repeat):
            random_node = random.choice(list(graph.nodes.keys()))
            if can_decrease(graph.nodes,graph.runs,random_node):
                graph = self.remove_node(graph,random_node)
                repeat = False
        return graph


    #Removes the node from a network.
    #For now, only 1 to 1 modules can be removed,
    #and 2 inputs/1 output networks are removed if one
    #input is the parent of another (we can still have parent and child networks
    # be inputs of a 2 inputs net if the parent net is branched with an identity module
    def remove_node(self,graph,node_to_remove):
        #next_nodes = graph.get_children_run(node_to_remove)
        #print(next_nodes)
        previous_node = [0,graph.inputs[0]]
        runs_to_delete=[]
        for i,e in enumerate(graph.runs):
            if (e['id'] == node_to_remove):
                previous_node = e['inputs'][list(e['inputs'])[0]][0]
                runs_to_delete.append(i)
                
            else:
                for input in e['inputs']:
                    if e['inputs'][input][0] == node_to_remove:
                        e['inputs'][input][0] = previous_node

        for i in runs_to_delete:
            del graph.runs[i]        
        #Remove nodes with 2 inputs if both inputs are the same or 
        # if one of the inputs is a parent of the other (without an identity node)
        nodes_to_remove = []
        for node in graph.nodes:
            if is_double_input(graph.nodes[node]['template']):
                node_runs = graph.get_parents_runs_wt_indices(node)
                max_run_l = 0
                runs_to_delete=[]
                for index,run_ in enumerate(node_runs):
                    run = run_[1]
                    keys_to_delete = []
                    for i1 in run['inputs']:
                        for i2 in run['inputs']:
                            if i1 != i2:
                                if graph.is_parent(run['inputs'][i1][0],run['inputs'][i2][0]):
                                    keys_to_delete.append(i1)
                        
                    for key in keys_to_delete:
                        del run['inputs'][key]

                    max_run_l = max(max_run_l,len(run['inputs']))
                    if len(run['inputs']) == 1:
                        print(run)
                        previous_node = run['inputs'][list(run['inputs'])[0]]
                        it_range = len(graph.runs) - run_[0]
                        if(index < (len(node_runs) - 1)):
                            it_range = node_runs[index+1][0] - node_runs[index][0]
                        for j in range(it_range):
                            trun = graph.runs[j+run_[0]]
                            for input in trun['inputs']:
                                if trun['inputs'][input][0] == node:
                                    trun['inputs'][input] = previous_node
                        runs_to_delete.append(run_[0])
                
                for i in runs_to_delete:
                    del graph.runs[i]

                if len(node_runs) <=1:
                    nodes_to_remove.append(node)
        del graph.nodes[node_to_remove]
        for i in nodes_to_remove:
            del graph.nodes[i]

        return graph
    