import random,os

script_directory = os.path.dirname(os.path.abspath(__file__))
path_templates = os.path.join(script_directory, 'templates')
path_basic_templates = os.path.join(script_directory, 'basic_templates')

def get_possibles_models():
    model_list = []
    for model_file in os.listdir(path_basic_templates):
        model_name = model_file.split(".")[0]
        model_list.append(model_name)
    return model_list

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
        n_nodes = len(net[0])
        while(repeat):
            random_node = random.choice(list(net[0].keys()))
            if can_decrease(net[0],net[1],random_node):
                next_nodes = get_next_nodes(net[1],random_node)
                previous_node = get_previous_nodes(net[1],random_node)[0]
                node_index = 0
                for i,e in enumerate(net[1]):
                    if (e['id'] == random_node):
                        node_index = i
                    else:
                        for j,input in enumerate(list(e['inputs'])):
                            if e['inputs'][input][0] == random_node:
                                net[1][i]['inputs'][input][0] = previous_node
                del net[1][node_index]
                del net[0][random_node]
                repeat = False
            #Remove Add node if both inputs of Add are the same
        return net